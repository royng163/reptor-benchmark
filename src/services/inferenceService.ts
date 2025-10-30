import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as poseDetection from "@tensorflow-models/pose-detection";
import { computeLetterbox, mapFromLetterbox } from "../utils/letterbox";

export type ModelId = "movenet" | "blazepose" | "yolo";
export type MediaType =
  | { type: "tensor"; tensor: tf.Tensor3D }
  | { type: "image"; uri: string }
  | { type: "video"; frameUri: string };

export interface Keypoint {
  x: number; // in source pixel space
  y: number;
  z?: number;
  visibility?: number; // [0, 1]
  name?: string;
}

export interface PoseResult {
  keypoints: Keypoint[];
  keypoints3D?: Keypoint[];
  timestamp?: number;
  fps?: number;
  avgInferenceTime?: number;
  srcWidth?: number;
  srcHeight?: number;
}

export class InferenceService {
  private detector?: poseDetection.PoseDetector;
  private model?: tf.GraphModel;
  private currentModelId?: ModelId;

  private frameCount = 0;
  private totalInferenceTime = 0;
  private sessionStartTime = 0;

  startMonitoring() {
    this.frameCount = 0;
    this.totalInferenceTime = 0;
    this.sessionStartTime = performance.now();
  }

  stopMonitoring() {
    this.sessionStartTime = 0;
  }

  async loadModel(modelId: ModelId) {
    if (this.currentModelId === modelId) {
      return; // Already loaded
    }

    await tf.setBackend("rn-webgl");
    await tf.ready();
    console.log(`TF.js backend: ${tf.getBackend()}`);

    // Dispose of the old model to free up memory
    this.model?.dispose();
    this.detector?.dispose();

    switch (modelId) {
      case "blazepose": {
        this.detector = await poseDetection.createDetector(poseDetection.SupportedModels.BlazePose, {
          runtime: "tfjs",
          modelType: "lite",
        });
        break;
      }
      case "movenet": {
        // this.detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
        //   modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        // });
        const modelJson = require(`../../assets/models/movenet/model`);
        const modelWeights = [
          require(`../../assets/models/movenet/group1-shard1of2.bin`),
          require(`../../assets/models/movenet/group1-shard2of2.bin`),
        ];

        this.model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
        break;
      }
      case "yolo": {
        const modelJson = require(`../../assets/models/yolo/model`);
        const modelWeights = [
          require(`../../assets/models/yolo/group1-shard1of3.bin`),
          require(`../../assets/models/yolo/group1-shard2of3.bin`),
          require(`../../assets/models/yolo/group1-shard3of3.bin`),
        ];

        this.model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
        break;
      }
    }

    this.currentModelId = modelId;
  }

  async runInference(input: tf.Tensor3D): Promise<any> {
    const t0 = performance.now();

    let res: PoseResult;
    if (this.currentModelId === "yolo") {
      res = await this.runYolo11Inference(input);
    } else if (this.currentModelId === "movenet") {
      res = await this.runMovenetInference(input);
    } else {
      const [srcH, srcW] = input.shape;
      const poses = await this.detector!.estimatePoses(input, { flipHorizontal: true, maxPoses: 1 });

      const t1 = performance.now();
      this.totalInferenceTime += t1 - t0;
      this.frameCount += 1;
      const elapsed = (performance.now() - this.sessionStartTime) / 1000;
      const fps = elapsed > 0 ? this.frameCount / elapsed : 0;
      const avgInferenceTime = this.totalInferenceTime / this.frameCount;

      if (!poses || poses.length === 0) {
        return { keypoints: [], keypoints3D: [], fps, avgInferenceTime, srcWidth: srcW, srcHeight: srcH };
      }

      const p = poses[0];
      const keypoints = p.keypoints.map((kp) => ({
        x: kp.x,
        y: kp.y,
        z: kp.z ?? undefined,
        visibility: kp.score,
        name: kp.name,
      }));
      const keypoints3D =
        p.keypoints3D?.map((kp) => ({
          x: kp.x,
          y: kp.y,
          z: kp.z ?? undefined,
          visibility: kp.score,
          name: kp.name,
        })) ?? [];

      return { keypoints, keypoints3D, fps, avgInferenceTime, srcWidth: srcW, srcHeight: srcH };
    }

    const t1 = performance.now();
    this.totalInferenceTime += t1 - t0;
    this.frameCount += 1;
    const elapsed = (performance.now() - this.sessionStartTime) / 1000;
    const fps = elapsed > 0 ? this.frameCount / elapsed : 0;
    const avgInferenceTime = this.totalInferenceTime / this.frameCount;
    return { ...res, fps, avgInferenceTime };
  }

  private async runMovenetInference(input: tf.Tensor3D): Promise<PoseResult> {
    const INPUT_SIZE = 192; // For MoveNet Single-Pose Lightning
    const [srcH, srcW] = input.shape;

    const tensor4d = tf.tidy(() => {
      const resized = tf.image.resizeBilinear(input, [INPUT_SIZE, INPUT_SIZE], true);
      const int32 = resized.toInt();
      return tf.expandDims(int32, 0) as tf.Tensor4D;
    });

    try {
      const out = this.model!.execute(tensor4d) as tf.Tensor;
      const kptsTensor = tf.squeeze(out, [0, 1]); // Shape: [1, 1, 17, 3] -> [17, 3]
      const kptsArr = (await kptsTensor.array()) as number[][];

      const kpts: Keypoint[] = kptsArr.map(([y, x, v]) => ({
        x: x * srcW,
        y: y * srcH,
        visibility: v,
      }));

      kptsTensor.dispose();
      out.dispose();

      return { keypoints: kpts, keypoints3D: [], srcWidth: srcW, srcHeight: srcH };
    } finally {
      tensor4d.dispose();
      await tf.nextFrame();
    }
  }

  private async runYolo11Inference(input: tf.Tensor3D): Promise<PoseResult> {
    const INPUT_SIZE = 640;
    const { tensor: frameTensor, width: srcW, height: srcH, dispose } = await this.prepareInputTensor(input);

    const letter = computeLetterbox(srcW, srcH, INPUT_SIZE);

    const tensor4d = tf.tidy(() => {
      const resized = tf.image.resizeBilinear(frameTensor, [letter.resized.height, letter.resized.width], true);
      const padded = tf.pad(resized, [
        [letter.dy, INPUT_SIZE - letter.resized.height - letter.dy],
        [letter.dx, INPUT_SIZE - letter.resized.width - letter.dx],
        [0, 0],
      ]);
      const normalized = tf.div(padded, 255);
      return tf.expandDims(normalized, 0) as tf.Tensor4D;
    });

    try {
      const out = this.model!.execute(tensor4d) as tf.Tensor;
      const transpose = tf.transpose(out, [0, 2, 1]);

      // Find the prediction with the highest confidence score
      const scores = tf.squeeze(tf.slice(transpose, [0, 0, 4], [-1, -1, 1])) as tf.Tensor1D;
      const topIdxTensor = tf.argMax(scores);
      const topIdx = await topIdxTensor.data();

      // Extract and reshape the keypoints for the best prediction
      const landmarks = tf.squeeze(tf.slice(transpose, [0, 0, 5], [-1, -1, -1])) as tf.Tensor2D;
      const kpTensor = tf.reshape(tf.gather(landmarks, topIdx), [17, 3]);
      const kpArr = (await kpTensor.array()) as number[][];

      const kpts: Keypoint[] = kpArr.map(([x, y, v]) => {
        const normalized = x <= 1 && y <= 1;
        const mapped = mapFromLetterbox(x, y, srcW, srcH, letter, normalized);
        return { x: mapped.x, y: mapped.y, visibility: v };
      });
      return { keypoints: kpts, keypoints3D: [], srcWidth: srcW, srcHeight: srcH };
    } finally {
      tensor4d.dispose();
      dispose();
      await tf.nextFrame();
    }
  }

  private async prepareInputTensor(input: tf.Tensor3D): Promise<{
    tensor: tf.Tensor3D;
    width: number;
    height: number;
    dispose: () => void;
  }> {
    const [height, width] = input.shape;
    return { tensor: input, width, height, dispose: () => undefined };
  }
}
