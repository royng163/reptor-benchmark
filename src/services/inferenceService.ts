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
          enableSmoothing: true,
        });
        break;
      }
      case "movenet": {
        this.detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        });
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

    // Warm up detector once (compile shaders)
    if (this.detector) {
      const warm = tf.zeros([256, 256, 3]) as tf.Tensor3D;
      try {
        await this.detector.estimatePoses(warm, { flipHorizontal: true, maxPoses: 1 });
      } finally {
        warm.dispose();
        await tf.nextFrame();
      }
    }
    this.currentModelId = modelId;
  }

  async runInference(input: tf.Tensor3D): Promise<any> {
    const t0 = performance.now();

    if (this.currentModelId === "yolo") {
      const res = await this.runModelInference(input);
      const t1 = performance.now();
      this.totalInferenceTime += t1 - t0;
      this.frameCount += 1;
      const elapsed = (performance.now() - this.sessionStartTime) / 1000;
      const fps = elapsed > 0 ? this.frameCount / elapsed : 0;
      const avgInferenceTime = this.totalInferenceTime / this.frameCount;
      return { ...res, fps, avgInferenceTime };
    }

    const [srcH, srcW] = input.shape;
    const poses = await this.detector!.estimatePoses(input, { flipHorizontal: true, maxPoses: 1 });

    // Debug: log keypoints
    console.log("Keypoints:", poses);

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

  private async runModelInference(input: tf.Tensor3D): Promise<PoseResult> {
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

      const boxes = tf.tidy(() => {
        const w = tf.slice(transpose, [0, 0, 2], [-1, -1, 1]);
        const h = tf.slice(transpose, [0, 0, 3], [-1, -1, 1]);
        const x1 = tf.sub(tf.slice(transpose, [0, 0, 0], [-1, -1, 1]), tf.div(w, 2));
        const y1 = tf.sub(tf.slice(transpose, [0, 0, 1], [-1, -1, 1]), tf.div(h, 2));
        return tf.squeeze(tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2));
      }) as tf.Tensor2D;

      const scores = tf.squeeze(tf.slice(transpose, [0, 0, 4], [-1, -1, 1])) as tf.Tensor1D;
      const landmarks = tf.squeeze(tf.slice(transpose, [0, 0, 5], [-1, -1, -1])) as tf.Tensor2D;
      const selected = await tf.image.nonMaxSuppressionAsync(boxes, scores, 50, 0.45, 0.3);
      const idxArr = await selected.array();
      const topIdx = idxArr[0];
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
