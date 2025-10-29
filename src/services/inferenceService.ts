import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import { computeLetterbox, mapFromLetterbox } from "../utils/letterbox";

export type ModelId = "movenet" | "blazepose" | "yolo";

export interface Keypoint {
  x: number;
  y: number;
  z?: number;
  visibility: number; // [0, 1]
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

type TFLiteTensor = { data: Float32Array | Uint8Array | Int32Array; dims: number[] };

interface RGBAFrame {
  rgba: Uint8Array; // RGBA8888
  width: number;
  height: number;
  modelInput: { w: number; h: number };
  preprocessed?: {
    tensor: Float32Array | Uint8Array | Int32Array; // e.g. Float32Array if dataType='float32' in the plugin
    dims: number[]; // e.g. [1, INPUT, INPUT, 3]
    normalized?: boolean; // true if already scaled to 0..1
  };
}

export class InferenceService {
  private model?: TensorflowModel;
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

    // Dispose of the old model to free up memory
    this.model = undefined;

    switch (modelId) {
      case "blazepose": {
        this.model = await loadTensorflowModel(require("../../assets/models/blazepose/pose_landmarks_detector.tflite"));
        break;
      }
      case "movenet": {
        this.model = await loadTensorflowModel(require("../../assets/models/movenet/movenet.tflite"));
        break;
      }
      case "yolo": {
        this.model = await loadTensorflowModel(require("../../assets/models/yolo/yolo11n-pose_float16.tflite"));
        break;
      }
    }

    this.currentModelId = modelId;
  }

  recordInference(durationMs: number): { fps: number; avg: number } {
    this.totalInferenceTime += durationMs;
    this.frameCount += 1;
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const elapsed = this.sessionStartTime > 0 ? (now - this.sessionStartTime) / 1000 : 0;
    const fps = elapsed > 0 ? this.frameCount / elapsed : 0;
    const avg = this.totalInferenceTime / Math.max(1, this.frameCount);
    return { fps, avg };
  }
}
