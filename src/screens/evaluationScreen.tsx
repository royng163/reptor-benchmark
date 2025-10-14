import React, { useState, useEffect, useRef, useCallback } from "react";
import { View, Text, Button, StyleSheet, Modal, Pressable } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Camera, CameraType } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { InferenceService, ModelId, Keypoint } from "../services/inferenceService";

const TensorCamera = cameraWithTensors(Camera);

export default function EvaluationScreen() {
  const [model, setModel] = useState<ModelId>();
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [status, setStatus] = useState("Loading...");
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [cameraViewDimensions, setCameraViewDimensions] = useState({ height: 1, width: 1 });
  const [srcDims, setSrcDims] = useState<{ w: number; h: number }>({ w: 1, h: 1 });
  const [fps, setFps] = useState(0);
  const [avgInferenceTime, setAvgInferenceTime] = useState(0);
  const inferenceService = useRef(new InferenceService()).current;
  const rafId = useRef<number | null>(null);
  const imageIterator = useRef<IterableIterator<tf.Tensor3D> | null>(null);
  const cameraContainerRef = useRef<View | null>(null);
  const [overlayRect, setOverlayRect] = useState({ x: 0, y: 0, width: 0, height: 0 });

  // Pick a light input size per model to reduce work
  const modelInput = React.useMemo(() => {
    switch (model) {
      case "movenet":
        return { w: 192, h: 192 };
      case "blazepose":
        return { w: 256, h: 256 };
      case "yolo":
        return { w: 320, h: 320 }; // smaller than 640 for speed
      default:
        return { w: 256, h: 256 };
    }
  }, [model]);

  const measureCameraInWindow = useCallback(() => {
    if (!cameraContainerRef.current) return;
    cameraContainerRef.current.measureInWindow((x, y, width, height) => {
      setOverlayRect({ x, y, width, height });
      setCameraViewDimensions({ width, height }); // used by renderKeypoints scaling
    });
  }, []);

  useEffect(() => {
    // re-measure on mount
    requestAnimationFrame(measureCameraInWindow);
  }, [measureCameraInWindow]);

  useEffect(() => {
    (async () => {
      await Camera.requestCameraPermissionsAsync();
      setStatus("Ready. Please select a model.");
    })();

    // Cleanup function to cancel animation frame when the component unmounts
    return () => {
      if (rafId.current) {
        cancelAnimationFrame(rafId.current);
      }
    };
  }, []);

  // This effect now controls the entire evaluation loop
  useEffect(() => {
    const loop = async () => {
      if (!isEvaluating || !imageIterator.current) {
        return;
      }

      const nextImageTensor = imageIterator.current.next().value as tf.Tensor3D;

      if (nextImageTensor) {
        const result = await inferenceService.runInference(nextImageTensor);
        tf.dispose(nextImageTensor);

        if (result) {
          setKeypoints(result.keypoints || []);
          setFps(result.fps || 0);
          setAvgInferenceTime(result.avgInferenceTime || 0);
          if (result.srcWidth && result.srcHeight) {
            setSrcDims({ w: result.srcWidth, h: result.srcHeight });
          }
        }
      }

      rafId.current = requestAnimationFrame(loop);
    };

    if (isEvaluating) {
      inferenceService.startMonitoring();
      // Start the loop
      rafId.current = requestAnimationFrame(loop);
    } else {
      inferenceService.stopMonitoring();
      // Stop the loop
      if (rafId.current) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
    }
  }, [isEvaluating]);

  const handleConfigChange = async (newModel: ModelId) => {
    // Stop evaluation before changing the model
    if (isEvaluating) {
      setIsEvaluating(false);
    }
    setModel(newModel);
    setKeypoints([]);
    setStatus(`Loading ${newModel}`);
    await inferenceService.loadModel(newModel);
    setStatus(`Ready to evaluate ${newModel}.`);
  };

  // Use useCallback to prevent this function from being recreated on every render
  const handleTensorCameraStream = useCallback((images: IterableIterator<tf.Tensor3D>) => {
    console.log("Camera stream is ready.");
    imageIterator.current = images;
  }, []);

  const toggleEvaluation = () => {
    if (!model) {
      setStatus("Please load a model first.");
      return;
    }
    if (!imageIterator.current) {
      setStatus("Camera is not ready yet.");
      return;
    }
    console.log(`Toggling evaluation, current state: ${isEvaluating}`);
    setIsEvaluating((prev) => !prev);
  };

  const renderKeypoints = () => {
    if (keypoints.length === 0) return null;

    // Source tensor size (what the model saw)
    const srcW = srcDims.w || modelInput.w;
    const srcH = srcDims.h || modelInput.h;

    const viewW = cameraViewDimensions.width;
    const viewH = cameraViewDimensions.height;

    // Camera preview behaves like "cover" (fills, cropping one axis)
    const scale = Math.max(viewW / srcW, viewH / srcH);
    const offsetX = (viewW - srcW * scale) / 2;
    const offsetY = (viewH - srcH * scale) / 2;

    return keypoints.map((kpt, i) => {
      if (kpt.visibility != null && kpt.visibility < 0.1) return null;
      const x = kpt.x * scale + offsetX;
      const y = kpt.y * scale + offsetY;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
      return <View key={i} style={[styles.keypoint, { left: x - 5, top: y - 5 }]} />;
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>TF.js Performance Evaluation</Text>

      <View style={styles.controls}>
        <Text style={styles.label}>Model</Text>
        <View style={styles.row}>
          <Button title="Movenet" onPress={() => handleConfigChange("movenet")} disabled={model === "movenet"} />
          <Button title="Blazepose" onPress={() => handleConfigChange("blazepose")} disabled={model === "blazepose"} />
          <Button title="YOLO" onPress={() => handleConfigChange("yolo")} disabled={model === "yolo"} />
        </View>
      </View>

      <View style={styles.results}>
        <Text>Status: {status}</Text>
        <Text style={styles.fpsText}>FPS: {fps.toFixed(2)}</Text>
        <Text>Avg. Inference Time: {avgInferenceTime.toFixed(2)} ms</Text>
      </View>

      <Button
        title={isEvaluating ? "Stop Evaluation" : "Start Evaluation"}
        onPress={toggleEvaluation}
        color={isEvaluating ? "red" : "green"}
        disabled={status.includes("Loading")}
      />

      <View
        style={styles.cameraContainer}
        ref={cameraContainerRef}
        onLayout={() => requestAnimationFrame(measureCameraInWindow)}
      >
        {/* Camera first (below) */}
        <TensorCamera
          style={styles.camera}
          type={"front" as CameraType}
          cameraTextureHeight={720}
          cameraTextureWidth={1280}
          resizeHeight={modelInput.h}
          resizeWidth={modelInput.w}
          resizeDepth={3}
          onReady={handleTensorCameraStream}
          autorender={true}
          useCustomShadersToResize={false}
        />
        {/* Overlay second (on top) */}
        <Modal transparent visible={isEvaluating} onRequestClose={() => setIsEvaluating(false)}>
          <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
            <View
              style={{
                position: "absolute",
                left: overlayRect.x,
                top: overlayRect.y,
                width: overlayRect.width,
                height: overlayRect.height,
              }}
              pointerEvents="none"
            >
              {renderKeypoints()}
              <View style={{ position: "absolute", left: 10, top: 10, width: 6, height: 6, backgroundColor: "red" }} />
            </View>

            {/* In-modal control since touches can't pass through Modal */}
            <Pressable
              onPress={toggleEvaluation}
              style={{
                position: "absolute",
                right: 16,
                top: overlayRect.y + 16,
                backgroundColor: "rgba(0,0,0,0.6)",
                paddingHorizontal: 12,
                paddingVertical: 8,
                borderRadius: 6,
              }}
            >
              <Text style={{ color: "#fff" }}>{isEvaluating ? "Stop" : "Start"}</Text>
            </Pressable>
          </View>
        </Modal>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 10, backgroundColor: "#f5f5f5" },
  title: { fontSize: 20, fontWeight: "bold", textAlign: "center", marginVertical: 10 },
  controls: { marginVertical: 10, paddingBottom: 10, borderBottomWidth: 1, borderColor: "#ddd" },
  row: { flexDirection: "row", justifyContent: "space-around", marginVertical: 5 },
  label: { textAlign: "center", fontWeight: "bold", marginTop: 10 },
  results: { marginVertical: 15, padding: 10, backgroundColor: "white", borderRadius: 5 },
  fpsText: { fontSize: 18, fontWeight: "bold", color: "#333" },
  cameraContainer: { flex: 1, width: "100%", marginTop: 10 /* position: 'relative' by default */ },
  camera: { flex: 1, width: "100%", zIndex: 1 },
  keypoint: {
    position: "absolute",
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "aqua",
  },
});
