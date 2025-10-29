import React, { useState, useEffect, useRef, useCallback } from "react";
import { View, Text, Button, StyleSheet, TextInput } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from "react-native-vision-camera";
import { Keypoint, ModelId } from "../services/inferenceService";
import { useTensorflowModel } from "react-native-fast-tflite";
import { useResizePlugin } from "vision-camera-resize-plugin";
import Animated, { useAnimatedProps, useAnimatedStyle, useSharedValue } from "react-native-reanimated";
import { computeLetterbox, mapFromLetterbox } from "../utils/letterbox";

const AnimatedTextInput = Animated.createAnimatedComponent(TextInput);

export default function EvaluationScreen() {
  const [selectedModel, setModel] = useState<ModelId>();
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [status, setStatus] = useState("Loading...");
  const [viewDimensions, setViewDimensions] = useState({ width: 1, height: 1 });

  const fps = useSharedValue(0);
  const avgInferenceTime = useSharedValue(0);
  const keypoints = useSharedValue<Keypoint[]>([]);

  const cameraRef = useRef<Camera | null>(null);
  const cameraContainerRef = useRef<View | null>(null);

  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("front") ?? useCameraDevice("back");

  const statsRef = useRef({
    frameCount: 0,
    sessionStart: 0,
    totalMs: 0,
    lastFrameTs: 0,
    lastUpdate: 0,
  });

  // Pick a light input size per model to reduce work
  const modelInput = React.useMemo(() => {
    switch (selectedModel) {
      case "movenet":
        return { w: 192, h: 192 };
      case "blazepose":
        return { w: 256, h: 256 };
      case "yolo":
        return { w: 640, h: 640 };
      default:
        return { w: 256, h: 256 };
    }
  }, [selectedModel]);

  // Select TFLite model file (static requires)
  const modelSource = React.useMemo(() => {
    switch (selectedModel) {
      case "yolo":
        return require("../../assets/models/yolo/yolo11n-pose_float16.tflite");
      case "movenet":
        return require("../../assets/models/movenet/movenet.tflite");
      case "blazepose":
        return require("../../assets/models/blazepose/pose_landmarks_detector.tflite");
      default:
        return require("../../assets/models/yolo/yolo11n-pose_float16.tflite");
    }
  }, [selectedModel]);

  const tflite = useTensorflowModel(modelSource, "nnapi");
  const { resize } = useResizePlugin();

  useEffect(() => {
    (async () => {
      // Ensure permission prompt is shown
      let granted = hasPermission;
      if (!granted) {
        granted = await requestPermission();
      }

      if (granted) {
        setStatus("Ready. Please select a model.");
      } else {
        setStatus("Camera permission denied.");
      }
    })();
  }, []);

  const handleConfigChange = useCallback(
    async (newModel: ModelId) => {
      // Stop evaluation before changing the model
      if (isEvaluating) setIsEvaluating(false);
      setModel(newModel);
      setStatus(`Loading ${newModel}`);
      // Reset stats and UI
      statsRef.current = { frameCount: 0, sessionStart: 0, totalMs: 0, lastFrameTs: 0, lastUpdate: 0 };
      fps.value = 0;
      avgInferenceTime.value = 0;
      keypoints.value = [];
    },
    [isEvaluating]
  );

  const toggleEvaluation = () => {
    // Start evaluation: check for prerequisites
    if (!selectedModel) {
      setStatus("Please load a model first.");
      return;
    }
    if (!device) {
      setStatus("Camera is not ready yet.");
      return;
    }

    // Reset stats and start evaluation
    statsRef.current = { frameCount: 0, sessionStart: 0, totalMs: 0, lastFrameTs: 0, lastUpdate: 0 };
    setIsEvaluating((value) => !value);
  };

  // Frame processor: resize -> runSync -> interpret -> send keypoints to JS
  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";
      // 1. Early exit if not evaluating
      if (!isEvaluating) {
        return;
      }

      // 2. Throttle frames to avoid overloading the CPU
      const targetFps = 24;
      const tsUs = frame?.timestamp ?? 0;
      const minDeltaUs = 1e6 / targetFps;
      if (statsRef.current.lastFrameTs > 0 && tsUs - statsRef.current.lastFrameTs < minDeltaUs) {
        return;
      }
      statsRef.current.lastFrameTs = tsUs;

      // 3. Check if model is loaded
      const model = tflite?.state === "loaded" ? tflite.model : undefined;
      if (model == null) {
        console.log("Model not loaded, skipping frame.");
        return;
      }

      const INPUT = modelInput.w;
      const srcW = frame.width;
      const srcH = frame.height;

      const t0 = performance.now?.() ?? Date.now();

      // 4. Resize frame and DEFENSIVELY check the result
      const resized = resize(frame, {
        scale: { width: INPUT, height: INPUT },
        pixelFormat: "rgb",
        dataType: selectedModel === "movenet" ? "uint8" : "float32",
      });

      // 5. Run model and DEFENSIVELY check outputs
      const outputs = model.runSync([resized]);
      const out0: any = outputs[0];
      const outData: Float32Array = out0.data ?? out0;

      // 6. Decode based on model
      const kpts: Keypoint[] = [];
      const letter = computeLetterbox(srcW, srcH, INPUT);

      if (selectedModel === "yolo") {
        // YOLO11 pose decode
        const N_PROPOSALS = 8400; // Number of proposals

        // The output is [1, 56, 8400]. We need to find the best proposal.
        let bestIdx = -1;
        let maxRawScore = -Infinity;
        const scoreOffset = 4 * N_PROPOSALS; // Confidence scores are at channel 4

        for (let i = 0; i < N_PROPOSALS; i++) {
          const score = outData[scoreOffset + i];
          if (score > maxRawScore) {
            maxRawScore = score;
            bestIdx = i;
          }
        }

        if (bestIdx >= 0) {
          for (let k = 0; k < 17; k++) {
            const base = (5 + k * 3) * N_PROPOSALS; // Base index for the k-th keypoint
            const xIn = outData[base + bestIdx];
            const yIn = outData[base + N_PROPOSALS + bestIdx];
            const v = outData[base + 2 * N_PROPOSALS + bestIdx];

            // Map coordinates from letterboxed input space back to original source image space
            const mapped = mapFromLetterbox(xIn, yIn, srcW, srcH, letter, true);

            kpts.push({ x: mapped.x, y: mapped.y, visibility: v });
          }
        } else {
          for (let k = 0; k < 17; k++) {
            kpts.push({ x: 0, y: 0, visibility: 0 });
          }
        }
      } else if (selectedModel === "movenet") {
        // Movenet decode: output is [1, 1, 17, 3] -> [y, x, score]
        for (let k = 0; k < 17; k++) {
          const yIn = outData[k * 3 + 0];
          const xIn = outData[k * 3 + 1];
          const v = outData[k * 3 + 2];

          // Map coordinates from letterboxed input space back to original source image space
          const mapped = mapFromLetterbox(xIn, yIn, srcW, srcH, letter, true);

          kpts.push({ x: mapped.x, y: mapped.y, visibility: v });
        }
      } else if (selectedModel === "blazepose") {
        // Blazepose decode: output is [1, 195] for 39 landmarks.
        // We use the first 33. Each landmark is [x, y, z, visibility, presence].
        const N_LANDMARKS = 33;
        for (let k = 0; k < N_LANDMARKS; k++) {
          const xIn = outData[k * 5 + 0];
          const yIn = outData[k * 5 + 1];
          const v = outData[k * 5 + 3];

          // Map coordinates from letterboxed input space back to original source image space
          const mapped = mapFromLetterbox(xIn, yIn, srcW, srcH, letter, true);

          kpts.push({ x: mapped.x, y: mapped.y, visibility: v });
        }
      } else {
        // No-op for other models or when no detection
        for (let k = 0; k < 17; k++) {
          kpts.push({ x: 0, y: 0, visibility: 0 });
        }
      }

      // 7. Update stats (only runs if all previous steps succeeded)
      const t1 = performance.now?.() ?? Date.now();
      if (statsRef.current.frameCount === 0) statsRef.current.sessionStart = t0;
      statsRef.current.frameCount++;
      statsRef.current.totalMs += t1 - t0;

      // 8. Throttle UI updates to avoid overwhelming the JS thread
      const now = Date.now();
      if (now - statsRef.current.lastUpdate > 500) {
        statsRef.current.lastUpdate = now;
        const elapsed = Math.max(1e-6, (t1 - statsRef.current.sessionStart) / 1000);
        const currentFps = statsRef.current.frameCount / elapsed;
        const currentAvgMs = statsRef.current.totalMs / statsRef.current.frameCount;

        keypoints.value = kpts;
        fps.value = currentFps;
        avgInferenceTime.value = currentAvgMs;
        console.log(`FPS: ${currentFps.toFixed(2)}, Avg Inference: ${currentAvgMs.toFixed(2)} ms`);
      }
    },
    [isEvaluating, tflite, resize, modelInput, selectedModel]
  );

  useEffect(() => {
    // Reflect model load state in status
    if (!selectedModel) return;
    if (tflite?.state === "loading") setStatus(`Loading ${selectedModel}...`);
    else if (tflite?.state === "loaded") setStatus(`Ready to evaluate ${selectedModel}.`);
    else if (tflite?.state === "error") setStatus(`Failed to load ${selectedModel}.`);
  }, [selectedModel, tflite?.state]);

  const animatedFpsProps = useAnimatedProps(() => {
    return { text: `FPS: ${fps.value.toFixed(2)}` } as any;
  });

  const animatedInferenceProps = useAnimatedProps(() => {
    return { text: `Avg. Inference Time: ${avgInferenceTime.value.toFixed(2)} ms` } as any;
  });

  // Pre-render 17 animated keypoint dots driven by SharedValues (no JS updates per frame)
  const KeypointDot = () => {
    const { width: viewW, height: viewH } = viewDimensions;
    // derive source dimensions from the selected model's input size
    const srcW = modelInput.w;
    const srcH = modelInput.h;

    // cover scaling
    const scale = Math.max(viewW / srcW, viewH / srcH);
    const offsetX = (viewW - srcW * scale) / 2;
    const offsetY = (viewH - srcH * scale) / 2;

    const numKeypoints = selectedModel === "blazepose" ? 33 : 17;

    return (
      <>
        {Array.from({ length: numKeypoints }).map((_, i) => {
          const animatedStyle = useAnimatedStyle(() => {
            const kp = keypoints.value[i];
            if (!kp || kp.visibility < 0.1) {
              return { display: "none" };
            }
            const x = kp.x * scale + offsetX - 5;
            const y = kp.y * scale + offsetY - 5;
            return {
              transform: [{ translateX: x }, { translateY: y }],
            };
          });
          return <Animated.View key={i} style={[styles.keypoint, animatedStyle]} />;
        })}
      </>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>TFLite Performance Evaluation</Text>

      <View style={styles.controls}>
        <Text style={styles.label}>Model</Text>
        <View style={styles.row}>
          <Button
            title="Movenet"
            onPress={() => handleConfigChange("movenet")}
            disabled={selectedModel === "movenet"}
          />
          <Button
            title="Blazepose"
            onPress={() => handleConfigChange("blazepose")}
            disabled={selectedModel === "blazepose"}
          />
          <Button title="YOLO" onPress={() => handleConfigChange("yolo")} disabled={selectedModel === "yolo"} />
        </View>
      </View>

      <View style={styles.results}>
        <Text>Status: {status}</Text>
        <AnimatedTextInput style={styles.fpsText} animatedProps={animatedFpsProps} editable={false} />
        <AnimatedTextInput style={styles.bodyText} animatedProps={animatedInferenceProps} editable={false} />
      </View>

      <Button
        title={isEvaluating ? "Stop Evaluation" : "Start Evaluation"}
        onPress={toggleEvaluation}
        color={isEvaluating ? "red" : "green"}
        disabled={tflite?.state !== "loaded" || !device || !hasPermission}
      />

      <View
        style={styles.cameraContainer}
        ref={cameraContainerRef}
        onLayout={(event) => {
          const { width, height } = event.nativeEvent.layout;
          setViewDimensions({ width, height });
        }}
      >
        {device && hasPermission ? (
          <Camera
            ref={cameraRef}
            style={styles.camera}
            device={device}
            isActive={true}
            resizeMode="cover"
            frameProcessor={frameProcessor}
          />
        ) : (
          <View style={[styles.camera, { alignItems: "center", justifyContent: "center" }]}>
            <Text>{!hasPermission ? "Camera permission required." : "No camera device found."}</Text>
          </View>
        )}

        <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
          {isEvaluating && <KeypointDot />}
        </View>
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
  fpsText: { fontSize: 18, fontWeight: "bold", color: "#333", padding: 0, margin: 0 },
  bodyText: { color: "#333", padding: 0, margin: 0 },
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
