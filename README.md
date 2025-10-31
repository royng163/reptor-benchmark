# Reptor TFLite Benchmark

This is a React Native application designed to benchmark the performance of various TFLite pose estimation models on a mobile device. It uses `react-native-vision-camera` for high-performance camera access and `react-native-fast-tflite` for running model inference directly on the UI thread.

The app measures and displays real-time Frames Per Second (FPS) and average inference time, providing a clear comparison of model efficiency.

## Features

- **Real-time Inference:** Runs pose estimation models on the live camera feed.
- **Multiple Model Support:** Easily switch between different TFLite models (YOLO, MoveNet, BlazePose).
- **Performance Metrics:** Displays live FPS and average inference time.
- **UI Thread Processing:** Utilizes Vision Camera Frame Processors to run inference synchronously for maximum speed.
- **Optimized UI:** Uses `react-native-reanimated` to render keypoints and display stats without blocking the JS thread.
- **Hardware Acceleration:** Configured to use the NNAPI delegate for hardware-accelerated inference.

## Benchmark Results

The following benchmarks were run on an **Android phone with a MediaTek Dimensity 1100 SoC**. The evaluation was run for 10 seconds for each model.

| Model             | Input Size | Average FPS | Avg. Inference Time |
| ----------------- | :--------: | :---------: | :-----------------: |
| **MoveNet**       | `192x192`  |  **~25.4**  |    **~38.5 ms**     |
| **BlazePose**     | `256x256`  |  **~19.8**  |    **~49.6 ms**     |
| **YOLOv11n-Pose** | `640x640`  |  **~2.0**   |    **~495.3 ms**    |

### Analysis

- **MoveNet (Single-Pose Lightning)** shows the best performance, achieving the highest FPS and lowest latency. Its small input size and efficient architecture make it ideal for real-time applications on mobile devices.
- **BlazePose** offers a good balance, providing 3D and a higher number of keypoints (33 vs. 17) with respectable performance.
- **YOLOv11n-Pose** is significantly slower, primarily due to its much larger input resolution (`640x640`). While powerful, it is too demanding for real-time use on this particular hardware without further optimization (e.g., quantization, smaller input size).

## Technology Stack

- **React Native**
- **Expo (Development Build)**
- **React Native Vision Camera:** For fast and efficient camera frame access.
- **react-native-fast-tflite:** For running TFLite models with hardware acceleration (NNAPI).
- **vision-camera-resize-plugin:** For fast, native image resizing on the UI thread.
- **React Native Reanimated:** For performant UI updates and animations.
- **TypeScript**

## Getting Started

### Prerequisites

- Node.js and npm/yarn
- A configured React Native development environment (Android Studio / Xcode).
- A physical Android or iOS device.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd reptor-benchmark
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

### Running the App

Because this project uses native libraries, you must create a development build and run it on a physical device. It will not work in the Expo Go app.

1.  **Run on Android:**

    ```bash
    npx expo run:android
    ```

2.  **Run on iOS:**
    ```bash
    npx expo run:ios
    ```

## How to Use

1.  Launch the app on your device.
2.  Select a model to load (Movenet, Blazepose, or YOLO).
3.  Wait for the status to indicate that the model is "Ready".
4.  Press the "Start Evaluation" button.
5.  The evaluation will run for 10 seconds, displaying live FPS and inference time, before stopping automatically. You can also stop it manually.
