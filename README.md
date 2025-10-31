# Reptor TF.js Benchmark

This is a React Native application designed to benchmark the performance of various pose estimation models using TensorFlow.js on a mobile device. It uses `expo-camera` and the `@tensorflow/tfjs-react-native` package to run inference against a live camera stream.

The app measures and displays real-time Frames Per Second (FPS) and average inference time, providing a clear comparison of model efficiency when running in a JavaScript-based environment with a WebGL backend.

## Features

- **Real-time Inference:** Runs pose estimation models on the live camera feed using TF.js.
- **Multiple Model Support:** Easily switch between different models (YOLO, MoveNet, BlazePose).
- **Performance Metrics:** Displays live FPS and average inference time.
- **WebGL Backend:** Utilizes the `rn-webgl` backend provided by `@tensorflow/tfjs-react-native` for GPU acceleration.
- **Cross-Platform:** Built with Expo for easier development across Android and iOS.

## Benchmark Results

The following benchmarks were run on an **Android phone with a MediaTek Dimensity 1100 SoC**. The evaluation was run for 10 seconds for each model.

| Model             | Input Size | Average FPS | Avg. Inference Time |
| ----------------- | :--------: | :---------: | :-----------------: |
| **MoveNet**       | `192x192`  |  **~3.4**   |    **~250.4 ms**    |
| **BlazePose**     | `256x256`  |  **~3.0**   |    **~251.1 ms**    |
| **YOLOv11n-Pose** | `640x640`  |  **~0.8**   |   **~1277.7 ms**    |

### Analysis

- **MoveNet (Single-Pose Lightning)** and **BlazePose** show surprisingly similar performance on this hardware, with MoveNet being marginally faster. Both models struggle to achieve high frame rates, indicating a significant overhead from the TF.js library and WebGL backend compared to native solutions.
- **YOLOv11n-Pose** is the slowest, which is expected given its complexity, at a input size of `640x640`.

## Technology Stack

- **React Native**
- **Expo (Development Build)**
- **TensorFlow.js** (`@tensorflow/tfjs`)
- **@tensorflow/tfjs-react-native:** For the WebGL backend and camera tensor stream.
- **Expo Camera:** For camera access.
- **TypeScript**

## Getting Started

### Prerequisites

- Node.js and npm/yarn
- A configured React Native development environment (Android Studio / Xcode).
- A physical Android or iOS device.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/royng163/reptor-benchmark.git
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
