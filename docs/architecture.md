# Architecture

## Overview

vision-inference-pipeline is a stage-oriented video inference framework with swappable backends. Each processing stage is defined by a C++ interface, and platform-specific implementations are selected at compile time or runtime.

## Data Flow

```
Input Source          Decode           Preprocess        Inference         Postprocess       Output
(RTSP/File)    ->   (MPP/NVDEC)   ->  (RGA/CUDA)   ->  (RKNN/TRT)   ->   (YOLO NMS)   ->  (Draw/Encode)
     |                  |                  |                |                  |                |
 FFmpegPacket      DecodedFrame        RgbImage       InferenceOutput    DetectionResult   Visualizer
   Source             (NV12)          (640x640)        (raw tensors)      (boxes+cls)      Encoder
```

## Module Architecture

```
main.cpp
  |
  +-- app_config          CLI argument parsing, runtime configuration
  |
  +-- backend_registry    Compile-time capability query (isCompiledIn())
  |
  +-- validateAppConfig   Cross-checks config against compiled backends
  |
  +-- pipeline_runner     Orchestrates the full decode->infer->output chain
  |
  +-- Interfaces (include/*.hpp)
  |     +-- IDecoderBackend     decode: submitPacket() -> receiveFrame()
  |     +-- IPreprocBackend     preprocess: DecodedFrame -> RgbImage
  |     +-- IInferenceBackend   infer: RgbImage -> InferenceOutput
  |     +-- IPostprocBackend    postprocess: InferenceOutput -> DetectionResult
  |     +-- IVisualizer         draw: DetectionResult + frame -> annotated frame
  |     +-- IEncoderBackend     encode: annotated frame -> output stream
  |
  +-- Backends (src/backends/*.cpp)
        +-- mpp_decoder         Rockchip MPP hardware decode
        +-- nvdec_decoder       NVIDIA NVDEC hardware decode
        +-- rga_preprocessor    Rockchip RGA color conversion + resize
        +-- cuda_preprocessor   NVIDIA CUDA color conversion + resize
        +-- rknn_infer          RKNN NPU inference
        +-- trt_infer           TensorRT GPU inference
        +-- yolo_postproc       YOLO detection head parsing + NMS
        +-- opencv_visualizer   OpenCV drawing (classic/yolo style)
        +-- mpp_encoder         Rockchip MPP H.264 encode
        +-- nvenc_encoder       NVIDIA NVENC encode
```

## Interface Pattern

Each backend stage follows the same pattern:

```cpp
// Interface (include/decoder_interface.hpp)
class IDecoderBackend {
  virtual void open(VideoCodec codec) = 0;
  virtual void submitPacket(const EncodedPacket& packet) = 0;
  virtual std::optional<DecodedFrame> receiveFrame() = 0;
  virtual std::string name() const = 0;
};

// Factory (src/backends/decoder_factory.cpp)
std::unique_ptr<IDecoderBackend> createDecoderBackend(DecoderBackendType type);

// Registry (src/backend_registry.cpp)
bool isCompiledIn(DecoderBackendType type);  // compile-time check
```

## Platform Paths

### NVIDIA Path
```
FFmpeg demux -> NVDEC decode -> CUDA preprocess -> TensorRT infer -> YOLO postproc -> OpenCV draw -> NVENC encode
```

### Rockchip Path
```
FFmpeg demux -> MPP decode -> RGA preprocess -> RKNN infer -> YOLO postproc -> CPU draw -> RGA convert -> MPP encode
```

## Build System

CMake with platform-specific feature flags:

```cmake
-DPLATFORM=rockchip   # enables MPP + RGA + RKNN
-DPLATFORM=nvidia     # enables NVDEC + CUDA + TensorRT + NVENC
-DPLATFORM=auto       # detects available SDKs
```

Each backend can be individually disabled:
```cmake
-DENABLE_MPP_DECODER=OFF
-DENABLE_RKNN_INFER=OFF
```
