# rk-video-pipeline-cpp

一个支持 **Rockchip** 和 **NVIDIA** 双平台的视频推理最小工程骨架。

## 支持的硬件后端

| 平台 | 解码器 | 预处理 | 推理 |
|------|--------|--------|------|
| **Rockchip** (RK3588/RK3568) | MPP 硬解 | RGA | RKNN NPU |
| **NVIDIA** (GPU) | NVDEC 硬解 | CUDA | TensorRT |

## 快速开始

### Rockchip 平台

```bash
cmake -S . -B build -DPLATFORM=rockchip
cmake --build build -j4

# 运行
./build/rk_video_pipeline --backend rockchip test.mp4 yolov5s.rknn 640 640
```

### NVIDIA 平台

```bash
cmake -S . -B build -DPLATFORM=nvidia
cmake --build build -j4

# 运行
./build/rk_video_pipeline --backend nvidia test.mp4 yolov5s.engine 640 640
```

### 自动检测

```bash
cmake -S . -B build
cmake --build build -j4

# 运行时指定后端
./build/rk_video_pipeline --backend rockchip test.mp4 model.rknn
./build/rk_video_pipeline --backend nvidia test.mp4 model.engine
```

## 命令行参数

```
Usage: rk_video_pipeline [options] <video_or_rtsp> <model_file> [width] [height]

Options:
  --backend <rockchip|nvidia>  选择后端平台
  --gpu <id>                   GPU 设备 ID (默认：0)
  --max-frames <n>             最大处理帧数 (默认：30)
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `VIDEO_DECODER_BACKEND` | 解码器后端 (mpp/nvdec/cpu) |
| `VIDEO_PREPROC_BACKEND` | 预处理后端 (rga/cuda/cpu) |
| `VIDEO_INFER_BACKEND` | 推理后端 (rknn/trt) |
| `CUDA_DEVICE` | CUDA 设备 ID |

## 目录结构

```
rk-video-pipeline-cpp/
├── include/
│   ├── decoder_interface.hpp    # 解码器接口
│   ├── preproc_interface.hpp    # 预处理接口
│   ├── infer_interface.hpp      # 推理接口
│   ├── pipeline_types.hpp       # 共享数据类型
│   ├── ffmpeg_packet_source.hpp # FFmpeg 解复用
│   └── backends/
│       ├── mpp_decoder.hpp      # Rockchip MPP 解码
│       ├── nvdec_decoder.hpp    # NVIDIA NVDEC 解码
│       ├── rga_preprocessor.hpp # Rockchip RGA 预处理
│       ├── cuda_preprocessor.hpp# CUDA 预处理
│       ├── rknn_infer.hpp       # RKNN 推理
│       └── trt_infer.hpp        # TensorRT 推理
│
├── src/
│   ├── main.cpp                 # 统一入口
│   ├── ffmpeg_packet_source.cpp
│   └── backends/
│       ├── mpp_decoder.cpp
│       ├── nvdec_decoder.cpp
│       ├── rga_preprocessor.cpp
│       ├── cuda_preprocessor.cpp
│       ├── rknn_infer.cpp
│       ├── trt_infer.cpp
│       ├── decoder_factory.cpp
│       ├── preproc_factory.cpp
│       └── infer_factory.cpp
│
├── CMakeLists.txt
└── README.md
```

## 构建选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `PLATFORM` | auto | 目标平台 (auto/rockchip/nvidia) |
| `ENABLE_MPP_DECODER` | ON | Rockchip MPP 解码器 |
| `ENABLE_NVDEC_DECODER` | ON | NVIDIA NVDEC 解码器 |
| `ENABLE_RGA_PREPROC` | ON | Rockchip RGA 预处理 |
| `ENABLE_CUDA_PREPROC` | ON | CUDA 预处理 |
| `ENABLE_RKNN_INFER` | ON | RKNN 推理 |
| `ENABLE_TRT_INFER` | ON | TensorRT 推理 |

## 依赖

### Rockchip 平台
- FFmpeg (libavformat, libavcodec, libavutil)
- Rockchip MPP (`rockchip_mpp`)
- Rockchip RGA (`rga`)
- RKNN Runtime (`rknnrt`)

### NVIDIA 平台
- FFmpeg (libavformat, libavcodec, libavutil)
- CUDA Toolkit
- TensorRT

## 处理流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  FFmpeg     │────▶│  Decoder    │────▶│ Preprocessor│────▶│  Inference  │
│  Demux      │     │ (MPP/NVDEC) │     │ (RGA/CUDA)  │     │ (RKNN/TRT)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## 已知限制

- 这是"最小可改造工程骨架"，不是完整生产工程
- CUDA 预处理需要完整实现零拷贝路径
- 当前未实现检测框解析、显示和推流
