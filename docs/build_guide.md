# Build Guide

## Prerequisites

### Common Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| CMake | >= 3.16 | Build system |
| FFmpeg | >= 5.0 | Demux + mux (libavformat, libavcodec, libavutil) |
| OpenCV | >= 4.x | Visualization (optional, for drawing) |

### Rockchip Platform

| Dependency | Source |
|------------|--------|
| MPP | [rockchip-mpp](https://github.com/rockchip-linux/mpp) |
| RGA | [librga](https://github.com/airockchip/librga) |
| RKNN Runtime | [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) |

### NVIDIA Platform

| Dependency | Version | Source |
|------------|---------|--------|
| CUDA Toolkit | >= 11.0 | [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) |
| TensorRT | >= 8.0 | [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) |
| NVDEC/NVENC | via FFmpeg | FFmpeg built with `--enable-nvdec --enable-nvenc` |

## Build Commands

### Rockchip (cross-compile for RK3588)

```bash
# On build host with RKNN SDK and cross-compiler
cmake -S . -B build-rockchip \
  -DPLATFORM=rockchip \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++

cmake --build build-rockchip -j$(nproc)
```

### NVIDIA (native build)

```bash
cmake -S . -B build -DPLATFORM=nvidia
cmake --build build -j$(nproc)
```

### Auto-detect

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `PLATFORM` | auto | Target platform: `auto`, `rockchip`, `nvidia` |
| `ENABLE_MPP_DECODER` | ON | Rockchip MPP decoder |
| `ENABLE_NVDEC_DECODER` | ON | NVIDIA NVDEC decoder |
| `ENABLE_RGA_PREPROC` | ON | Rockchip RGA preprocessor |
| `ENABLE_CUDA_PREPROC` | ON | CUDA preprocessor |
| `ENABLE_RKNN_INFER` | ON | RKNN inference |
| `ENABLE_TRT_INFER` | ON | TensorRT inference |

## Output

Build produces:

```
build/video_pipeline          # Main executable
build/app_config_test         # Unit tests
```

## Running

```bash
# Rockchip
./build-rockchip/video_pipeline --backend rockchip test.mp4 model.rknn 640 640

# NVIDIA
./build/video_pipeline --backend nvidia test.mp4 model.engine 640 640
```

## Troubleshooting

### FFmpeg not found
```bash
# Ubuntu/Debian
sudo apt install libavformat-dev libavcodec-dev libavutil-dev

# Or build FFmpeg from source with NVDEC/NVENC support
```

### TensorRT not found
Set `TENSORRT_DIR` environment variable or pass to CMake:
```bash
cmake -S . -B build -DTENSORRT_DIR=/path/to/TensorRT
```

### RKNN runtime not found
Ensure `third_party/rknn/lib/librknnrt.so` exists. See [third_party/rknn/README.md](../third_party/rknn/README.md).
