# TensorRT Environment Setup

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA GPU | >= RTX 2060 / T4 | GPU with NVDEC/NVENC |
| CUDA Toolkit | >= 11.0 | GPU compute |
| cuDNN | >= 8.0 | Deep learning primitives |
| TensorRT | >= 8.0 | Inference optimization |
| FFmpeg | >= 5.0 | With NVDEC/NVENC support |

## Install CUDA Toolkit

```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-2

# Verify
nvcc --version
nvidia-smi
```

## Install cuDNN

```bash
sudo apt install libcudnn8 libcudnn8-dev

# Verify
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## Install TensorRT

```bash
# Option 1: apt (Ubuntu)
sudo apt install libnvinfer8 libnvinfer-dev libnvinfer-plugin-dev

# Option 2: tar package from NVIDIA
# Download from https://developer.nvidia.com/tensorrt
tar -xzvf TensorRT-8.x.x.Linux.x86_64-gnu.cuda-12.x.tar.gz
export TENSORRT_DIR=$(pwd)/TensorRT-8.x.x
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
```

## Build FFmpeg with NVDEC/NVENC

```bash
# Install NVIDIA headers
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make install

# Build FFmpeg
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
./configure \
  --enable-nvdec \
  --enable-nvenc \
  --enable-cuda \
  --enable-cuvid \
  --enable-nonfree \
  --enable-libnpp
make -j$(nproc)
sudo make install
```

## Convert Model to TensorRT Engine

```bash
# From ONNX
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n.engine \
  --fp16

# INT8 quantization (requires calibration)
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n_int8.engine \
  --int8 \
  --calib=calibration_cache.bin
```

## Build the Pipeline

```bash
cd vision-inference-pipeline
cmake -S . -B build -DPLATFORM=nvidia
cmake --build build -j$(nproc)
```

## Verify

```bash
# Check GPU
nvidia-smi

# Run pipeline
./build/video_pipeline --backend nvidia --verbose test.mp4 yolov8n.engine 640 640

# Expected verbose output:
# [PIPELINE] stages: NVDEC -> CUDA -> TensorRT -> NVENC
# [TRT] input_binding: images
# [TRT] output_binding: output0
# [TRT] input_mode: device
```

## Performance Tuning

| Parameter | Effect |
|-----------|--------|
| `--fp16` | 2x faster inference, minimal accuracy loss |
| `--int8` | 3-4x faster, requires calibration data |
| `--workspace 4096` | More workspace for layer optimization |
| `--gpu 0` | Select GPU device |

## Common Issues

### `AV_PIX_FMT_CUDA` not supported
- Ensure FFmpeg is built with `--enable-cuvid` and `--enable-cuda`
- Check CUDA driver version matches CUDA toolkit

### TensorRT engine incompatible
- Engines are platform-specific; rebuild on target machine
- TensorRT version must match between build and runtime

### NVENC encoder not available
- Check GPU supports NVENC: `nvidia-smi -q | grep Encoder`
- Ensure FFmpeg is built with `--enable-nvenc`
