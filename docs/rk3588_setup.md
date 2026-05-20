# RK3588 Environment Setup

## Hardware

- Board: Rockchip RK3588-based SBC (e.g., Orange Pi 5, Rock 5B, Firefly ROC-RK3588S)
- NPU: 6 TOPS, 3 cores
- RAM: >= 4GB recommended

## OS Image

Use the official Debian/Ubuntu image provided by your board vendor. Ensure the kernel version supports NPU drivers (>= 5.10).

## Install NPU Driver

```bash
# Check if NPU driver is loaded
dmesg | grep -i rknpu

# If not loaded, install the NPU driver from your board vendor
# Typically included in the OS image or available as a .deb package
```

## Install RKNN Runtime

```bash
# Option 1: From rknn-toolkit2 SDK
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2/runtime/RK3588/Linux/librknn_api/aarch64/

# Copy to project
cp librknnrt.so /path/to/vision-inference-pipeline/third_party/rknn/lib/
```

## Install MPP (Media Process Platform)

```bash
# MPP is typically pre-installed on Rockchip OS images
# Verify:
ls /usr/lib/aarch64-linux-gnu/librockchip_mpp.so*

# If not available:
git clone https://github.com/rockchip-linux/mpp.git
cd mpp/build/linux/aarch64
./make-Makefiles.bash
make -j$(nproc)
sudo make install
```

## Install RGA (Raster Graphic Acceleration)

```bash
# RGA is typically pre-installed
# Verify:
ls /usr/lib/aarch64-linux-gnu/librga.so*

# If not available:
git clone https://github.com/airockchip/librga.git
cd librga
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

## Install FFmpeg

```bash
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavutil-dev

# Verify hardware decode support
ffmpeg -hwaccels | grep rkmpp
```

## Build the Pipeline

```bash
cd vision-inference-pipeline
cmake -S . -B build-rockchip -DPLATFORM=rockchip
cmake --build build-rockchip -j$(nproc)
```

## Convert Model to RKNN

```bash
# Install rknn-toolkit2 on your development machine (not the board)
pip install rknn-toolkit2

# Convert ONNX to RKNN
python -c "
from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
rknn.load_onnx(model='yolov8n.onnx')
rknn.build(do_quantization=True, dataset='calibration_list.txt')
rknn.export_rknn('yolov8n.rknn')
"
```

## Verify

```bash
# Check NPU is available
cat /sys/kernel/debug/rknpu/version

# Run pipeline
./build-rockchip/video_pipeline --backend rockchip --verbose test.mp4 yolov8n.rknn 640 640
```

## Performance Tuning

| Parameter | Recommendation |
|-----------|----------------|
| `--infer-workers 2` | Balanced throughput/stability |
| `--rknn-zero-copy false` | Stable for annotated output |
| `--rknn-core-mask 0_1_2` | Use all 3 NPU cores |
| `--encoder-fps 30` | Cap output FPS to avoid slow-motion |
