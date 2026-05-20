# YOLOv8n Benchmark - NVIDIA GPU (FP32)

## Environment

| Item | Value |
|------|-------|
| Platform | Windows / NVIDIA GPU |
| Backend | PyTorch (pt) |
| Precision | FP32 |
| Device | CUDA |
| Input Size | 640x640 |
| Test Video | bus_loop.mp4 (30 frames) |

## Results

| Metric | Value |
|--------|-------|
| Mean Inference | 7.50 ms |
| Std Inference | 1.14 ms |
| Min Inference | 6.16 ms |
| Max Inference | 11.15 ms |
| Mean FPS | 136.0 |
| Min FPS | 89.7 |
| Max FPS | 162.3 |

## Notes

- Warmup: 3 frames excluded
- Measured via `vision-inference-benchmark` project
- This is PyTorch baseline; TensorRT INT8 expected to be 3-5x faster
