# YOLO26n Benchmark - NVIDIA GPU (FP32)

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
| Mean Inference | 8.49 ms |
| Std Inference | 0.78 ms |
| Min Inference | 7.59 ms |
| Max Inference | 10.66 ms |
| Mean FPS | 118.7 |
| Min FPS | 93.8 |
| Max FPS | 131.7 |

## Notes

- Warmup: 3 frames excluded
- Measured via `vision-inference-benchmark` project
- YOLO26n has slightly higher latency than YOLOv8n but more stable (lower std)
