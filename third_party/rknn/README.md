# RKNN Runtime Library

## Getting librknnrt.so

The RKNN runtime library (`librknnrt.so`) is not included in this repository because it is platform-specific and version-dependent.

### Download from Rockchip

1. Visit the [RKNN SDK](https://github.com/airockchip/rknn-toolkit2) repository
2. Navigate to `runtime/<your_chip>/Linux/librknn_api/aarch64/`
3. Copy `librknnrt.so` to `third_party/rknn/lib/`

### Version Compatibility

| Chip | Recommended SDK | Notes |
|------|----------------|-------|
| RK3588 | rknn-toolkit2 >= 1.5 | 6 TOPS NPU, 3 cores |
| RK3568 | rknn-toolkit2 >= 1.4 | 1 TOPS NPU, 1 core |

### Directory Structure

```
third_party/rknn/
  include/
    rknn_api.h        # RKNN C API header (tracked in git)
  lib/
    librknnrt.so      # RKNN runtime (NOT tracked, download separately)
```
