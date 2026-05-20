# Deployment Comparison

## Positioning

This repository is most compelling when presented as a comparison of deployment strategies rather than just a code sample.

## Comparison Table

| Scenario | Strength | Weakness | Best Fit |
|----------|----------|----------|----------|
| Rockchip edge box | Low power, on-device inference, no GPU server needed | Smaller ecosystem, RKNN conversion workflow | Fixed-function edge deployment |
| NVIDIA workstation | Strong dev velocity, TensorRT, CUDA tooling | Windows requires MSVC for full CUDA path | Internal tooling, demos, lab validation |
| NVIDIA Linux server | Best support for CUDA/TensorRT/NVDEC/NVENC | Higher ops complexity | Production inference service |
| CPU-only fallback | Lowest dependency burden | Poor performance for real-time inference | Functional testing only |

## What To Highlight In A Portfolio

### 1. Systems Thinking

Show that the project is not just "run a model", but a full path:

- input transport
- hardware decode
- pixel format conversion
- inference runtime binding
- postprocess
- optional visualization / output

### 2. Tradeoff Awareness

Document tradeoffs explicitly:

- Rockchip gives edge efficiency
- NVIDIA gives tooling maturity and high peak throughput
- Windows is acceptable for development
- Linux is the better production target for CUDA/TensorRT

### 3. Benchmark-Backed Claims

Every claim should point back to reproducible benchmark reports under `benchmarks/results/`.

## Suggested Showcase Narrative

Use this structure in a portfolio or case study:

1. Problem: need one inference pipeline skeleton that spans edge and GPU targets.
2. Architecture: stage-oriented interfaces with swappable backends.
3. Engineering work: backend selection, capability validation, benchmark instrumentation.
4. Results: FPS and stage latency comparisons across platforms.
5. Lessons: zero-copy and toolchain constraints dominate real deployment quality.
