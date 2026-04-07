#include "infer_interface.hpp"
#include "backends/rknn_infer.hpp"
#include "backends/trt_infer.hpp"

#include <stdexcept>
#include <cstdlib>

/**
 * 检测当前平台可用的推理后端
 */
InferBackendType detectAvailableInferBackend() {
  // 编译时指定优先
#if defined(ROCKCHIP_PLATFORM)
  return InferBackendType::kRockchipRknn;
#elif defined(NVIDIA_PLATFORM)
  return InferBackendType::kNvidiaTrt;
#endif

  // 运行时通过环境变量指定
  const char* env = std::getenv("VIDEO_INFER_BACKEND");
  if (env) {
    std::string backend(env);
    if (backend == "rknn" || backend == "rockchip") {
      return InferBackendType::kRockchipRknn;
    }
    if (backend == "trt" || backend == "tensorrt" || backend == "nvidia") {
      return InferBackendType::kNvidiaTrt;
    }
  }

  return InferBackendType::kAuto;
}

/**
 * 创建推理后端实例
 */
std::unique_ptr<IInferenceBackend> createInferBackend(InferBackendType type) {
  if (type == InferBackendType::kAuto) {
    type = detectAvailableInferBackend();
  }

  switch (type) {
#if defined(ROCKCHIP_PLATFORM) || !defined(PLATFORM_STRICT_CHECK)
    case InferBackendType::kRockchipRknn:
      return std::make_unique<RknnInfer>();
#endif

#if defined(NVIDIA_PLATFORM) || !defined(PLATFORM_STRICT_CHECK)
    case InferBackendType::kNvidiaTrt: {
      auto infer = std::make_unique<TrtInfer>();
      if (const char* gpu_id = std::getenv("CUDA_DEVICE")) {
        infer->setGpuId(std::atoi(gpu_id));
      }
      return infer;
#endif

    default:
      throw std::runtime_error("Unknown or unavailable infer backend type");
  }
}
