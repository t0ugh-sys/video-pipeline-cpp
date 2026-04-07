#include "preproc_interface.hpp"
#include "backends/rga_preprocessor.hpp"
#include "backends/cuda_preprocessor.hpp"

#include <stdexcept>
#include <cstdlib>

/**
 * 检测当前平台可用的预处理后端
 */
PreprocBackendType detectAvailablePreprocBackend() {
  // 编译时指定优先
#if defined(ROCKCHIP_PLATFORM)
  return PreprocBackendType::kRockchipRga;
#elif defined(NVIDIA_PLATFORM)
  return PreprocBackendType::kNvidiaCuda;
#endif

  // 运行时通过环境变量指定
  const char* env = std::getenv("VIDEO_PREPROC_BACKEND");
  if (env) {
    std::string backend(env);
    if (backend == "rga" || backend == "rockchip") {
      return PreprocBackendType::kRockchipRga;
    }
    if (backend == "cuda" || backend == "nvidia") {
      return PreprocBackendType::kNvidiaCuda;
    }
    if (backend == "cpu") {
      return PreprocBackendType::kCpu;
    }
  }

  return PreprocBackendType::kAuto;
}

/**
 * 创建预处理后端实例
 */
std::unique_ptr<IPreprocessorBackend> createPreprocBackend(PreprocBackendType type) {
  if (type == PreprocBackendType::kAuto) {
    type = detectAvailablePreprocBackend();
  }

  switch (type) {
#if defined(ROCKCHIP_PLATFORM) || !defined(PLATFORM_STRICT_CHECK)
    case PreprocBackendType::kRockchipRga:
      return std::make_unique<RgaPreprocessor>();
#endif

#if defined(NVIDIA_PLATFORM) || !defined(PLATFORM_STRICT_CHECK)
    case PreprocBackendType::kNvidiaCuda: {
      auto preproc = std::make_unique<CudaPreprocessor>();
      if (const char* gpu_id = std::getenv("CUDA_DEVICE")) {
        preproc->setGpuId(std::atoi(gpu_id));
      }
      return preproc;
#endif

    case PreprocBackendType::kCpu:
      throw std::runtime_error("CPU preprocessor not yet implemented");

    default:
      throw std::runtime_error("Unknown or unavailable preproc backend type");
  }
}
