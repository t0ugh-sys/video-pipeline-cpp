#include "decoder_interface.hpp"
#include "backends/mpp_decoder.hpp"
#include "backends/nvdec_decoder.hpp"

#include <stdexcept>

/**
 * 检测当前平台可用的解码器后端
 * 按优先级返回：MPP > NVDEC > CPU
 */
DecoderBackendType detectAvailableDecoderBackend() {
  // 1. 优先检测 Rockchip MPP (通过检查头文件或设备节点)
#ifdef ROCKCHIP_PLATFORM
  return DecoderBackendType::kRockchipMpp;
#endif

  // 2. 检测 NVIDIA GPU
#ifdef NVIDIA_PLATFORM
  return DecoderBackendType::kNvidiaNvdec;
#endif

  // 3. 运行时检测 (简化：通过环境变量或命令行指定)
  const char* env = std::getenv("VIDEO_DECODER_BACKEND");
  if (env) {
    std::string backend(env);
    if (backend == "mpp" || backend == "rockchip") {
      return DecoderBackendType::kRockchipMpp;
    }
    if (backend == "nvdec" || backend == "nvidia") {
      return DecoderBackendType::kNvidiaNvdec;
    }
    if (backend == "cpu") {
      return DecoderBackendType::kCpu;
    }
  }

  // 默认自动选择
  return DecoderBackendType::kAuto;
}

/**
 * 创建解码器后端实例
 */
std::unique_ptr<IDecoderBackend> createDecoderBackend(DecoderBackendType type) {
  // 自动检测
  if (type == DecoderBackendType::kAuto) {
    type = detectAvailableDecoderBackend();
  }

  switch (type) {
    case DecoderBackendType::kRockchipMpp:
      return std::make_unique<MppDecoder>();

    case DecoderBackendType::kNvidiaNvdec: {
      auto decoder = std::make_unique<NvdecDecoder>();
      // 从环境变量读取 GPU ID
      if (const char* gpu_id = std::getenv("CUDA_DEVICE")) {
        decoder->setGpuId(std::atoi(gpu_id));
      }
      return decoder;
    }

    case DecoderBackendType::kCpu:
      // CPU 解码可以用 FFmpeg 软件解码实现
      // 这里暂时用 NVDEC 回退 (不设置 hwaccel)
      throw std::runtime_error("CPU decoder not yet implemented");

    default:
      throw std::runtime_error("Unknown decoder backend type");
  }
}
