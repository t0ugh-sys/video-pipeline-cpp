#include "encoder_interface.hpp"
#include "backends/nvenc_encoder.hpp"
#include "backends/mpp_encoder.hpp"

#include <stdexcept>

EncoderBackendType detectAvailableEncoderBackend() {
#ifdef NVIDIA_PLATFORM
  return EncoderBackendType::kNvidiaNvEnc;
#endif
#ifdef ROCKCHIP_PLATFORM
  return EncoderBackendType::kRockchipMpp;
#endif
  return EncoderBackendType::kCpu;
}

std::unique_ptr<IEncoderBackend> createEncoderBackend(EncoderBackendType type) {
  if (type == EncoderBackendType::kAuto) {
    type = detectAvailableEncoderBackend();
  }

  switch (type) {
    case EncoderBackendType::kNvidiaNvEnc:
      return std::make_unique<NvencEncoder>();

    case EncoderBackendType::kRockchipMpp:
      return std::make_unique<MppEncoder>();

    case EncoderBackendType::kCpu:
      // TODO: 实现 CPU 软编码
      throw std::runtime_error("CPU encoder not yet implemented");

    default:
      throw std::runtime_error("Unknown encoder backend type");
  }
}
