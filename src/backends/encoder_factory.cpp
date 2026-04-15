#include "backend_registry.hpp"
#include "encoder_interface.hpp"

#include <stdexcept>

#if defined(ENABLE_NVENC_ENCODER)
#include "backends/nvenc_encoder.hpp"
#endif

EncoderBackendType detectAvailableEncoderBackend() {
#if defined(ENABLE_NVENC_ENCODER)
  return EncoderBackendType::kNvidiaNvEnc;
#else
  return EncoderBackendType::kAuto;
#endif
}

std::unique_ptr<IEncoderBackend> createEncoderBackend(EncoderBackendType type) {
  if (type == EncoderBackendType::kAuto) {
    type = detectAvailableEncoderBackend();
  }

  switch (type) {
#if defined(ENABLE_NVENC_ENCODER)
    case EncoderBackendType::kNvidiaNvEnc:
      return std::make_unique<NvencEncoder>();
#endif

    case EncoderBackendType::kRockchipMpp:
      throw std::runtime_error(
          "Encoder backend 'rockchip-mpp' is compiled but not ready for production output. "
          "Use another encoder backend.");

    case EncoderBackendType::kCpu:
    default:
      throw std::runtime_error(
          "Encoder backend '" + toString(type) +
          "' is not available in this build. Available: " + availableEncoderBackends());
  }
}
