#pragma once

#include "pipeline_types.hpp"

#include <memory>
#include <string>

struct PreprocessOptions {
  bool letterbox = false;
  std::uint8_t paddingValue = 114;
};

class IPreprocessorBackend {
 public:
  virtual ~IPreprocessorBackend() = default;

  virtual RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight,
      const PreprocessOptions& options = {}) = 0;

  virtual std::string name() const = 0;
};

enum class PreprocBackendType {
  kAuto,
  kRockchipRga,
  kNvidiaCuda,
  kCpu,
};

std::unique_ptr<IPreprocessorBackend> createPreprocBackend(PreprocBackendType type = PreprocBackendType::kAuto);

PreprocBackendType detectAvailablePreprocBackend();
