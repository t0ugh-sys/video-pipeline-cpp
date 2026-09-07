#pragma once

#include "pipeline_types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct PreprocessOptions {
  bool letterbox = false;
  std::uint8_t paddingValue = 114;
  bool needsCpuData = false;
  // Optional externally-owned RGB target. The RGA backend writes directly
  // into this DMA-BUF when dmaFd is valid.
  int outputDmaFd = -1;
  std::size_t outputDmaSize = 0;
  void* outputVirtAddr = nullptr;
  std::uint64_t outputPhysAddr = 0;
  int outputOffset = 0;
  std::uint32_t outputFlags = 0;
  void* outputPrivData = nullptr;
  int outputWstride = 0;
  int outputHstride = 0;
  bool strictZeroCopy = false;
};

class IPreprocessorBackend {
 public:
  virtual ~IPreprocessorBackend() = default;

  virtual RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight,
      const PreprocessOptions& options = {}) = 0;

  virtual void setMaxInflightFrames(std::size_t maxInflightFrames) {
    (void)maxInflightFrames;
  }

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
