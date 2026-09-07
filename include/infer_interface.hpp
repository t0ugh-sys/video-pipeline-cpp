#pragma once

#include "pipeline_types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

enum class RknnCoreMaskMode {
  kAuto,
  kCore0,
  kCore1,
  kCore2,
  kCore0_1,
  kCore0_2,
  kCore1_2,
  kCore0_1_2,
  kAll,
};

struct InferRuntimeConfig {
  int workerIndex = 0;
  int workerCount = 1;
  bool verbose = false;
  bool usePersistentInputMemory = false;
  RknnCoreMaskMode rknnCoreMask = RknnCoreMaskMode::kAuto;
};

struct InferenceInputMemory {
  int dmaFd = -1;
  void* virtAddr = nullptr;
  std::uint64_t physAddr = 0;
  int offset = 0;
  std::uint32_t flags = 0;
  void* privData = nullptr;
  std::size_t dmaSize = 0;
  int wstride = 0;
  int hstride = 0;

  bool valid() const { return dmaFd >= 0 && dmaSize > 0 && wstride > 0 && hstride > 0; }
};

class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  virtual void open(const ModelConfig& config, const InferRuntimeConfig& runtime = {}) = 0;

  virtual void close() = 0;

  virtual InferenceOutput infer(const RgbImage& image) = 0;

  virtual InferenceInputMemory inputMemory() const { return {}; }

  virtual int inputWidth() const = 0;

  virtual int inputHeight() const = 0;

  virtual std::string name() const = 0;
};

enum class InferBackendType {
  kAuto,
  kRockchipRknn,
  kNvidiaTrt,
  kOnnxRuntime,
};

std::unique_ptr<IInferenceBackend> createInferBackend(InferBackendType type = InferBackendType::kAuto);

InferBackendType detectAvailableInferBackend();
