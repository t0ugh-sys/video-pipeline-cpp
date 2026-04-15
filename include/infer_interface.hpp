#pragma once

#include "pipeline_types.hpp"

#include <memory>
#include <string>

class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  virtual void open(const ModelConfig& config) = 0;

  virtual InferenceOutput infer(const RgbImage& image) = 0;

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
