#pragma once

#include "detection.hpp"
#include "pipeline_types.hpp"

#include <memory>
#include <string>
#include <vector>

enum class ModelOutputLayout {
  kAuto,
  kYolov8Flat,
  kYolov8RknnBranch6,
  kYolov8RknnBranch9,
  kYolo26E2E,
};

struct PostprocessOptions {
  float confThreshold = 0.25f;
  float nmsThreshold = 0.45f;
  std::string labelsPath;
  std::vector<std::string> labels;
  ModelOutputLayout outputLayout = ModelOutputLayout::kAuto;
  bool verbose = false;
};

class IPostprocessor {
 public:
  virtual ~IPostprocessor() = default;

  virtual DetectionResult postprocess(
      const InferenceOutput& output,
      const RgbImage& modelInput,
      int originalWidth,
      int originalHeight,
      int64_t pts) = 0;

  virtual std::string name() const = 0;
};

enum class PostprocBackendType {
  kAuto,
  kYoloV8,
  kYolo26,
  kYoloV5,
};

std::unique_ptr<IPostprocessor> createPostprocBackend(
    PostprocBackendType type = PostprocBackendType::kAuto,
    const PostprocessOptions& options = {});

PostprocBackendType detectAvailablePostprocBackend();
