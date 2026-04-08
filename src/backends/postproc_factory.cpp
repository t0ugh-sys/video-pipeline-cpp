#include "postproc_interface.hpp"
#include "backends/yolo_postproc.hpp"

#include <stdexcept>

PostprocBackendType detectAvailablePostprocBackend() {
  // 默认自动选择
  return PostprocBackendType::kAuto;
}

std::unique_ptr<IPostprocessor> createPostprocBackend(PostprocBackendType type) {
  if (type == PostprocBackendType::kAuto) {
    // 默认使用 YOLOv8
    type = PostprocBackendType::kYoloV8;
  }

  switch (type) {
    case PostprocBackendType::kYoloV8:
      return std::make_unique<YoloPostprocessor>(YoloVersion::kYolov8);

    case PostprocBackendType::kYolo26:
      return std::make_unique<YoloPostprocessor>(YoloVersion::kYolo26);

    default:
      throw std::runtime_error("Unknown postproc backend type");
  }
}
