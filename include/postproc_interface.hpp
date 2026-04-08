#pragma once

#include "detection.hpp"
#include "pipeline_types.hpp"

#include <memory>
#include <string>

/**
 * 后处理抽象接口
 * 支持：YOLO v5/v8/v26 等模型的后处理
 */
class IPostprocessor {
 public:
  virtual ~IPostprocessor() = default;

  /**
   * 后处理推理输出
   * @param output 推理输出 tensor
   * @param modelWidth 模型输入宽度
   * @param modelHeight 模型输入高度
   * @param originalWidth 原图宽度
   * @param originalHeight 原图高度
   * @param pts 帧时间戳
   * @return 检测结果
   */
  virtual DetectionResult postprocess(
      const std::vector<float>& output,
      int modelWidth,
      int modelHeight,
      int originalWidth,
      int originalHeight,
      int64_t pts) = 0;

  /** 获取后端名称 */
  virtual std::string name() const = 0;
};

/**
 * 后处理类型枚举
 */
enum class PostprocBackendType {
  kAuto,      ///< 自动选择
  kYoloV8,    ///< YOLOv8 (84 = 4 bbox + 80 classes)
  kYolo26,    ///< YOLO26 (端到端无 NMS 或需要 NMS)
  kYoloV5,    ///< YOLOv5 (85 = 4 bbox + 1 conf + 80 classes)
};

/**
 * 创建后处理后端实例
 */
std::unique_ptr<IPostprocessor> createPostprocBackend(PostprocBackendType type = PostprocBackendType::kAuto);

/**
 * 检测当前平台可用的后处理后端
 */
PostprocBackendType detectAvailablePostprocBackend();
