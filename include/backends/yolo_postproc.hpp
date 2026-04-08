#pragma once

#include "../postproc_interface.hpp"

/**
 * YOLO 后处理实现
 * 支持 YOLOv8 (84=4+80), YOLO26, YOLOv5 (85=4+1+80)
 */
class YoloPostprocessor : public IPostprocessor {
 public:
  explicit YoloPostprocessor(YoloVersion version, float confThreshold = 0.25f, float iouThreshold = 0.45f);

  DetectionResult postprocess(
      const std::vector<float>& output,
      int modelWidth,
      int modelHeight,
      int originalWidth,
      int originalHeight,
      int64_t pts) override;

  std::string name() const override;

 private:
  /**
   * YOLOv8/YOLOv5 后处理 (需要 NMS)
   * 输出格式: (batch, 84, 8400) 或 (batch, 85, 8400)
   */
  DetectionResult postprocessYolov8(
      const std::vector<float>& output,
      int modelWidth,
      int modelHeight,
      int originalWidth,
      int originalHeight,
      int64_t pts);

  /**
   * YOLO26 端到端无 NMS 后处理 (一对一头部)
   * 输出格式: (batch, 300, 6) -> [x1, y1, x2, y2, conf, class]
   */
  DetectionResult postprocessYolo26E2E(
      const std::vector<float>& output,
      int modelWidth,
      int modelHeight,
      int originalWidth,
      int originalHeight,
      int64_t pts);

  /**
   * YOLO26 传统模式后处理 (需要 NMS)
   * 输出格式: (batch, 84, 8400)
   */
  DetectionResult postprocessYolo26Legacy(
      const std::vector<float>& output,
      int modelWidth,
      int modelHeight,
      int originalWidth,
      int originalHeight,
      int64_t pts);

  /**
   * 计算 IoU
   */
  static float computeIoU(const BoundingBox& a, const BoundingBox& b);

  /**
   * NMS 非极大值抑制
   */
  static std::vector<BoundingBox> nms(std::vector<BoundingBox>& boxes, float iouThreshold);

  /**
   * 坐标映射到原图尺寸
   */
  static void scaleBoxes(
      std::vector<BoundingBox>& boxes,
      int modelWidth,
      int modelHeight,
      int originalWidth,
      int originalHeight);

  YoloVersion version_;
  float confThreshold_;
  float iouThreshold_;
};
