#include "backends/yolo_postproc.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {

// 从文件加载类标签
std::vector<std::string> loadLabels(const std::string& path) {
  std::vector<std::string> labels;
  std::ifstream file(path);
  if (!file.is_open()) {
    return {};
  }
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

// 默认路径
constexpr const char* kDefaultLabelFile = "models/coco_labels.txt";

std::vector<std::string> getLabels() {
  static std::vector<std::string> labels = loadLabels(kDefaultLabelFile);
  return labels;
}

}  // namespace

YoloPostprocessor::YoloPostprocessor(
    YoloVersion version,
    float confThreshold,
    float iouThreshold)
    : version_(version), confThreshold_(confThreshold), iouThreshold_(iouThreshold) {}

std::string YoloPostprocessor::name() const {
  switch (version_) {
    case YoloVersion::kYolov8:
      return "YOLOv8";
    case YoloVersion::kYolo26:
      return "YOLO26";
    default:
      return "Unknown";
  }
}

DetectionResult YoloPostprocessor::postprocess(
    const std::vector<float>& output,
    int modelWidth,
    int modelHeight,
    int originalWidth,
    int originalHeight,
    int64_t pts) {
  switch (version_) {
    case YoloVersion::kYolov8:
      return postprocessYolov8(
          output, modelWidth, modelHeight, originalWidth, originalHeight, pts);
    case YoloVersion::kYolo26:
      // 默认使用端到端无 NMS 模式
      return postprocessYolo26E2E(
          output, modelWidth, modelHeight, originalWidth, originalHeight, pts);
    default:
      throw std::runtime_error("Unsupported YOLO version");
  }
}

DetectionResult YoloPostprocessor::postprocessYolov8(
    const std::vector<float>& output,
    int modelWidth,
    int modelHeight,
    int originalWidth,
    int originalHeight,
    int64_t pts) {
  DetectionResult result;
  result.pts = pts;
  result.imageWidth = originalWidth;
  result.imageHeight = originalHeight;

  // YOLOv8 输出: (batch, 84, 8400)
  // 84 = 4 (bbox) + 80 (classes)
  const int kNumClasses = 80;
  const int kNumAnchors = 8400;
  const int kNumProposals = kNumAnchors;

  std::vector<BoundingBox> boxes;
  boxes.reserve(kNumProposals);

  for (int i = 0; i < kNumProposals; ++i) {
    // 获取 bbox 和类别分数
    const float* ptr = output.data() + i * (4 + kNumClasses);

    // bbox: cx, cy, w, h (归一化)
    float cx = ptr[0];
    float cy = ptr[1];
    float w = ptr[2];
    float h = ptr[3];

    // 转换为绝对坐标
    cx *= modelWidth;
    cy *= modelHeight;
    w *= modelWidth;
    h *= modelHeight;

    // 计算左上角和右下角
    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    // 找到最高分数的类别
    float maxScore = 0.0f;
    int maxClassId = 0;
    for (int c = 0; c < kNumClasses; ++c) {
      float score = ptr[4 + c];
      if (score > maxScore) {
        maxScore = score;
        maxClassId = c;
      }
    }

    // 置信度过滤
    if (maxScore < confThreshold_) {
      continue;
    }

    BoundingBox box;
    box.x1 = x1;
    box.y1 = y1;
    box.x2 = x2;
    box.y2 = y2;
    box.score = maxScore;
    box.classId = maxClassId;
    auto labels = getLabels();
    if (maxClassId >= 0 && maxClassId < static_cast<int>(labels.size())) {
      box.label = labels[maxClassId];
    }
    boxes.push_back(box);
  }

  // NMS
  boxes = nms(boxes, iouThreshold_);

  // 坐标映射到原图
  scaleBoxes(boxes, modelWidth, modelHeight, originalWidth, originalHeight);

  result.boxes = std::move(boxes);
  return result;
}

DetectionResult YoloPostprocessor::postprocessYolo26E2E(
    const std::vector<float>& output,
    int modelWidth,
    int modelHeight,
    int originalWidth,
    int originalHeight,
    int64_t pts) {
  DetectionResult result;
  result.pts = pts;
  result.imageWidth = originalWidth;
  result.imageHeight = originalHeight;

  // YOLO26 端到端输出: (batch, 300, 6)
  // 6 = x1, y1, x2, y2, conf, class
  const int kMaxDetections = 300;
  const int kNumOutput = 6;

  std::vector<BoundingBox> boxes;

  for (int i = 0; i < kMaxDetections; ++i) {
    const float* ptr = output.data() + i * kNumOutput;

    float conf = ptr[4];
    if (conf < confThreshold_) {
      continue;
    }

    BoundingBox box;
    box.x1 = ptr[0];
    box.y1 = ptr[1];
    box.x2 = ptr[2];
    box.y2 = ptr[3];
    box.score = conf;
    box.classId = static_cast<int>(ptr[5]);

    auto labels = getLabels();
    if (box.classId >= 0 && box.classId < static_cast<int>(labels.size())) {
      box.label = labels[box.classId];
    }

    boxes.push_back(box);
  }

  // 端到端模式不需要 NMS，直接映射坐标
  scaleBoxes(boxes, modelWidth, modelHeight, originalWidth, originalHeight);

  result.boxes = std::move(boxes);
  return result;
}

DetectionResult YoloPostprocessor::postprocessYolo26Legacy(
    const std::vector<float>& output,
    int modelWidth,
    int modelHeight,
    int originalWidth,
    int originalHeight,
    int64_t pts) {
  // YOLO26 传统模式与 YOLOv8 相同
  return postprocessYolov8(output, modelWidth, modelHeight, originalWidth, originalHeight, pts);
}

float YoloPostprocessor::computeIoU(const BoundingBox& a, const BoundingBox& b) {
  // 计算交集区域
  float x1 = std::max(a.x1, b.x1);
  float y1 = std::max(a.y1, b.y1);
  float x2 = std::min(a.x2, b.x2);
  float y2 = std::min(a.y2, b.y2);

  float interW = std::max(0.0f, x2 - x1);
  float interH = std::max(0.0f, y2 - y1);
  float interArea = interW * interH;

  // 计算并集区域
  float areaA = a.area();
  float areaB = b.area();
  float unionArea = areaA + areaB - interArea;

  if (unionArea <= 0.0f) {
    return 0.0f;
  }

  return interArea / unionArea;
}

std::vector<BoundingBox> YoloPostprocessor::nms(std::vector<BoundingBox>& boxes, float iouThreshold) {
  if (boxes.empty()) {
    return {};
  }

  // 按置信度降序排序
  std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
    return a.score > b.score;
  });

  std::vector<BoundingBox> result;
  std::vector<bool> suppressed(boxes.size(), false);

  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }

    result.push_back(boxes[i]);

    for (std::size_t j = i + 1; j < boxes.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      // 同一类别才做 NMS
      if (boxes[i].classId != boxes[j].classId) {
        continue;
      }

      float iou = computeIoU(boxes[i], boxes[j]);
      if (iou > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

void YoloPostprocessor::scaleBoxes(
    std::vector<BoundingBox>& boxes,
    int modelWidth,
    int modelHeight,
    int originalWidth,
    int originalHeight) {
  if (boxes.empty()) {
    return;
  }

  float scaleX = static_cast<float>(originalWidth) / static_cast<float>(modelWidth);
  float scaleY = static_cast<float>(originalHeight) / static_cast<float>(modelHeight);

  for (auto& box : boxes) {
    box.x1 *= scaleX;
    box.y1 *= scaleY;
    box.x2 *= scaleX;
    box.y2 *= scaleY;
  }
}
