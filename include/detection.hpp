#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

/**
 * 检测框结构
 */
struct BoundingBox {
  float x1 = 0.0f;    // 左上角 x
  float y1 = 0.0f;    // 左上角 y
  float x2 = 0.0f;    // 右下角 x
  float y2 = 0.0f;    // 右下角 y
  float score = 0.0f; // 置信度 (class_conf)
  int classId = 0;    // 类别 ID
  std::string label;  // 类别名称

  float width() const { return x2 - x1; }
  float height() const { return y2 - y1; }
  float area() const { return width() * height(); }
};

/**
 * 单帧检测结果
 */
struct DetectionResult {
  int64_t pts = 0;                       // 时间戳
  std::vector<BoundingBox> boxes;         // 检测框列表
  int imageWidth = 0;                     // 原图宽度
  int imageHeight = 0;                   // 原图高度
};

/**
 * YOLO 模型类型
 */
enum class YoloVersion {
  kYolov8,   // 84 = 4 bbox + 80 classes, 需要 NMS (一对多头部)
  kYolo26,   // 端到端无 NMS (一对一头部): (N, 300, 6)
             // 传统模式 (一对多头部): (N, 84, 8400), 需要 NMS
};

/**
 * COCO 80 类标签
 */
static const std::vector<std::string> kCocoLabels = {
    "person",        "bicycle",       "car",           "motorcycle",
    "airplane",      "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "couch",         "potted plant",  "bed",
    "dining table",  "toilet",        "tv",            "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"};
