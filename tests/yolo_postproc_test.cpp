#include "backends/yolo_postproc.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

InferenceTensor makeTensor(std::vector<std::int64_t> shape) {
  InferenceTensor tensor;
  tensor.name = "test";
  tensor.layout = "NCHW";
  tensor.shape = std::move(shape);
  tensor.dataType = TensorDataType::kFloat32;
  tensor.quantization = TensorQuantizationType::kNone;
  return tensor;
}

RgbImage makeImage() {
  RgbImage image;
  image.width = 640;
  image.height = 640;
  image.wstride = 640;
  image.hstride = 640;
  image.format = PixelFormat::kRgb888;
  return image;
}

bool testRejectsYolo26E2E() {
  YoloPostprocessor postproc(YoloVersion::kYolo26, PostprocessOptions{});
  InferenceOutput output = {makeTensor({1, 300, 6})};
  try {
    (void)postproc.postprocess(output, makeImage(), 640, 640, 0);
    return expect(false, "expected yolo26_e2e to throw unsupported");
  } catch (const std::exception& error) {
    return expect(
        std::string(error.what()).find("YOLO26 E2E layout unsupported on RKNN") !=
            std::string::npos,
        "expected yolo26_e2e unsupported message");
  }
}

bool testRejectsUnknownSingleOutputAutoLayout() {
  YoloPostprocessor postproc(YoloVersion::kYolov8, PostprocessOptions{});
  InferenceOutput output = {makeTensor({1, 32, 100})};
  try {
    (void)postproc.postprocess(output, makeImage(), 640, 640, 0);
    return expect(false, "expected unknown single-output auto layout to throw");
  } catch (const std::exception& error) {
    return expect(
        std::string(error.what()).find("Unsupported single-output YOLO tensor in auto layout mode") !=
            std::string::npos,
        "expected unsupported single-output auto-layout message");
  }
}

BoundingBox makeBox(float x1, float y1, float x2, float y2, float score, int classId) {
  BoundingBox box;
  box.x1 = x1;
  box.y1 = y1;
  box.x2 = x2;
  box.y2 = y2;
  box.score = score;
  box.classId = classId;
  return box;
}

bool testIouIdentical() {
  const auto a = makeBox(10, 10, 50, 50, 0.9f, 0);
  const auto b = makeBox(10, 10, 50, 50, 0.8f, 0);
  const float iou = YoloPostprocessor::computeIoU(a, b);
  return expect(std::abs(iou - 1.0f) < 1e-5f, "IoU of identical boxes should be 1.0");
}

bool testIouNoOverlap() {
  const auto a = makeBox(0, 0, 10, 10, 0.9f, 0);
  const auto b = makeBox(20, 20, 30, 30, 0.8f, 0);
  const float iou = YoloPostprocessor::computeIoU(a, b);
  return expect(std::abs(iou) < 1e-5f, "IoU of non-overlapping boxes should be 0.0");
}

bool testIouPartialOverlap() {
  // Two 10x10 boxes overlapping in a 5x10 region => intersection=50, union=200-50=150
  const auto a = makeBox(0, 0, 10, 10, 0.9f, 0);
  const auto b = makeBox(5, 0, 15, 10, 0.8f, 0);
  const float iou = YoloPostprocessor::computeIoU(a, b);
  const float expected = 50.0f / 150.0f;
  return expect(std::abs(iou - expected) < 1e-5f,
                "IoU of half-overlapping boxes should be 1/3, got " + std::to_string(iou));
}

bool testIouContainment() {
  // Small box fully inside big box: intersection=area_small, union=area_big
  const auto big = makeBox(0, 0, 100, 100, 0.9f, 0);
  const auto small = makeBox(25, 25, 75, 75, 0.8f, 0);
  const float iou = YoloPostprocessor::computeIoU(big, small);
  const float expected = (50.0f * 50.0f) / (10000.0f + 2500.0f - 2500.0f);
  return expect(std::abs(iou - expected) < 1e-5f,
                "IoU of contained box should be 0.25, got " + std::to_string(iou));
}

bool testNmsRemovesDuplicates() {
  // Two highly overlapping boxes, same class — NMS should keep only the higher-scored one
  std::vector<BoundingBox> boxes = {
      makeBox(10, 10, 50, 50, 0.9f, 0),
      makeBox(12, 12, 52, 52, 0.7f, 0),
  };
  auto result = YoloPostprocessor::nms(boxes, 0.5f);
  return expect(result.size() == 1, "NMS should keep 1 box, got " + std::to_string(result.size())) &&
         expect(result[0].score == 0.9f, "NMS should keep the higher-scored box");
}

bool testNmsKeepsDifferentClasses() {
  // Two overlapping boxes but different classes — NMS should keep both
  std::vector<BoundingBox> boxes = {
      makeBox(10, 10, 50, 50, 0.9f, 0),
      makeBox(10, 10, 50, 50, 0.8f, 1),
  };
  auto result = YoloPostprocessor::nms(boxes, 0.5f);
  return expect(result.size() == 2,
                "NMS should keep both boxes for different classes, got " + std::to_string(result.size()));
}

bool testNmsKeepsNonOverlapping() {
  // Two non-overlapping boxes, same class — NMS should keep both
  std::vector<BoundingBox> boxes = {
      makeBox(0, 0, 10, 10, 0.9f, 0),
      makeBox(100, 100, 110, 110, 0.8f, 0),
  };
  auto result = YoloPostprocessor::nms(boxes, 0.5f);
  return expect(result.size() == 2,
                "NMS should keep both non-overlapping boxes, got " + std::to_string(result.size()));
}

bool testNmsKeepsHigherScore() {
  // Three boxes: A and B overlap, B and C overlap, A and C don't
  // A (score=0.95) suppresses B (score=0.8), C (score=0.7) survives
  std::vector<BoundingBox> boxes = {
      makeBox(10, 10, 50, 50, 0.95f, 0),
      makeBox(20, 20, 60, 60, 0.80f, 0),
      makeBox(200, 200, 240, 240, 0.70f, 0),
  };
  auto result = YoloPostprocessor::nms(boxes, 0.3f);
  return expect(result.size() == 2,
                "NMS should keep A and C, got " + std::to_string(result.size())) &&
         expect(result[0].score == 0.95f, "First result should be A (highest score)") &&
         expect(result[1].score == 0.70f, "Second result should be C");
}

bool testNmsEmptyInput() {
  std::vector<BoundingBox> boxes;
  auto result = YoloPostprocessor::nms(boxes, 0.5f);
  return expect(result.empty(), "NMS on empty input should return empty");
}

}  // namespace

int main() {
  bool ok = true;
  ok = ok && testRejectsYolo26E2E();
  ok = ok && testRejectsUnknownSingleOutputAutoLayout();
  ok = ok && testIouIdentical();
  ok = ok && testIouNoOverlap();
  ok = ok && testIouPartialOverlap();
  ok = ok && testIouContainment();
  ok = ok && testNmsRemovesDuplicates();
  ok = ok && testNmsKeepsDifferentClasses();
  ok = ok && testNmsKeepsNonOverlapping();
  ok = ok && testNmsKeepsHigherScore();
  ok = ok && testNmsEmptyInput();
  if (ok) {
    std::cout << "all yolo_postproc tests passed\n";
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
