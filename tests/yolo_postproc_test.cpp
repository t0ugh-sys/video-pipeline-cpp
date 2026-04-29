#include "backends/yolo_postproc.hpp"

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
        std::string(error.what()).find("currently unsupported") != std::string::npos,
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

}  // namespace

int main() {
  bool ok = true;
  ok = ok && testRejectsYolo26E2E();
  ok = ok && testRejectsUnknownSingleOutputAutoLayout();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
