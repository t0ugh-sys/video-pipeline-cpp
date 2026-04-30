#include "pipeline_runner.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

AppConfig makeRockchipConfig() {
  AppConfig config;
  config.source.uri = "input.mp4";
  config.model.modelPath = "model.rknn";
  config.model.inputWidth = 640;
  config.model.inputHeight = 640;
  config.decoderBackend = DecoderBackendType::kRockchipMpp;
  config.preprocBackend = PreprocBackendType::kRockchipRga;
  config.inferBackend = InferBackendType::kRockchipRknn;
  config.inferWorkers = 1;
  return config;
}

bool expectValidationFailure(const AppConfig& config, const std::string& needle, const std::string& message) {
  try {
    validateAppConfig(config);
    return expect(false, message);
  } catch (const std::exception& error) {
    return expect(std::string(error.what()).find(needle) != std::string::npos, message);
  }
}

bool testRejectsRockchipEncoderOutputH265() {
  AppConfig config = makeRockchipConfig();
  config.encoderOutput = "out.h264";
  config.encoderCodec = "h265";
  return expectValidationFailure(
      config,
      "does not support --encoder-codec h265 yet",
      "expected rockchip encoder-output h265 to be rejected");
}

bool testRejectsRockchipAnnotatedOutputH265() {
  AppConfig config = makeRockchipConfig();
  config.visual.outputVideo = "annotated.mp4";
  config.encoderCodec = "h265";
  return expectValidationFailure(
      config,
      "annotated output path does not support --encoder-codec h265 yet",
      "expected rockchip annotated output h265 to be rejected");
}

bool testRejectsRockchipAnnotatedHevcPath() {
  AppConfig config = makeRockchipConfig();
  config.visual.outputVideo = "annotated.hevc";
  return expectValidationFailure(
      config,
      "does not support .h265/.hevc output yet",
      "expected rockchip annotated .hevc path to be rejected");
}

bool testRejectsRockchipEncoderOutputHevcPath() {
  AppConfig config = makeRockchipConfig();
  config.encoderOutput = "raw.hevc";
  return expectValidationFailure(
      config,
      "does not support .h265/.hevc output yet",
      "expected rockchip encoder-output .hevc path to be rejected");
}

bool testRejectsInvalidOutputRtspScheme() {
  AppConfig config = makeRockchipConfig();
  config.visual.outputRtsp = "http://127.0.0.1/live";
  return expectValidationFailure(
      config,
      "output-rtsp must start with rtsp://",
      "expected invalid output-rtsp scheme to be rejected");
}

bool testRejectsConflictingAnnotatedTargets() {
  AppConfig config = makeRockchipConfig();
  config.visual.outputVideo = "annotated.mp4";
  config.visual.outputRtsp = "rtsp://127.0.0.1/live";
  return expectValidationFailure(
      config,
      "Specify only one annotated output target",
      "expected conflicting annotated targets to be rejected");
}

}  // namespace

int main() {
  bool ok = true;
  ok = ok && testRejectsRockchipEncoderOutputH265();
  ok = ok && testRejectsRockchipAnnotatedOutputH265();
  ok = ok && testRejectsRockchipAnnotatedHevcPath();
  ok = ok && testRejectsRockchipEncoderOutputHevcPath();
  ok = ok && testRejectsInvalidOutputRtspScheme();
  ok = ok && testRejectsConflictingAnnotatedTargets();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
