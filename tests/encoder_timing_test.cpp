#include "app_config.hpp"
#include "encoder_timing.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void expectTrue(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

void expectEqual(int actual, int expected, const std::string& message) {
  if (actual != expected) {
    std::cerr << "FAILED: " << message << " expected=" << expected << " actual=" << actual << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  AppConfig config;
  SourceVideoInfo highFps;
  highFps.fpsNum = 5625;
  highFps.fpsDen = 32;

  expectEqual(resolveEncoderFps(config, highFps), 176, "default encoder fps should round source fps");
  expectEqual(resolveEncoderFpsNum(config, highFps), 5625, "default encoder fps numerator should preserve source");
  expectEqual(resolveEncoderFpsDen(config, highFps), 32, "default encoder fps denominator should preserve source");
  expectTrue(shouldKeepEncodedFrame(0, highFps, resolveEncoderFps(config, highFps)),
             "first frame should be kept");
  expectTrue(shouldKeepEncodedFrame(1, highFps, resolveEncoderFps(config, highFps)),
             "source-fps passthrough should keep subsequent frames");
  expectTrue(shouldKeepEncodedFrame(179, highFps, resolveEncoderFps(config, highFps)),
             "source-fps passthrough should keep all frames");

  config.encoderFps = 30;
  expectEqual(resolveEncoderFps(config, highFps), 30, "explicit encoder fps should override source fps");
  expectEqual(resolveEncoderFpsNum(config, highFps), 30, "explicit encoder fps numerator should be fixed");
  expectEqual(resolveEncoderFpsDen(config, highFps), 1, "explicit encoder fps denominator should be fixed");
  expectTrue(!shouldKeepEncodedFrame(1, highFps, resolveEncoderFps(config, highFps)),
             "explicit 30fps override should drop frames on high-fps input");
  expectTrue(shouldKeepEncodedFrame(5, highFps, resolveEncoderFps(config, highFps)),
             "explicit 30fps override should still keep some frames");

  SourceVideoInfo ntsc;
  ntsc.fpsNum = 30000;
  ntsc.fpsDen = 1001;
  config.encoderFps = 0;
  expectEqual(resolveEncoderFps(config, ntsc), 30, "29.97fps source should round to 30 for bitrate heuristics");
  expectEqual(resolveEncoderFpsNum(config, ntsc), 30000, "29.97fps numerator should be preserved");
  expectEqual(resolveEncoderFpsDen(config, ntsc), 1001, "29.97fps denominator should be preserved");
  expectTrue(shouldKeepEncodedFrame(1, ntsc, resolveEncoderFps(config, ntsc)),
             "29.97fps source should keep frames by default");

  return 0;
}
