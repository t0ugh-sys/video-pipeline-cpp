#include "encoder_timing.hpp"

#include <algorithm>

int resolveEncoderFps(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo) {
  if (config.encoderFps > 0) {
    return config.encoderFps;
  }
  if (sourceVideoInfo.fpsNum > 0 && sourceVideoInfo.fpsDen > 0) {
    return std::max(1, (sourceVideoInfo.fpsNum + sourceVideoInfo.fpsDen / 2) / sourceVideoInfo.fpsDen);
  }
  return 30;
}

int resolveEncoderFpsNum(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo) {
  if (config.encoderFps > 0) {
    return config.encoderFps;
  }
  if (sourceVideoInfo.fpsNum > 0 && sourceVideoInfo.fpsDen > 0) {
    return sourceVideoInfo.fpsNum;
  }
  return 30;
}

int resolveEncoderFpsDen(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo) {
  if (config.encoderFps > 0) {
    return 1;
  }
  if (sourceVideoInfo.fpsNum > 0 && sourceVideoInfo.fpsDen > 0) {
    return sourceVideoInfo.fpsDen;
  }
  return 1;
}

bool shouldKeepEncodedFrame(
    std::size_t zeroBasedFrameIndex,
    const SourceVideoInfo& sourceVideoInfo,
    int targetFps) {
  if (zeroBasedFrameIndex == 0) {
    return true;
  }
  if (targetFps <= 0 || sourceVideoInfo.fpsNum <= 0 || sourceVideoInfo.fpsDen <= 0) {
    return true;
  }

  const long long sourceNum = static_cast<long long>(sourceVideoInfo.fpsNum);
  const long long sourceDen = static_cast<long long>(sourceVideoInfo.fpsDen);
  const long long targetNum = static_cast<long long>(targetFps) * sourceDen;
  if (targetNum >= sourceNum) {
    return true;
  }

  const long long prevCount =
      (static_cast<long long>(zeroBasedFrameIndex) * targetNum) / sourceNum;
  const long long currCount =
      (static_cast<long long>(zeroBasedFrameIndex + 1) * targetNum) / sourceNum;
  return currCount != prevCount;
}
