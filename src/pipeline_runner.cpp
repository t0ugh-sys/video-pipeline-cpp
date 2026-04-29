#include "pipeline_runner.hpp"

#include "backend_registry.hpp"
#include "decoder_interface.hpp"
#include "encoder_interface.hpp"
#include "ffmpeg_packet_source.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"
#include "visualizer.hpp"

#include "../../rknn_model_zoo/utils/font.h"

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
extern "C" {
#include <mpp_buffer.h>
}
#include <im2d.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct PreparedFrame {
  std::size_t index = 0;
  int64_t pts = 0;
  int originalWidth = 0;
  int originalHeight = 0;
  DecodedFrame decodedFrame;
  RgbImage inferenceImage;
  double decodeMs = 0.0;
  double preprocMs = 0.0;
};

struct ProcessedFrame {
  std::size_t index = 0;
  int64_t pts = 0;
  DecodedFrame decodedFrame;
  DetectionResult result;
  double decodeMs = 0.0;
  double preprocMs = 0.0;
  double inferMs = 0.0;
  double postprocMs = 0.0;
};

constexpr int kModelZooBoxThickness = 3;
constexpr int kModelZooFontPixelSize = 10;

struct RgbColor {
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
};

constexpr RgbColor kUltralyticsPalette[] = {
    {4, 42, 255},   {11, 219, 235}, {243, 243, 243}, {0, 223, 183},  {17, 31, 104},
    {255, 111, 221}, {255, 68, 79}, {204, 237, 0},   {0, 243, 68},   {189, 0, 255},
    {0, 180, 255},  {221, 0, 186}, {0, 255, 255},   {38, 192, 0},   {1, 255, 179},
    {125, 36, 255}, {123, 0, 104}, {255, 27, 108},  {252, 109, 47}, {162, 255, 11},
};

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  void push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    notFull_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
    if (closed_) {
      throw std::runtime_error("queue closed");
    }
    queue_.push_back(std::move(value));
    notEmpty_.notify_one();
  }

  bool pop(T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    notEmpty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    value = std::move(queue_.front());
    queue_.pop_front();
    notFull_.notify_one();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    notEmpty_.notify_all();
    notFull_.notify_all();
  }

 private:
  std::size_t capacity_;
  std::deque<T> queue_;
  bool closed_ = false;
  std::mutex mutex_;
  std::condition_variable notEmpty_;
  std::condition_variable notFull_;
};

template <typename BackendType>
void requireCompiledIn(
    BackendType type,
    const char* stageName,
    bool (*predicate)(BackendType),
    std::string (*availableFn)(),
    std::string (*nameFn)(BackendType)) {
  if (type == BackendType::kAuto || predicate(type)) {
    return;
  }

  throw std::runtime_error(
      std::string("Requested ") + stageName + " backend '" + nameFn(type) +
      "' is not available in this build. Available: " + availableFn());
}

void maybeDumpFirstFrame(const AppConfig& config, const RgbImage& image, std::size_t frameCount) {
  if (!config.dumpFirstFrame || frameCount != 1) {
    return;
  }

  const std::string path = "dump_first_frame.ppm";
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open dump_first_frame.ppm for writing");
  }

  output << "P6\n" << image.width << " " << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.data.data()), static_cast<std::streamsize>(image.data.size()));
}

PostprocessOptions makePostprocessOptions(const AppConfig& config) {
  return PostprocessOptions{
      config.confThreshold,
      config.nmsThreshold,
      config.labelsPath,
      {},
      config.modelOutputLayout,
      config.verbose};
}

std::pair<int, int> computeDisplaySize(const AppConfig& config, int sourceWidth, int sourceHeight) {
  const int maxWidth = config.visual.displayMaxWidth;
  const int maxHeight = config.visual.displayMaxHeight;
  if (maxWidth <= 0 && maxHeight <= 0) {
    return {sourceWidth, sourceHeight};
  }

  float scale = 1.0f;
  if (maxWidth > 0) {
    scale = std::min(scale, static_cast<float>(maxWidth) / static_cast<float>(sourceWidth));
  }
  if (maxHeight > 0) {
    scale = std::min(scale, static_cast<float>(maxHeight) / static_cast<float>(sourceHeight));
  }
  scale = std::min(scale, 1.0f);

  const int width = std::max(1, static_cast<int>(sourceWidth * scale));
  const int height = std::max(1, static_cast<int>(sourceHeight * scale));
  return {width, height};
}

DetectionResult scaleDetectionResult(const DetectionResult& result, int targetWidth, int targetHeight) {
  if (result.imageWidth <= 0 || result.imageHeight <= 0 ||
      (result.imageWidth == targetWidth && result.imageHeight == targetHeight)) {
    return result;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(result.imageWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(result.imageHeight);

  DetectionResult scaled = result;
  scaled.imageWidth = targetWidth;
  scaled.imageHeight = targetHeight;
  for (auto& box : scaled.boxes) {
    box.x1 *= scaleX;
    box.x2 *= scaleX;
    box.y1 *= scaleY;
    box.y2 *= scaleY;
  }
  return scaled;
}

RknnCoreMaskMode resolveAutoRknnCoreMask(int workerIndex, int workerCount) {
  switch (workerCount) {
    case 1:
      return RknnCoreMaskMode::kCore0_1_2;
    case 2:
      return workerIndex == 0 ? RknnCoreMaskMode::kCore0_1 : RknnCoreMaskMode::kCore2;
    case 3:
      if (workerIndex == 0) return RknnCoreMaskMode::kCore0;
      if (workerIndex == 1) return RknnCoreMaskMode::kCore1;
      return RknnCoreMaskMode::kCore2;
    default:
      return RknnCoreMaskMode::kAuto;
  }
}

InferRuntimeConfig makeInferRuntimeConfig(const AppConfig& config, int workerIndex, int workerCount) {
  InferRuntimeConfig runtime;
  runtime.workerIndex = workerIndex;
  runtime.workerCount = std::max(1, workerCount);
  runtime.verbose = config.verbose;
  runtime.rknnCoreMask =
      config.rknnCoreMask == RknnCoreMaskMode::kAuto
          ? resolveAutoRknnCoreMask(workerIndex, runtime.workerCount)
          : config.rknnCoreMask;
  return runtime;
}

std::string toLowerAscii(std::string value) {
  for (char& ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return value;
}

bool hasSuffixIgnoreCase(const std::string& value, const std::string& suffix) {
  const std::string lowerValue = toLowerAscii(value);
  const std::string lowerSuffix = toLowerAscii(suffix);
  return lowerValue.size() >= lowerSuffix.size() &&
         lowerValue.compare(lowerValue.size() - lowerSuffix.size(), lowerSuffix.size(), lowerSuffix) == 0;
}

bool startsWithIgnoreCase(const std::string& value, const std::string& prefix) {
  const std::string lowerValue = toLowerAscii(value);
  const std::string lowerPrefix = toLowerAscii(prefix);
  return lowerValue.size() >= lowerPrefix.size() &&
         lowerValue.compare(0, lowerPrefix.size(), lowerPrefix) == 0;
}

bool isRtspUrl(const std::string& value) {
  return startsWithIgnoreCase(value, "rtsp://");
}

std::string annotatedOutputTarget(const AppConfig& config) {
  if (!config.visual.outputRtsp.empty()) {
    return config.visual.outputRtsp;
  }
  return config.visual.outputVideo;
}

bool hasAnnotatedOutputTarget(const AppConfig& config) {
  return !annotatedOutputTarget(config).empty();
}

PostprocBackendType resolvePostprocBackend(const AppConfig& config) {
  if (config.postprocBackend != PostprocBackendType::kAuto) {
    return config.postprocBackend;
  }

  if (config.modelOutputLayout == ModelOutputLayout::kYolo26E2E) {
    return PostprocBackendType::kYolo26;
  }

  return PostprocBackendType::kYoloV8;
}

const char* toModelOutputLayoutName(ModelOutputLayout layout) {
  switch (layout) {
    case ModelOutputLayout::kAuto:
      return "auto";
    case ModelOutputLayout::kYolov8Flat:
      return "yolov8_flat_8400x84";
    case ModelOutputLayout::kYolov8RknnBranch6:
      return "yolov8_rknn_branch_6";
    case ModelOutputLayout::kYolov8RknnBranch9:
      return "yolov8_rknn_branch_9";
    case ModelOutputLayout::kYolo26E2E:
      return "yolo26_e2e";
  }
  return "unknown";
}

bool wantsHardwareEncodedAnnotatedOutput(const AppConfig& config) {
  const std::string outputTarget = annotatedOutputTarget(config);
  if (outputTarget.empty()) {
    return false;
  }

  if (isRtspUrl(outputTarget)) {
    return true;
  }

  return hasSuffixIgnoreCase(outputTarget, ".h264") ||
         hasSuffixIgnoreCase(outputTarget, ".264") ||
         hasSuffixIgnoreCase(outputTarget, ".h265") ||
         hasSuffixIgnoreCase(outputTarget, ".hevc") ||
         hasSuffixIgnoreCase(outputTarget, ".mp4");
}

bool usesRgaAnnotatedOutput(const AppConfig& config) {
  return wantsHardwareEncodedAnnotatedOutput(config) &&
         config.visual.outputOverlayMode == OutputOverlayMode::kRga;
}

bool backendUsesNvidiaEncoder() {
  return detectAvailableEncoderBackend() == EncoderBackendType::kNvidiaNvEnc;
}

int resolveEncoderFps(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo) {
  if (config.encoderFps > 0) {
    return config.encoderFps;
  }
  if (sourceVideoInfo.fpsNum > 0 && sourceVideoInfo.fpsDen > 0) {
    const int fps = std::max(1, (sourceVideoInfo.fpsNum + sourceVideoInfo.fpsDen / 2) / sourceVideoInfo.fpsDen);
    if (fps > 60) {
      return 30;
    }
    return fps;
  }
  return 30;
}

int resolveEncoderFpsNum(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo) {
  if (config.encoderFps > 0) {
    return config.encoderFps;
  }
  if (sourceVideoInfo.fpsNum > 0 && sourceVideoInfo.fpsDen > 0) {
    const int fps = std::max(1, (sourceVideoInfo.fpsNum + sourceVideoInfo.fpsDen / 2) / sourceVideoInfo.fpsDen);
    if (fps > 60) {
      return 30;
    }
    return sourceVideoInfo.fpsNum;
  }
  return 30;
}

int resolveEncoderFpsDen(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo) {
  if (config.encoderFps > 0) {
    return 1;
  }
  if (sourceVideoInfo.fpsNum > 0 && sourceVideoInfo.fpsDen > 0) {
    const int fps = std::max(1, (sourceVideoInfo.fpsNum + sourceVideoInfo.fpsDen / 2) / sourceVideoInfo.fpsDen);
    if (fps > 60) {
      return 1;
    }
    return sourceVideoInfo.fpsDen;
  }
  return 1;
}

int computeAutoEncoderBitrate(int width, int height, int fps) {
  const long long pixelsPerSecond =
      static_cast<long long>(std::max(1, width)) *
      static_cast<long long>(std::max(1, height)) *
      static_cast<long long>(std::max(1, fps));
  const long long estimated = pixelsPerSecond / 10;
  const long long clamped = std::clamp<long long>(estimated, 4'000'000LL, 40'000'000LL);
  return static_cast<int>(clamped);
}

int resolveEncoderBitrate(const AppConfig& config, int width, int height, int fps) {
  if (config.encoderBitrate > 0) {
    return config.encoderBitrate;
  }
  return computeAutoEncoderBitrate(width, height, fps);
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

  // Map the input frame index onto the target output timeline using integer
  // accumulation. We keep only frames that advance the output frame count, so
  // high-fps sources can be encoded at a sane output fps without slow-motion.
  const long long prevCount =
      (static_cast<long long>(zeroBasedFrameIndex) * targetNum) / sourceNum;
  const long long currCount =
      (static_cast<long long>(zeroBasedFrameIndex + 1) * targetNum) / sourceNum;
  return currCount != prevCount;
}

void fillRgbaRect(
    std::vector<std::uint8_t>& rgba,
    int imageWidth,
    int imageHeight,
    int x,
    int y,
    int width,
    int height,
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b,
    std::uint8_t a) {
  const int x0 = std::clamp(x, 0, imageWidth);
  const int y0 = std::clamp(y, 0, imageHeight);
  const int x1 = std::clamp(x + width, 0, imageWidth);
  const int y1 = std::clamp(y + height, 0, imageHeight);
  if (x0 >= x1 || y0 >= y1) {
    return;
  }

  for (int yy = y0; yy < y1; ++yy) {
    for (int xx = x0; xx < x1; ++xx) {
      const std::size_t offset =
          static_cast<std::size_t>((yy * imageWidth + xx) * 4);
      rgba[offset + 0] = r;
      rgba[offset + 1] = g;
      rgba[offset + 2] = b;
      rgba[offset + 3] = a;
    }
  }
}

RgbColor ultralyticsColorForClass(int classId) {
  const std::size_t paletteSize = sizeof(kUltralyticsPalette) / sizeof(kUltralyticsPalette[0]);
  const std::size_t index =
      static_cast<std::size_t>(classId >= 0 ? classId : 0) % paletteSize;
  return kUltralyticsPalette[index];
}

RgbColor ultralyticsTextColor(const RgbColor& background) {
  const int luminance = static_cast<int>(background.r) + static_cast<int>(background.g) + static_cast<int>(background.b);
  if (luminance >= 600) {
    return {16, 16, 16};
  }
  return {255, 255, 255};
}

int resizeNearestC1(
    const unsigned char* srcPixels,
    int srcWidth,
    int srcHeight,
    unsigned char* dstPixels,
    int dstWidth,
    int dstHeight) {
  for (int i = 0; i < dstHeight; ++i) {
    const int y = std::clamp((i * srcHeight) / dstHeight, 0, srcHeight - 1);
    for (int j = 0; j < dstWidth; ++j) {
      const int x = std::clamp((j * srcWidth) / dstWidth, 0, srcWidth - 1);
      dstPixels[i * dstWidth + j] = srcPixels[y * srcWidth + x];
    }
  }

  return 0;
}

void drawRectangleOnOverlay(
    std::vector<std::uint8_t>& rgba,
    int imageWidth,
    int imageHeight,
    int x,
    int y,
    int width,
    int height,
    int thickness,
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b,
    std::uint8_t a) {
  if (width <= 0 || height <= 0 || thickness <= 0) {
    return;
  }

  fillRgbaRect(rgba, imageWidth, imageHeight, x, y, width, thickness, r, g, b, a);
  fillRgbaRect(rgba, imageWidth, imageHeight, x, y + height - thickness, width, thickness, r, g, b, a);
  fillRgbaRect(rgba, imageWidth, imageHeight, x, y, thickness, height, r, g, b, a);
  fillRgbaRect(rgba, imageWidth, imageHeight, x + width - thickness, y, thickness, height, r, g, b, a);
}

void drawTextOnOverlay(
    std::vector<std::uint8_t>& rgba,
    int width,
    int height,
    const char* text,
    int x,
    int y,
    int fontPixelSize,
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b) {
  std::vector<unsigned char> resizedFontBitmap(
      static_cast<std::size_t>(fontPixelSize * fontPixelSize * 2));

  const int n = static_cast<int>(std::strlen(text));
  int cursorX = x;
  int cursorY = y;
  for (int i = 0; i < n; ++i) {
    const char ch = text[i];
    if (ch == '\n') {
      cursorX = x;
      cursorY += fontPixelSize * 2;
      continue;
    }
    if (std::isprint(static_cast<unsigned char>(ch)) == 0) {
      continue;
    }

    const int fontBitmapIndex = ch - ' ';
    if (fontBitmapIndex < 0 || fontBitmapIndex >= 95) {
      continue;
    }
    const unsigned char* fontBitmap = mono_font_data[fontBitmapIndex];
    resizeNearestC1(fontBitmap, 20, 40, resizedFontBitmap.data(), fontPixelSize, fontPixelSize * 2);

    for (int yy = cursorY; yy < cursorY + fontPixelSize * 2; ++yy) {
      if (yy < 0) {
        continue;
      }
      if (yy >= height) {
        break;
      }

      const unsigned char* alpha = resizedFontBitmap.data() +
          static_cast<std::size_t>(yy - cursorY) * fontPixelSize;
      for (int xx = cursorX; xx < cursorX + fontPixelSize; ++xx) {
        if (xx < 0) {
          continue;
        }
        if (xx >= width) {
          break;
        }

        const unsigned char a = alpha[xx - cursorX] >= 128 ? 255 : 0;
        const std::size_t offset =
            static_cast<std::size_t>((yy * width + xx) * 4);
        rgba[offset + 0] = static_cast<unsigned char>((rgba[offset + 0] * (255 - a) + r * a) / 255);
        rgba[offset + 1] = static_cast<unsigned char>((rgba[offset + 1] * (255 - a) + g * a) / 255);
        rgba[offset + 2] = static_cast<unsigned char>((rgba[offset + 2] * (255 - a) + b * a) / 255);
        rgba[offset + 3] = std::max(rgba[offset + 3], a);
      }
    }

    cursorX += fontPixelSize;
  }
}

void drawYoloLabelBoxOnOverlay(
    std::vector<std::uint8_t>& rgba,
    int imageWidth,
    int imageHeight,
    const char* text,
    int anchorX,
    int anchorY,
    int fontPixelSize,
    std::uint8_t bgR,
    std::uint8_t bgG,
    std::uint8_t bgB,
    std::uint8_t bgA,
    std::uint8_t textR,
    std::uint8_t textG,
    std::uint8_t textB) {
  if (text == nullptr || text[0] == '\0') {
    return;
  }

  const int textLength = static_cast<int>(std::strlen(text));
  const int charWidth = std::max(6, fontPixelSize);
  const int padX = std::max(4, fontPixelSize / 2);
  const int padY = std::max(3, fontPixelSize / 3);
  const int labelWidth = textLength * charWidth + padX * 2;
  const int labelHeight = fontPixelSize * 2 + padY * 2;

  int labelX = std::clamp(anchorX, 0, std::max(0, imageWidth - 1));
  int labelY = anchorY - labelHeight;
  if (labelY < 0) {
    labelY = std::clamp(anchorY + 1, 0, std::max(0, imageHeight - 1));
  }

  fillRgbaRect(
      rgba,
      imageWidth,
      imageHeight,
      labelX,
      labelY,
      std::min(labelWidth, std::max(0, imageWidth - labelX)),
      std::min(labelHeight, std::max(0, imageHeight - labelY)),
      bgR,
      bgG,
      bgB,
      bgA);
  drawTextOnOverlay(
      rgba,
      imageWidth,
      imageHeight,
      text,
      labelX + padX,
      labelY + padY,
      fontPixelSize,
      textR,
      textG,
      textB);
}

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
std::shared_ptr<void> makeMppBufferOwner(MppBuffer buffer) {
  return std::shared_ptr<void>(buffer, [](void* opaque) {
    if (opaque != nullptr) {
      mpp_buffer_put(static_cast<MppBuffer>(opaque));
    }
  });
}

DecodedFrame makeAnnotatedEncodeFrame(
    const DecodedFrame& frame,
    const DetectionResult& result,
    const VisualConfig& config) {
  if (frame.dmaFd < 0 || result.boxes.empty()) {
    return frame;
  }

  // Keep these groups alive across frames so the hardware overlay path does not
  // churn DRM buffers on every annotated output frame.
  static MppBufferGroup outputGroup = nullptr;
  if (outputGroup == nullptr) {
    if (mpp_buffer_group_get_internal(
            &outputGroup,
            static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)) != MPP_OK) {
      throw std::runtime_error("mpp_buffer_group_get_internal failed for hardware overlay output");
    }
  }
  static MppBufferGroup rgbaGroup = nullptr;
  if (rgbaGroup == nullptr) {
    if (mpp_buffer_group_get_internal(
            &rgbaGroup,
            static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)) != MPP_OK) {
      throw std::runtime_error("mpp_buffer_group_get_internal failed for hardware overlay RGBA scratch");
    }
  }

  const int horStride = frame.horizontalStride > 0 ? frame.horizontalStride : frame.width;
  const int verStride = frame.verticalStride > 0 ? frame.verticalStride : frame.height;
  const std::size_t outputBytes =
      static_cast<std::size_t>(horStride) * static_cast<std::size_t>(verStride) * 3 / 2;
  const std::size_t rgbaBytes =
      static_cast<std::size_t>(horStride) * static_cast<std::size_t>(verStride) * 4;
  const std::size_t overlayBytes =
      static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height) * 4;

  MppBuffer outputBuffer = nullptr;
  if (mpp_buffer_get(outputGroup, &outputBuffer, outputBytes) != MPP_OK || outputBuffer == nullptr) {
    throw std::runtime_error("mpp_buffer_get failed for hardware overlay NV12 output buffer");
  }
  MppBuffer rgbaBuffer = nullptr;
  if (mpp_buffer_get(rgbaGroup, &rgbaBuffer, rgbaBytes) != MPP_OK || rgbaBuffer == nullptr) {
    mpp_buffer_put(outputBuffer);
    throw std::runtime_error("mpp_buffer_get failed for hardware overlay RGBA scratch buffer");
  }

  std::vector<std::uint8_t> overlayData(overlayBytes, 0);
  rga_buffer_handle_t srcHandle = 0;
  rga_buffer_handle_t overlayHandle = 0;
  rga_buffer_handle_t rgbaHandle = 0;
  rga_buffer_handle_t outputHandle = 0;
  try {
    srcHandle = importbuffer_fd(frame.dmaFd, horStride, verStride, RK_FORMAT_YCbCr_420_SP);
    if (srcHandle == 0) {
      throw std::runtime_error("RGA importbuffer_fd failed for hardware overlay source");
    }

    overlayHandle = importbuffer_virtualaddr(overlayData.data(), static_cast<int>(overlayBytes));
    if (overlayHandle == 0) {
      throw std::runtime_error("RGA importbuffer_virtualaddr failed for hardware overlay RGBA handle");
    }

    const int rgbaFd = mpp_buffer_get_fd(rgbaBuffer);
    if (rgbaFd < 0) {
      throw std::runtime_error("mpp_buffer_get_fd failed for hardware overlay RGBA scratch");
    }
    rgbaHandle = importbuffer_fd(rgbaFd, horStride, verStride, RK_FORMAT_RGBA_8888);
    if (rgbaHandle == 0) {
      throw std::runtime_error("RGA importbuffer_fd failed for hardware overlay RGBA scratch handle");
    }

    const int outputFd = mpp_buffer_get_fd(outputBuffer);
    if (outputFd < 0) {
      throw std::runtime_error("mpp_buffer_get_fd failed for hardware overlay NV12 output");
    }
    outputHandle = importbuffer_fd(outputFd, horStride, verStride, RK_FORMAT_YCbCr_420_SP);
    if (outputHandle == 0) {
      throw std::runtime_error("RGA importbuffer_fd failed for hardware overlay NV12 output handle");
    }

    rga_buffer_t src = wrapbuffer_handle(
        srcHandle, frame.width, frame.height, RK_FORMAT_YCbCr_420_SP, horStride, verStride);
    rga_buffer_t overlay = wrapbuffer_handle(
        overlayHandle, frame.width, frame.height, RK_FORMAT_RGBA_8888);
    rga_buffer_t rgba = wrapbuffer_handle(
        rgbaHandle, frame.width, frame.height, RK_FORMAT_RGBA_8888, horStride, verStride);
    rga_buffer_t dst = wrapbuffer_handle(
        outputHandle, frame.width, frame.height, RK_FORMAT_YCbCr_420_SP, horStride, verStride);

    // Clear the host RGBA overlay on CPU first. On this board the RGA_COLORFILL
    // path for this overlay plane is not stable and fails with Invalid argument.
    std::memset(overlayData.data(), 0, overlayBytes);
    IM_STATUS status = IM_STATUS_SUCCESS;
    const int thickness = std::max(1, kModelZooBoxThickness);
    for (const auto& box : result.boxes) {
      const int x1 = std::clamp(static_cast<int>(box.x1), 0, frame.width - 1);
      const int y1 = std::clamp(static_cast<int>(box.y1), 0, frame.height - 1);
      const int x2 = std::clamp(static_cast<int>(box.x2), x1 + 1, frame.width);
      const int y2 = std::clamp(static_cast<int>(box.y2), y1 + 1, frame.height);
      const RgbColor classColor = ultralyticsColorForClass(box.classId);
      drawRectangleOnOverlay(
          overlayData,
          frame.width,
          frame.height,
          x1,
          y1,
          std::max(1, x2 - x1),
          std::max(1, y2 - y1),
          thickness,
          config.style == VisualStyle::kYolo ? classColor.r : 0,
          config.style == VisualStyle::kYolo ? classColor.g : 0,
          config.style == VisualStyle::kYolo ? classColor.b : 255,
          255);

      char text[256] = {};
      if (config.showLabel && !box.label.empty() && config.showConf) {
        std::snprintf(text, sizeof(text), "%s %.1f%%", box.label.c_str(), box.score * 100.0f);
      } else if (config.showLabel && !box.label.empty()) {
        std::snprintf(text, sizeof(text), "%s", box.label.c_str());
      } else if (config.showConf) {
        std::snprintf(text, sizeof(text), "%.1f%%", box.score * 100.0f);
      }
      if (text[0] != '\0') {
        if (config.style == VisualStyle::kYolo) {
          const RgbColor textColor = ultralyticsTextColor(classColor);
          drawYoloLabelBoxOnOverlay(
              overlayData,
              frame.width,
              frame.height,
              text,
              x1,
              y1,
              kModelZooFontPixelSize,
              classColor.r,
              classColor.g,
              classColor.b,
              220,
              textColor.r,
              textColor.g,
              textColor.b);
        } else {
          drawTextOnOverlay(
              overlayData,
              frame.width,
              frame.height,
              text,
              x1,
              y1 - 20,
              kModelZooFontPixelSize,
              255,
              0,
              0);
        }
      }
    }

    // Use the official stable building blocks from librga:
    // 1. imcvtcolor: NV12 -> RGBA
    // 2. imblend: RGBA overlay over RGBA background
    // 3. imcvtcolor: RGBA -> NV12
    // This avoids the direct YUV alpha composite path that produced black
    // output on this board/driver combination.
    int checkStatus = imcheck(src, rgba, {}, {});
    if (checkStatus != IM_STATUS_NOERROR) {
      throw std::runtime_error(
          std::string("RGA imcheck failed for NV12->RGBA conversion: ") +
          imStrError_t(static_cast<IM_STATUS>(checkStatus)));
    }
    status = imcvtcolor(src, rgba, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGBA_8888);
    if (status != IM_STATUS_SUCCESS) {
      throw std::runtime_error(
          std::string("RGA NV12->RGBA conversion failed for hardware overlay: ") + imStrError_t(status));
    }

    checkStatus = imcheck(overlay, rgba, {}, {});
    if (checkStatus != IM_STATUS_NOERROR) {
      throw std::runtime_error(
          std::string("RGA imcheck failed for RGBA overlay blend: ") +
          imStrError_t(static_cast<IM_STATUS>(checkStatus)));
    }
    status = imblend(overlay, rgba, IM_ALPHA_BLEND_SRC_OVER | IM_ALPHA_BLEND_PRE_MUL);
    if (status != IM_STATUS_SUCCESS) {
      throw std::runtime_error(
          std::string("RGA RGBA overlay blend failed: ") + imStrError_t(status));
    }

    checkStatus = imcheck(rgba, dst, {}, {});
    if (checkStatus != IM_STATUS_NOERROR) {
      throw std::runtime_error(
          std::string("RGA imcheck failed for RGBA->NV12 conversion: ") +
          imStrError_t(static_cast<IM_STATUS>(checkStatus)));
    }
    status = imcvtcolor(rgba, dst, RK_FORMAT_RGBA_8888, RK_FORMAT_YCbCr_420_SP);
    if (status != IM_STATUS_SUCCESS) {
      throw std::runtime_error(
          std::string("RGA RGBA->NV12 conversion failed for hardware overlay: ") + imStrError_t(status));
    }

    releasebuffer_handle(srcHandle);
    releasebuffer_handle(overlayHandle);
    releasebuffer_handle(rgbaHandle);
    releasebuffer_handle(outputHandle);
    mpp_buffer_put(rgbaBuffer);

    DecodedFrame annotated = frame;
    annotated.dmaFd = mpp_buffer_get_fd(outputBuffer);
    annotated.nativeHandle = makeMppBufferOwner(outputBuffer);
    return annotated;
  } catch (...) {
    if (srcHandle != 0) releasebuffer_handle(srcHandle);
    if (overlayHandle != 0) releasebuffer_handle(overlayHandle);
    if (rgbaHandle != 0) releasebuffer_handle(rgbaHandle);
    if (outputHandle != 0) releasebuffer_handle(outputHandle);
    if (rgbaBuffer != nullptr) mpp_buffer_put(rgbaBuffer);
    if (outputBuffer != nullptr) mpp_buffer_put(outputBuffer);
    throw;
  }
}
#else
DecodedFrame makeAnnotatedEncodeFrame(const DecodedFrame&, const DetectionResult&, const VisualConfig&) {
  throw std::runtime_error("Hardware box drawing requires Rockchip RGA support");
}
#endif

}  // namespace

void validateAppConfig(const AppConfig& config) {
  if (config.source.uri.empty()) {
    throw std::runtime_error("Input source is required");
  }
  if (config.model.modelPath.empty()) {
    throw std::runtime_error("Model path is required");
  }
  if (config.model.inputWidth <= 0 || config.model.inputHeight <= 0) {
    throw std::runtime_error("Model input size must be positive");
  }
  if (config.maxFrames < 0) {
    throw std::runtime_error("maxFrames must be greater than or equal to 0");
  }
  if (config.inferWorkers <= 0) {
    throw std::runtime_error("inferWorkers must be greater than 0");
  }

  requireCompiledIn(config.decoderBackend, "decoder", isCompiledIn, availableDecoderBackends, toString);
  requireCompiledIn(config.preprocBackend, "preprocessor", isCompiledIn, availablePreprocBackends, toString);
  requireCompiledIn(config.inferBackend, "inference", isCompiledIn, availableInferBackends, toString);
  requireCompiledIn(
      resolvePostprocBackend(config),
      "postprocessor",
      isCompiledIn,
      availablePostprocBackends,
      toString);

  if (!config.visual.outputVideo.empty() && !config.visual.outputRtsp.empty()) {
    throw std::runtime_error("Specify only one annotated output target: use either --output-video or --output-rtsp");
  }

  if (!config.visual.outputRtsp.empty() && !isRtspUrl(config.visual.outputRtsp)) {
    throw std::runtime_error("output-rtsp must start with rtsp://");
  }

  if (hasAnnotatedOutputTarget(config) &&
      !wantsHardwareEncodedAnnotatedOutput(config) &&
      detectAvailableEncoderBackend() == EncoderBackendType::kRockchipMpp) {
    throw std::runtime_error(
        "On the Rockchip path, annotated output uses hardware encoding. "
        "Use .h264/.264/.h265/.hevc raw bitstream, .mp4, or rtsp:// for muxed streaming output.");
  }

  if (!config.encoderOutput.empty()) {
    auto encoder = createEncoderBackend(EncoderBackendType::kAuto);
    if (!encoder->supportsDecodedFrameInput()) {
      throw std::runtime_error(
          "encoder-output currently requires an encoder backend that accepts decoded NV12 frames. "
          "The selected auto encoder is '" +
          encoder->name() +
          "', which only accepts RGB images. Use --output-video for the NVIDIA path or switch to Rockchip MPP.");
    }
  }

  if (config.visual.outputOverlayMode == OutputOverlayMode::kRga &&
      hasAnnotatedOutputTarget(config) &&
      backendUsesNvidiaEncoder()) {
    throw std::runtime_error(
        "output-overlay=rga is only available on the Rockchip RGA path. Use --output-overlay cpu on the NVIDIA platform.");
  }

  if (config.visual.display ||
      (hasAnnotatedOutputTarget(config) && config.visual.outputOverlayMode != OutputOverlayMode::kRga)) {
    const auto visualizer = createVisualizer();
    if (!visualizer->isAvailable()) {
      throw std::runtime_error(
          "Visualization requested, but no visualizer backend is available in this build");
    }
  }
}

void runPipeline(const AppConfig& config) {
  const std::string annotatedOutputPath = annotatedOutputTarget(config);
  const bool needsHardwareEncodedAnnotatedVideo = wantsHardwareEncodedAnnotatedOutput(config);
  const bool useRgaAnnotatedOutput = usesRgaAnnotatedOutput(config);
  const bool needsVisualizerDraw = config.visual.display ||
                                   (!annotatedOutputPath.empty() && !useRgaAnnotatedOutput);
  const bool needsDisplayFrame = needsVisualizerDraw || config.dumpFirstFrame;
  const std::size_t inferenceQueueCapacity = static_cast<std::size_t>(std::max(2, config.inferWorkers * 2));
  const std::size_t rgaMaxInflightFrames = static_cast<std::size_t>(std::max(1, config.inferWorkers * 2 + 4));
  const PostprocBackendType resolvedPostprocBackend = resolvePostprocBackend(config);

  if (config.verbose) {
    std::cerr << "[PIPELINE] postproc requested=" << toString(config.postprocBackend)
              << " resolved=" << toString(resolvedPostprocBackend)
              << " model_output_layout=" << toModelOutputLayoutName(config.modelOutputLayout)
              << " model=" << config.model.modelPath << "\n";
  }

  int inferInputWidth = config.model.inputWidth;
  int inferInputHeight = config.model.inputHeight;
  {
    auto inferProbe = createInferBackend(config.inferBackend);
    inferProbe->open(config.model, makeInferRuntimeConfig(config, 0, config.inferWorkers));
    inferInputWidth = inferProbe->inputWidth() > 0 ? inferProbe->inputWidth() : config.model.inputWidth;
    inferInputHeight = inferProbe->inputHeight() > 0 ? inferProbe->inputHeight() : config.model.inputHeight;
    if (config.verbose) {
      std::cerr << "[PIPELINE] infer_probe backend=" << inferProbe->name()
                << " input=" << inferInputWidth << "x" << inferInputHeight << "\n";
    }
  }

  auto decoder = createDecoderBackend(config.decoderBackend);
  auto preproc = createPreprocBackend(config.preprocBackend);
  preproc->setMaxInflightFrames(rgaMaxInflightFrames);

  FFmpegPacketSource packetSource;
  packetSource.open(config.source);
  const SourceVideoInfo sourceVideoInfo = packetSource.videoInfo();
  decoder->open(packetSource.codec());
  if (config.verbose) {
    std::cerr << "[PIPELINE] stages decoder=" << decoder->name()
              << " preproc=" << preproc->name()
              << " infer=" << toString(config.inferBackend == InferBackendType::kAuto
                     ? detectAvailableInferBackend()
                     : config.inferBackend)
              << " postproc=" << toString(resolvedPostprocBackend)
              << " source=" << sourceVideoInfo.width << "x" << sourceVideoInfo.height
              << " fps=" << sourceVideoInfo.fpsNum << "/" << sourceVideoInfo.fpsDen
              << " infer_workers=" << config.inferWorkers << "\n";
  }

  BoundedQueue<PreparedFrame> preparedQueue(inferenceQueueCapacity);
  BoundedQueue<ProcessedFrame> processedQueue(inferenceQueueCapacity);

  std::exception_ptr workerError;
  std::mutex errorMutex;
  auto storeError = [&](std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(errorMutex);
    if (!workerError) {
      workerError = error;
    }
  };

  std::vector<std::thread> inferWorkers;
  inferWorkers.reserve(static_cast<std::size_t>(config.inferWorkers));
  for (int workerIndex = 0; workerIndex < config.inferWorkers; ++workerIndex) {
    inferWorkers.emplace_back([&, workerIndex] {
      try {
        auto infer = createInferBackend(config.inferBackend);
        infer->open(config.model, makeInferRuntimeConfig(config, workerIndex, config.inferWorkers));
        auto postproc = createPostprocBackend(resolvedPostprocBackend, makePostprocessOptions(config));

        PreparedFrame prepared;
        while (preparedQueue.pop(prepared)) {
          const auto inferStart = Clock::now();
          const InferenceOutput output = infer->infer(prepared.inferenceImage);
          const auto inferEnd = Clock::now();
          const auto postStart = inferEnd;
          const DetectionResult result = postproc->postprocess(
              output,
              prepared.inferenceImage,
              prepared.originalWidth,
              prepared.originalHeight,
              prepared.pts);
          const auto postEnd = Clock::now();

          ProcessedFrame processed;
          processed.index = prepared.index;
          processed.pts = prepared.pts;
          processed.decodedFrame = std::move(prepared.decodedFrame);
          processed.result = result;
          processed.decodeMs = prepared.decodeMs;
          processed.preprocMs = prepared.preprocMs;
          processed.inferMs = Ms(inferEnd - inferStart).count();
          processed.postprocMs = Ms(postEnd - postStart).count();
          processedQueue.push(std::move(processed));
        }
      } catch (...) {
        storeError(std::current_exception());
        preparedQueue.close();
        processedQueue.close();
      }
    });
  }

  std::thread outputThread([&] {
    try {
      std::unique_ptr<IVisualizer> visualizer;
      std::unique_ptr<IPreprocessorBackend> displayPreproc;
      std::unique_ptr<IEncoderBackend> encoder;
      std::unique_ptr<IEncoderBackend> annotatedVideoEncoder;
      bool visualizerInitialized = false;
      bool encoderInitialized = false;
      bool annotatedVideoEncoderInitialized = false;
      const int outputEncoderFps = resolveEncoderFps(config, sourceVideoInfo);
      if (needsVisualizerDraw) {
        visualizer = createVisualizer();
      }
      if (needsDisplayFrame) {
        displayPreproc = createPreprocBackend(config.preprocBackend);
        displayPreproc->setMaxInflightFrames(rgaMaxInflightFrames);
      }
      if (!config.encoderOutput.empty()) {
        encoder = createEncoderBackend(EncoderBackendType::kAuto);
        if (config.verbose) {
          std::cerr << "[PIPELINE] raw_encoder backend=" << encoder->name() << "\n";
        }
      }
      if (needsHardwareEncodedAnnotatedVideo) {
        annotatedVideoEncoder = createEncoderBackend(EncoderBackendType::kAuto);
        if (config.verbose) {
          std::cerr << "[PIPELINE] annotated_encoder backend=" << annotatedVideoEncoder->name() << "\n";
        }
      }

      std::map<std::size_t, ProcessedFrame> pending;
      std::size_t nextIndex = 0;
      std::size_t displayedCount = 0;
      const auto outputStart = Clock::now();
      ProcessedFrame processed;
      while (processedQueue.pop(processed)) {
        pending.emplace(processed.index, std::move(processed));
        while (true) {
          auto it = pending.find(nextIndex);
          if (it == pending.end()) {
            break;
          }

          ProcessedFrame current = std::move(it->second);
          pending.erase(it);
          ++displayedCount;

          if (encoder && current.decodedFrame.dmaFd < 0) {
            throw std::runtime_error(
                "encoder-output requested, but decoded frame does not provide a valid dma fd");
          }
          if (encoder && !encoderInitialized) {
            EncoderConfig encCfg;
            encCfg.outputPath = config.encoderOutput;
            encCfg.codec = config.encoderCodec;
            encCfg.fps = outputEncoderFps;
            encCfg.fpsNum = resolveEncoderFpsNum(config, sourceVideoInfo);
            encCfg.fpsDen = resolveEncoderFpsDen(config, sourceVideoInfo);
            encCfg.width = current.decodedFrame.width;
            encCfg.height = current.decodedFrame.height;
            encCfg.horStride = current.decodedFrame.horizontalStride > 0
                ? current.decodedFrame.horizontalStride
                : current.decodedFrame.width;
            encCfg.verStride = current.decodedFrame.verticalStride > 0
                ? current.decodedFrame.verticalStride
                : current.decodedFrame.height;
            encCfg.bitrate = resolveEncoderBitrate(config, encCfg.width, encCfg.height, outputEncoderFps);
            encCfg.inputFormat = PixelFormat::kNv12;
            if (config.verbose) {
              std::cerr << "[PIPELINE] init_raw_encoder backend=" << encoder->name()
                        << " codec=" << encCfg.codec
                        << " path=" << encCfg.outputPath
                        << " size=" << encCfg.width << "x" << encCfg.height
                        << " stride=" << encCfg.horStride << "x" << encCfg.verStride
                        << " fps=" << encCfg.fpsNum << "/" << encCfg.fpsDen
                        << " bitrate=" << encCfg.bitrate << "\n";
            }
            encoder->init(encCfg);
            encoderInitialized = true;
          }
          const bool keepEncodedFrame =
              shouldKeepEncodedFrame(displayedCount - 1, sourceVideoInfo, outputEncoderFps);
          if (encoder && encoderInitialized && keepEncodedFrame) {
            encoder->encodeDecodedFrame(current.decodedFrame, current.pts);
          }

          double displayPreprocMs = 0.0;
          std::optional<RgbImage> displayImage;
          std::optional<DetectionResult> displayResult;
          if (needsDisplayFrame) {
            const auto [displayWidth, displayHeight] =
                needsVisualizerDraw
                    ? computeDisplaySize(config, current.decodedFrame.width, current.decodedFrame.height)
                    : std::pair<int, int>{current.decodedFrame.width, current.decodedFrame.height};
            const auto displayPreprocStart = Clock::now();
            displayImage = displayPreproc->convertAndResize(
                current.decodedFrame,
                displayWidth,
                displayHeight,
                PreprocessOptions{false, 114, true});
            displayResult = scaleDetectionResult(current.result, displayWidth, displayHeight);
            displayPreprocMs = Ms(Clock::now() - displayPreprocStart).count();
          }

          const bool shouldLogProgress =
              displayedCount == 1 ||
              (config.progressEvery > 0 && (displayedCount % static_cast<std::size_t>(config.progressEvery) == 0));
          if (shouldLogProgress) {
            const double elapsedSeconds =
                std::max(1e-6, std::chrono::duration<double>(Clock::now() - outputStart).count());
            const double fps = static_cast<double>(displayedCount) / elapsedSeconds;
            std::cout << "frame=" << displayedCount
                      << " pts=" << current.pts
                      << " detections=" << current.result.boxes.size()
                      << " fps=" << fps;
            if (config.verbose) {
              std::cout << " decode_ms=" << current.decodeMs
                        << " preproc_ms=" << current.preprocMs
                        << " infer_ms=" << current.inferMs
                        << " post_ms=" << current.postprocMs;
              if (needsDisplayFrame) {
                std::cout << " display_preproc_ms=" << displayPreprocMs;
              }
            }
            std::cout << "\n";
          }

          if (displayImage.has_value()) {
            maybeDumpFirstFrame(config, displayImage.value(), displayedCount);
          }

          if (needsVisualizerDraw && displayImage.has_value() && displayResult.has_value()) {
            if (!visualizerInitialized) {
              VisualConfig visualConfig = config.visual;
              if (needsHardwareEncodedAnnotatedVideo) {
                visualConfig.outputVideo.clear();
                visualConfig.outputRtsp.clear();
              }
              if (config.verbose) {
                std::cerr << "[PIPELINE] init_visualizer size="
                          << displayImage->width << "x" << displayImage->height << "\n";
              }
              visualizer->init(displayImage->width, displayImage->height, visualConfig);
              visualizerInitialized = true;
            }
            const RgbImage drawnImage = visualizer->draw(displayImage.value(), displayResult.value());
            if (annotatedVideoEncoder) {
              if (!annotatedVideoEncoderInitialized) {
                EncoderConfig encCfg;
                encCfg.outputPath = annotatedOutputPath;
                encCfg.codec = config.encoderCodec;
                encCfg.fps = outputEncoderFps;
                encCfg.fpsNum = resolveEncoderFpsNum(config, sourceVideoInfo);
                encCfg.fpsDen = resolveEncoderFpsDen(config, sourceVideoInfo);
                encCfg.width = useRgaAnnotatedOutput ? current.decodedFrame.width : drawnImage.width;
                encCfg.height = useRgaAnnotatedOutput ? current.decodedFrame.height : drawnImage.height;
                encCfg.bitrate = resolveEncoderBitrate(config, encCfg.width, encCfg.height, outputEncoderFps);
                encCfg.inputFormat = useRgaAnnotatedOutput ? PixelFormat::kNv12 : PixelFormat::kRgb888;
                if (config.verbose) {
                  std::cerr << "[PIPELINE] init_annotated_encoder backend=" << annotatedVideoEncoder->name()
                            << " codec=" << encCfg.codec
                            << " path=" << encCfg.outputPath
                            << " size=" << encCfg.width << "x" << encCfg.height
                            << " fps=" << encCfg.fpsNum << "/" << encCfg.fpsDen
                            << " bitrate=" << encCfg.bitrate
                            << " input_format=" << (encCfg.inputFormat == PixelFormat::kNv12 ? "NV12" : "RGB888")
                            << "\n";
                }
                annotatedVideoEncoder->init(encCfg);
                annotatedVideoEncoderInitialized = true;
              }
              if (keepEncodedFrame) {
                if (useRgaAnnotatedOutput) {
                  DecodedFrame annotatedFrame =
                      makeAnnotatedEncodeFrame(current.decodedFrame, current.result, config.visual);
                  annotatedVideoEncoder->encodeDecodedFrame(annotatedFrame, current.pts);
                } else {
                  annotatedVideoEncoder->encode(drawnImage, current.pts);
                }
              }
            }
            (void)drawnImage;
            visualizer->show();
          } else if (annotatedVideoEncoder && useRgaAnnotatedOutput) {
            if (!annotatedVideoEncoderInitialized) {
              EncoderConfig encCfg;
              encCfg.outputPath = annotatedOutputPath;
              encCfg.codec = config.encoderCodec;
              encCfg.fps = outputEncoderFps;
              encCfg.fpsNum = resolveEncoderFpsNum(config, sourceVideoInfo);
              encCfg.fpsDen = resolveEncoderFpsDen(config, sourceVideoInfo);
              encCfg.width = current.decodedFrame.width;
              encCfg.height = current.decodedFrame.height;
              encCfg.horStride = current.decodedFrame.horizontalStride > 0
                  ? current.decodedFrame.horizontalStride
                  : current.decodedFrame.width;
              encCfg.verStride = current.decodedFrame.verticalStride > 0
                  ? current.decodedFrame.verticalStride
                  : current.decodedFrame.height;
              encCfg.bitrate = resolveEncoderBitrate(config, encCfg.width, encCfg.height, outputEncoderFps);
              encCfg.inputFormat = PixelFormat::kNv12;
              if (config.verbose) {
                std::cerr << "[PIPELINE] init_annotated_encoder backend=" << annotatedVideoEncoder->name()
                          << " codec=" << encCfg.codec
                          << " path=" << encCfg.outputPath
                          << " size=" << encCfg.width << "x" << encCfg.height
                          << " stride=" << encCfg.horStride << "x" << encCfg.verStride
                          << " fps=" << encCfg.fpsNum << "/" << encCfg.fpsDen
                          << " bitrate=" << encCfg.bitrate
                          << " input_format=NV12\n";
              }
              annotatedVideoEncoder->init(encCfg);
              annotatedVideoEncoderInitialized = true;
            }
            if (keepEncodedFrame) {
              DecodedFrame annotatedFrame =
                  makeAnnotatedEncodeFrame(current.decodedFrame, current.result, config.visual);
              annotatedVideoEncoder->encodeDecodedFrame(annotatedFrame, current.pts);
            }
          }

          ++nextIndex;
        }
      }

      if (encoder) {
        if (!encoderInitialized) {
          throw std::runtime_error(
              "encoder-output requested, but no decodable frame with a valid dma fd reached the output stage");
        }
        encoder->flush();
      }
      if (annotatedVideoEncoder) {
        if (!annotatedVideoEncoderInitialized) {
          throw std::runtime_error("output-video requested, but no frame reached the annotated video encoder");
        }
        annotatedVideoEncoder->flush();
      }
      if (visualizer) {
        visualizer->close();
      }
    } catch (...) {
      storeError(std::current_exception());
      preparedQueue.close();
      processedQueue.close();
    }
  });

  try {
    bool eosSubmitted = false;
    std::size_t producedFrames = 0;
    bool loggedFirstDecodedFrame = false;
    while (!eosSubmitted && (config.maxFrames == 0 || producedFrames < static_cast<std::size_t>(config.maxFrames))) {
      const EncodedPacket packet = packetSource.readPacket();
      decoder->submitPacket(packet);
      eosSubmitted = packet.endOfStream;

      while (true) {
        const auto decodeStart = Clock::now();
        std::optional<DecodedFrame> decodedFrame = decoder->receiveFrame();
        const auto decodeEnd = Clock::now();
        if (!decodedFrame.has_value()) {
          break;
        }

        if (config.verbose && !loggedFirstDecodedFrame) {
          loggedFirstDecodedFrame = true;
          std::cerr << "[PIPELINE] first_decoded_frame"
                    << " size=" << decodedFrame->width << "x" << decodedFrame->height
                    << " stride=" << decodedFrame->horizontalStride << "x" << decodedFrame->verticalStride
                    << " chroma_stride=" << decodedFrame->chromaStride
                    << " format=" << (decodedFrame->format == PixelFormat::kNv12 ? "NV12" : "unknown")
                    << " native_format=" << decodedFrame->nativeFormat
                    << " on_device=" << (decodedFrame->isOnDevice ? "true" : "false")
                    << " dma_fd=" << decodedFrame->dmaFd
                    << "\n";
        }

        PreparedFrame prepared;
        prepared.index = producedFrames;
        prepared.pts = decodedFrame->pts;
        prepared.originalWidth = decodedFrame->width;
        prepared.originalHeight = decodedFrame->height;
        prepared.decodeMs = Ms(decodeEnd - decodeStart).count();

        const auto preprocStart = Clock::now();
        prepared.inferenceImage = preproc->convertAndResize(
            decodedFrame.value(),
            inferInputWidth,
            inferInputHeight,
            PreprocessOptions{config.letterbox, 114, !config.rknnZeroCopy});
        prepared.decodedFrame = std::move(decodedFrame.value());
        const auto preprocEnd = Clock::now();
        prepared.preprocMs = Ms(preprocEnd - preprocStart).count();

        preparedQueue.push(std::move(prepared));
        ++producedFrames;
        if (config.maxFrames > 0 && producedFrames >= static_cast<std::size_t>(config.maxFrames)) {
          break;
        }
      }
    }
  } catch (...) {
    storeError(std::current_exception());
  }

  preparedQueue.close();
  for (auto& worker : inferWorkers) {
    worker.join();
  }
  processedQueue.close();
  outputThread.join();

  if (workerError) {
    std::rethrow_exception(workerError);
  }

  // Some board-side Rockchip library combinations still crash during process
  // teardown after a fully successful run. Once all work is completed and no
  // error is pending, exit immediately to avoid destructing backend objects.
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
