#include "pipeline_runner.hpp"

#include "backend_registry.hpp"
#include "decoder_interface.hpp"
#include "encoder_interface.hpp"
#include "ffmpeg_packet_source.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"
#include "visualizer.hpp"

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
extern "C" {
#include <mpp_buffer.h>
}
#include <im2d.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
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

bool wantsHardwareEncodedAnnotatedOutput(const AppConfig& config) {
  if (config.visual.outputVideo.empty()) {
    return false;
  }

  return hasSuffixIgnoreCase(config.visual.outputVideo, ".h264") ||
         hasSuffixIgnoreCase(config.visual.outputVideo, ".264") ||
         hasSuffixIgnoreCase(config.visual.outputVideo, ".h265") ||
         hasSuffixIgnoreCase(config.visual.outputVideo, ".hevc");
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

  static MppBufferGroup overlayGroup = nullptr;
  static MppBufferGroup outputGroup = nullptr;
  if (overlayGroup == nullptr) {
    if (mpp_buffer_group_get_internal(
            &overlayGroup,
            static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)) != MPP_OK) {
      throw std::runtime_error("mpp_buffer_group_get_internal failed for RGBA overlay");
    }
  }
  if (outputGroup == nullptr) {
    if (mpp_buffer_group_get_internal(
            &outputGroup,
            static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)) != MPP_OK) {
      throw std::runtime_error("mpp_buffer_group_get_internal failed for hardware overlay output");
    }
  }

  const int horStride = frame.horizontalStride > 0 ? frame.horizontalStride : frame.width;
  const int verStride = frame.verticalStride > 0 ? frame.verticalStride : frame.height;
  const std::size_t outputBytes =
      static_cast<std::size_t>(horStride) * static_cast<std::size_t>(verStride) * 3 / 2;
  const std::size_t overlayBytes =
      static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height) * 4;

  MppBuffer overlayBuffer = nullptr;
  MppBuffer outputBuffer = nullptr;
  if (mpp_buffer_get(overlayGroup, &overlayBuffer, overlayBytes) != MPP_OK || overlayBuffer == nullptr) {
    throw std::runtime_error("mpp_buffer_get failed for hardware overlay RGBA buffer");
  }
  if (mpp_buffer_get(outputGroup, &outputBuffer, outputBytes) != MPP_OK || outputBuffer == nullptr) {
    mpp_buffer_put(overlayBuffer);
    throw std::runtime_error("mpp_buffer_get failed for hardware overlay NV12 output buffer");
  }

  rga_buffer_handle_t srcHandle = 0;
  rga_buffer_handle_t overlayHandle = 0;
  rga_buffer_handle_t outputHandle = 0;
  try {
    srcHandle = importbuffer_fd(frame.dmaFd, horStride, verStride, RK_FORMAT_YCbCr_420_SP);
    if (srcHandle == 0) {
      throw std::runtime_error("RGA importbuffer_fd failed for hardware overlay source");
    }

    const int overlayFd = mpp_buffer_get_fd(overlayBuffer);
    if (overlayFd < 0) {
      throw std::runtime_error("mpp_buffer_get_fd failed for hardware overlay RGBA buffer");
    }
    overlayHandle = importbuffer_fd(overlayFd, static_cast<int>(overlayBytes));
    if (overlayHandle == 0) {
      throw std::runtime_error("RGA importbuffer_fd failed for hardware overlay RGBA handle");
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
    rga_buffer_t dst = wrapbuffer_handle(
        outputHandle, frame.width, frame.height, RK_FORMAT_YCbCr_420_SP, horStride, verStride);

    im_rect fullRect{};
    fullRect.x = 0;
    fullRect.y = 0;
    fullRect.width = frame.width;
    fullRect.height = frame.height;

    IM_STATUS status = imfill(overlay, fullRect, 0x00000000);
    if (status != IM_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("RGA imfill failed for hardware overlay: ") + imStrError_t(status));
    }

    const int thickness = std::max(1, static_cast<int>(config.bboxThickness));
    for (const auto& box : result.boxes) {
      const int x1 = std::clamp(static_cast<int>(box.x1), 0, frame.width - 1);
      const int y1 = std::clamp(static_cast<int>(box.y1), 0, frame.height - 1);
      const int x2 = std::clamp(static_cast<int>(box.x2), x1 + 1, frame.width);
      const int y2 = std::clamp(static_cast<int>(box.y2), y1 + 1, frame.height);
      im_rect rect{};
      rect.x = x1;
      rect.y = y1;
      rect.width = std::max(1, x2 - x1);
      rect.height = std::max(1, y2 - y1);

      status = imrectangle(overlay, rect, 0xff00ff00, thickness);
      if (status != IM_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("RGA imrectangle failed while drawing hardware boxes: ") + imStrError_t(status));
      }
    }

    im_rect srcRect{};
    srcRect.x = 0;
    srcRect.y = 0;
    srcRect.width = frame.width;
    srcRect.height = frame.height;
    im_rect bgRect = srcRect;
    im_rect dstRect = srcRect;
    status = improcess(
        src,
        dst,
        overlay,
        srcRect,
        dstRect,
        bgRect,
        -1,
        nullptr,
        nullptr,
        IM_SYNC | IM_ALPHA_BLEND_DST_OVER | IM_ALPHA_BLEND_PRE_MUL);
    if (status != IM_STATUS_SUCCESS) {
      throw std::runtime_error(
          std::string("RGA improcess failed for hardware overlay composite: ") + imStrError_t(status));
    }

    releasebuffer_handle(srcHandle);
    releasebuffer_handle(overlayHandle);
    releasebuffer_handle(outputHandle);

    DecodedFrame annotated = frame;
    annotated.dmaFd = mpp_buffer_get_fd(outputBuffer);
    annotated.nativeHandle = makeMppBufferOwner(outputBuffer);
    return annotated;
  } catch (...) {
    if (srcHandle != 0) releasebuffer_handle(srcHandle);
    if (overlayHandle != 0) releasebuffer_handle(overlayHandle);
    if (outputHandle != 0) releasebuffer_handle(outputHandle);
    if (overlayBuffer != nullptr) mpp_buffer_put(overlayBuffer);
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
  requireCompiledIn(config.postprocBackend, "postprocessor", isCompiledIn, availablePostprocBackends, toString);

  if (!config.visual.outputRtsp.empty()) {
    throw std::runtime_error(
        "output-rtsp is disabled on the hardware-first path because it still depends on an unimplemented streaming sink.");
  }

  if (!config.visual.outputVideo.empty() &&
      !wantsHardwareEncodedAnnotatedOutput(config) &&
      detectAvailableEncoderBackend() == EncoderBackendType::kRockchipMpp) {
    throw std::runtime_error(
        "On the Rockchip path, --output-video now uses hardware encoding and writes a raw bitstream. "
        "Use a .h264/.264/.h265/.hevc output path.");
  }

  if (config.visual.display || !config.visual.outputVideo.empty()) {
    const auto visualizer = createVisualizer();
    if (!visualizer->isAvailable()) {
      throw std::runtime_error(
          "Visualization requested, but no visualizer backend is available in this build");
    }
  }
}

void runPipeline(const AppConfig& config) {
  const bool needsVisualization = config.visual.display || !config.visual.outputVideo.empty();
  const bool needsDisplayFrame = needsVisualization || config.dumpFirstFrame;
  const bool needsHardwareEncodedAnnotatedVideo = wantsHardwareEncodedAnnotatedOutput(config);
  const std::size_t inferenceQueueCapacity = static_cast<std::size_t>(std::max(2, config.inferWorkers * 2));
  const std::size_t rgaMaxInflightFrames = static_cast<std::size_t>(std::max(1, config.inferWorkers * 2 + 4));

  int inferInputWidth = config.model.inputWidth;
  int inferInputHeight = config.model.inputHeight;
  {
    auto inferProbe = createInferBackend(config.inferBackend);
    inferProbe->open(config.model, makeInferRuntimeConfig(config, 0, config.inferWorkers));
    inferInputWidth = inferProbe->inputWidth() > 0 ? inferProbe->inputWidth() : config.model.inputWidth;
    inferInputHeight = inferProbe->inputHeight() > 0 ? inferProbe->inputHeight() : config.model.inputHeight;
  }

  auto decoder = createDecoderBackend(config.decoderBackend);
  auto preproc = createPreprocBackend(config.preprocBackend);
  preproc->setMaxInflightFrames(rgaMaxInflightFrames);

  FFmpegPacketSource packetSource;
  packetSource.open(config.source);
  decoder->open(packetSource.codec());

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
        auto postproc = createPostprocBackend(config.postprocBackend, makePostprocessOptions(config));

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
      if (needsVisualization) {
        visualizer = createVisualizer();
      }
      if (needsDisplayFrame) {
        displayPreproc = createPreprocBackend(config.preprocBackend);
        displayPreproc->setMaxInflightFrames(rgaMaxInflightFrames);
      }
      if (!config.encoderOutput.empty()) {
        encoder = createEncoderBackend(EncoderBackendType::kAuto);
      }
      if (needsHardwareEncodedAnnotatedVideo) {
        annotatedVideoEncoder = createEncoderBackend(EncoderBackendType::kAuto);
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
            encCfg.bitrate = config.encoderBitrate;
            encCfg.fps = config.encoderFps;
            encCfg.width = current.decodedFrame.width;
            encCfg.height = current.decodedFrame.height;
            encCfg.horStride = current.decodedFrame.horizontalStride > 0
                ? current.decodedFrame.horizontalStride
                : current.decodedFrame.width;
            encCfg.verStride = current.decodedFrame.verticalStride > 0
                ? current.decodedFrame.verticalStride
                : current.decodedFrame.height;
            encCfg.inputFormat = PixelFormat::kNv12;
            encoder->init(encCfg);
            encoderInitialized = true;
          }
          if (encoder && encoderInitialized) {
            encoder->encodeDecodedFrame(current.decodedFrame, current.pts);
          }

          double displayPreprocMs = 0.0;
          std::optional<RgbImage> displayImage;
          std::optional<DetectionResult> displayResult;
          if (needsDisplayFrame) {
            const auto [displayWidth, displayHeight] =
                needsVisualization
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

          if (needsVisualization && displayImage.has_value() && displayResult.has_value()) {
            if (!visualizerInitialized) {
              VisualConfig visualConfig = config.visual;
              if (needsHardwareEncodedAnnotatedVideo) {
                visualConfig.outputVideo.clear();
              }
              visualizer->init(displayImage->width, displayImage->height, visualConfig);
              visualizerInitialized = true;
            }
            const RgbImage drawnImage = visualizer->draw(displayImage.value(), displayResult.value());
            if (annotatedVideoEncoder) {
              if (!annotatedVideoEncoderInitialized) {
                EncoderConfig encCfg;
                encCfg.outputPath = config.visual.outputVideo;
                encCfg.codec = config.encoderCodec;
                encCfg.bitrate = config.encoderBitrate;
                encCfg.fps = config.encoderFps;
                encCfg.width = drawnImage.width;
                encCfg.height = drawnImage.height;
                encCfg.inputFormat = PixelFormat::kRgb888;
                annotatedVideoEncoder->init(encCfg);
                annotatedVideoEncoderInitialized = true;
              }
              annotatedVideoEncoder->encode(drawnImage, current.pts);
            }
            (void)drawnImage;
            visualizer->show();
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
