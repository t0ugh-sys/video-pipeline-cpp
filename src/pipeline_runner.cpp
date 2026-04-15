#include "pipeline_runner.hpp"

#include "backend_registry.hpp"
#include "decoder_interface.hpp"
#include "ffmpeg_packet_source.hpp"
#include "infer_interface.hpp"
#include "postproc_interface.hpp"
#include "preproc_interface.hpp"
#include "visualizer.hpp"

#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

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

  requireCompiledIn(config.decoderBackend, "decoder", isCompiledIn, availableDecoderBackends, toString);
  requireCompiledIn(config.preprocBackend, "preprocessor", isCompiledIn, availablePreprocBackends, toString);
  requireCompiledIn(config.inferBackend, "inference", isCompiledIn, availableInferBackends, toString);
  requireCompiledIn(config.postprocBackend, "postprocessor", isCompiledIn, availablePostprocBackends, toString);

  const bool needsVisualization =
      config.visual.display ||
      !config.visual.outputVideo.empty() ||
      !config.visual.outputRtsp.empty();
  if (needsVisualization) {
    const auto visualizer = createVisualizer();
    if (!visualizer->isAvailable()) {
      throw std::runtime_error(
          "Visualization output requested, but no visualizer backend is available in this build");
    }
  }
}

void runPipeline(const AppConfig& config) {
  auto decoder = createDecoderBackend(config.decoderBackend);
  auto preproc = createPreprocBackend(config.preprocBackend);
  auto infer = createInferBackend(config.inferBackend);
  auto postproc = createPostprocBackend(
      config.postprocBackend,
      PostprocessOptions{
          config.confThreshold,
          config.nmsThreshold,
          config.labelsPath,
          {},
          config.modelOutputLayout,
          config.verbose});
  auto visualizer = createVisualizer();

  infer->open(config.model);

  FFmpegPacketSource packetSource;
  packetSource.open(config.source);
  decoder->open(packetSource.codec());

  bool visualizerInitialized = false;
  std::size_t frameCount = 0;
  bool eosSubmitted = false;

  while (true) {
    if (!eosSubmitted) {
      const EncodedPacket packet = packetSource.readPacket();
      decoder->submitPacket(packet);
      eosSubmitted = packet.endOfStream;
    }

    bool producedFrame = false;
    while (true) {
      const std::optional<DecodedFrame> decodedFrame = decoder->receiveFrame();
      if (!decodedFrame.has_value()) {
        break;
      }
      producedFrame = true;

      const RgbImage inferenceImage = preproc->convertAndResize(
          decodedFrame.value(),
          infer->inputWidth(),
          infer->inputHeight(),
          PreprocessOptions{config.letterbox, 114});
      const InferenceOutput output = infer->infer(inferenceImage);
      const DetectionResult result = postproc->postprocess(
          output,
          inferenceImage,
          decodedFrame->width,
          decodedFrame->height,
          decodedFrame->pts);

      ++frameCount;
      std::cout << "frame=" << frameCount
                << " pts=" << decodedFrame->pts
                << " detections=" << result.boxes.size() << "\n";

      maybeDumpFirstFrame(config, inferenceImage, frameCount);

      if (config.visual.display || !config.visual.outputVideo.empty() || !config.visual.outputRtsp.empty()) {
        if (!visualizerInitialized && decodedFrame->width > 0 && decodedFrame->height > 0) {
          visualizer->init(decodedFrame->width, decodedFrame->height, config.visual);
          visualizerInitialized = true;
        }

        if (visualizerInitialized) {
          const RgbImage originalImage = preproc->convertAndResize(
              decodedFrame.value(),
              decodedFrame->width,
              decodedFrame->height,
              PreprocessOptions{});
          const RgbImage drawnImage = visualizer->draw(originalImage, result);
          (void)drawnImage;
          visualizer->show();
        }
      }

      if (config.maxFrames > 0 && frameCount >= static_cast<std::size_t>(config.maxFrames)) {
        visualizer->close();
        return;
      }
    }

    if (eosSubmitted && !producedFrame) {
      break;
    }
  }

  visualizer->close();
}

