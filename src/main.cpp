#include "decoder_interface.hpp"
#include "preproc_interface.hpp"
#include "infer_interface.hpp"
#include "ffmpeg_packet_source.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

namespace {

struct Config {
  InputSourceConfig source;
  ModelConfig model;
  DecoderBackendType decoderBackend = DecoderBackendType::kAuto;
  PreprocBackendType preprocBackend = PreprocBackendType::kAuto;
  InferBackendType inferBackend = InferBackendType::kAuto;
  int gpuId = 0;
  int maxFrames = 30;  // 默认处理 30 帧后停止
};

Config parseConfig(int argc, char* argv[]) {
  Config config;

  // 简单参数解析
  int argIndex = 1;

  // 解析 --backend 参数
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--backend" && i + 1 < argc) {
      std::string backend = argv[++i];
      if (backend == "rockchip" || backend == "mpp") {
        config.decoderBackend = DecoderBackendType::kRockchipMpp;
        config.preprocBackend = PreprocBackendType::kRockchipRga;
        config.inferBackend = InferBackendType::kRockchipRknn;
      } else if (backend == "nvidia" || backend == "nvdec") {
        config.decoderBackend = DecoderBackendType::kNvidiaNvdec;
        config.preprocBackend = PreprocBackendType::kNvidiaCuda;
        config.inferBackend = InferBackendType::kNvidiaTrt;
      }
    } else if (arg == "--gpu" && i + 1 < argc) {
      config.gpuId = std::atoi(argv[++i]);
    } else if (arg == "--max-frames" && i + 1 < argc) {
      config.maxFrames = std::atoi(argv[++i]);
    } else if (arg[0] != '-') {
      // 位置参数
      if (config.source.uri.empty()) {
        config.source.uri = arg;
      } else if (config.model.modelPath.empty()) {
        config.model.modelPath = arg;
      }
    }
  }

  // 解析模型输入尺寸
  if (argc >= 4) {
    // 检查是否有非选项参数
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-' && argv[i-1][0] != '-') {
        if (config.model.modelPath.empty()) {
          config.model.modelPath = argv[i];
        } else {
          config.model.inputWidth = std::atoi(argv[i]);
          if (i + 1 < argc) {
            config.model.inputHeight = std::atoi(argv[++i]);
          }
        }
      }
    }
  }

  return config;
}

void printUsage(const char* program) {
  std::cerr << "Usage: " << program << " [options] <video_or_rtsp> <model_file> [width] [height]\n\n";
  std::cerr << "Options:\n";
  std::cerr << "  --backend <rockchip|nvidia>  选择后端平台\n";
  std::cerr << "  --gpu <id>                   GPU 设备 ID (默认：0)\n";
  std::cerr << "  --max-frames <n>             最大处理帧数 (默认：30)\n";
  std::cerr << "\nExamples:\n";
  std::cerr << "  # Rockchip 平台\n";
  std::cerr << "  " << program << " --backend rockchip test.mp4 yolov5s.rknn 640 640\n";
  std::cerr << "  # NVIDIA 平台\n";
  std::cerr << "  " << program << " --backend nvidia test.mp4 yolov5s.engine 640 640\n";
}

void runPipeline(const Config& config) {
  std::cout << "=== Video Inference Pipeline ===\n";
  std::cout << "Source: " << config.source.uri << "\n";
  std::cout << "Model: " << config.model.modelPath << "\n";
  std::cout << "Input: " << config.model.inputWidth << "x" << config.model.inputHeight << "\n";
  std::cout << "Max Frames: " << config.maxFrames << "\n";
  std::cout << "===============================\n\n";

  // 1. 创建解码器
  auto decoder = createDecoderBackend(config.decoderBackend);
  std::cout << "[1/4] Decoder: " << decoder->name() << "\n";

  // 2. 创建预处理器
  auto preproc = createPreprocBackend(config.preprocBackend);
  std::cout << "[2/4] Preprocessor: " << preproc->name() << "\n";

  // 3. 创建推理引擎
  auto infer = createInferBackend(config.inferBackend);
  std::cout << "[3/4] Inference: " << infer->name() << "\n";
  infer->open(config.model);
  std::cout << "[4/4] Model loaded, input: " << infer->inputWidth() << "x" << infer->inputHeight() << "\n";

  // 4. 打开视频源
  FFmpegPacketSource packetSource;
  packetSource.open(config.source);
  std::cout << "\n[Pipeline] Starting...\n";

  // 5. 初始化解码器
  decoder->open(packetSource.codec());

  // 6. 主循环
  std::size_t frameCount = 0;
  while (true) {
    const EncodedPacket packet = packetSource.readPacket();
    const std::optional<DecodedFrame> decodedFrame = decoder->decode(packet);

    // EOF 处理
    if (packet.endOfStream && !decodedFrame.has_value()) {
      break;
    }

    if (!decodedFrame.has_value()) {
      continue;
    }

    // 预处理
    const RgbImage image = preproc->convertAndResize(
        decodedFrame.value(),
        infer->inputWidth(),
        infer->inputHeight());

    // 推理
    const std::vector<float> output = infer->infer(image);
    ++frameCount;

    std::cout << "frame=" << frameCount
              << " pts=" << decodedFrame->pts
              << " output_size=" << output.size() << "\n";

    if (config.maxFrames > 0 && frameCount >= static_cast<std::size_t>(config.maxFrames)) {
      std::cout << "Reached max frames (" << config.maxFrames << "), stopping.\n";
      break;
    }
  }

  std::cout << "\n=== Pipeline Complete ===\n";
  std::cout << "Total frames processed: " << frameCount << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }

  // 处理帮助参数
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    }
  }

  try {
    const Config config = parseConfig(argc, argv);

    // 验证必要参数
    if (config.source.uri.empty()) {
      std::cerr << "Error: Missing video/RTSP source\n\n";
      printUsage(argv[0]);
      return 1;
    }
    if (config.model.modelPath.empty()) {
      std::cerr << "Error: Missing model file\n\n";
      printUsage(argv[0]);
      return 1;
    }

    // 设置环境变量
    if (config.gpuId >= 0) {
      std::setenv("CUDA_DEVICE", std::to_string(config.gpuId).c_str(), 1);
    }

    runPipeline(config);
    return 0;

  } catch (const std::exception& error) {
    std::cerr << "\n[ERROR] Pipeline failed: " << error.what() << '\n';
    return 1;
  }
}
