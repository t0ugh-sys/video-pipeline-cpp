#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class VideoCodec {
  kUnknown,
  kH264,
  kH265,
};

enum class PixelFormat {
  kUnknown,
  kNv12,
  kRgb888,
};

enum class TensorDataType {
  kUnknown,
  kFloat32,
  kUint8,
  kInt8,
  kInt32,
};

struct EncodedPacket {
  std::vector<std::uint8_t> data;
  std::int64_t pts = 0;
  bool keyFrame = false;
  bool endOfStream = false;
};

struct LetterboxInfo {
  bool enabled = false;
  float scale = 1.0f;
  int resizedWidth = 0;
  int resizedHeight = 0;
  int padLeft = 0;
  int padTop = 0;
  int padRight = 0;
  int padBottom = 0;
};

struct DecodedFrame {
  int width = 0;
  int height = 0;
  int horizontalStride = 0;
  int verticalStride = 0;
  int chromaStride = 0;
  PixelFormat format = PixelFormat::kUnknown;
  int dmaFd = -1;
  std::int64_t pts = 0;
  bool isOnDevice = false;
  std::uintptr_t deviceY = 0;
  std::uintptr_t deviceUv = 0;
  // Keeps the backend-native resource alive for the lifetime of this frame.
  // For MPP this owns the underlying MppFrame/MppBuffer.
  std::shared_ptr<void> nativeHandle;
  std::vector<std::uint8_t> yData;
  std::vector<std::uint8_t> uvData;
};

struct RgbImage {
  int width = 0;
  int height = 0;
  PixelFormat format = PixelFormat::kRgb888;
  LetterboxInfo letterbox;
  std::vector<std::uint8_t> data;
};

struct InferenceTensor {
  std::string name;
  std::string layout;
  std::vector<std::int64_t> shape;
  TensorDataType dataType = TensorDataType::kUnknown;
  std::vector<float> data;
};

using InferenceOutput = std::vector<InferenceTensor>;

struct InputSourceConfig {
  std::string uri;
};

struct ModelConfig {
  std::string modelPath;
  int inputWidth = 640;
  int inputHeight = 640;
};
