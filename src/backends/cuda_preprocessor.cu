#include "backends/cuda_preprocessor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

void checkCudaStatus(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

inline std::uint8_t clampToByte(float value) {
  if (value < 0.0f) {
    return 0;
  }
  if (value > 255.0f) {
    return 255;
  }
  return static_cast<std::uint8_t>(value);
}

__device__ inline std::uint8_t clampDevice(float value) {
  if (value < 0.0f) {
    return 0;
  }
  if (value > 255.0f) {
    return 255;
  }
  return static_cast<std::uint8_t>(value);
}

struct ResizePlan {
  LetterboxInfo letterbox;
  int resizedWidth = 0;
  int resizedHeight = 0;
};

ResizePlan makeResizePlan(
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    bool letterbox) {
  ResizePlan plan;
  if (!letterbox) {
    plan.resizedWidth = dstWidth;
    plan.resizedHeight = dstHeight;
    return plan;
  }

  const float scale = std::min(
      static_cast<float>(dstWidth) / static_cast<float>(srcWidth),
      static_cast<float>(dstHeight) / static_cast<float>(srcHeight));
  plan.letterbox.enabled = true;
  plan.letterbox.scale = scale;
  plan.resizedWidth = std::max(1, static_cast<int>(srcWidth * scale));
  plan.resizedHeight = std::max(1, static_cast<int>(srcHeight * scale));
  plan.letterbox.resizedWidth = plan.resizedWidth;
  plan.letterbox.resizedHeight = plan.resizedHeight;
  plan.letterbox.padLeft = (dstWidth - plan.resizedWidth) / 2;
  plan.letterbox.padTop = (dstHeight - plan.resizedHeight) / 2;
  plan.letterbox.padRight = dstWidth - plan.resizedWidth - plan.letterbox.padLeft;
  plan.letterbox.padBottom = dstHeight - plan.resizedHeight - plan.letterbox.padTop;
  return plan;
}

void copyIntoLetterboxedOutput(
    const std::vector<std::uint8_t>& resized,
    int resizedWidth,
    int resizedHeight,
    const ResizePlan& plan,
    std::uint8_t paddingValue,
    RgbImage& output) {
  output.data.assign(output.data.size(), paddingValue);
  const std::size_t resizedStride = static_cast<std::size_t>(resizedWidth * 3);
  const std::size_t dstStride = static_cast<std::size_t>(output.width * 3);
  for (int y = 0; y < resizedHeight; ++y) {
    std::memcpy(
        output.data.data() + static_cast<std::size_t>(y + plan.letterbox.padTop) * dstStride +
            static_cast<std::size_t>(plan.letterbox.padLeft * 3),
        resized.data() + static_cast<std::size_t>(y) * resizedStride,
        resizedStride);
  }
}

__global__ void nv12ToRgbResizeKernel(
    const std::uint8_t* srcY,
    const std::uint8_t* srcUv,
    int srcWidth,
    int srcHeight,
    int srcYStride,
    int srcUvStride,
    std::uint8_t* dstRgb,
    int dstWidth,
    int dstHeight) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstWidth || y >= dstHeight) {
    return;
  }

  const float srcX = static_cast<float>(x) * static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
  const float srcYf = static_cast<float>(y) * static_cast<float>(srcHeight) / static_cast<float>(dstHeight);
  const int sampleX = min(max(static_cast<int>(srcX), 0), srcWidth - 1);
  const int sampleY = min(max(static_cast<int>(srcYf), 0), srcHeight - 1);

  const float Y = static_cast<float>(srcY[sampleY * srcYStride + sampleX]);
  const int uvX = (sampleX / 2) * 2;
  const int uvY = sampleY / 2;
  const int uvIndex = uvY * srcUvStride + uvX;
  const float U = static_cast<float>(srcUv[uvIndex]) - 128.0f;
  const float V = static_cast<float>(srcUv[uvIndex + 1]) - 128.0f;
  const float C = max(0.0f, Y - 16.0f);

  const std::size_t dstIndex = static_cast<std::size_t>((y * dstWidth + x) * 3);
  dstRgb[dstIndex + 0] = clampDevice(1.164f * C + 1.596f * V);
  dstRgb[dstIndex + 1] = clampDevice(1.164f * C - 0.392f * U - 0.813f * V);
  dstRgb[dstIndex + 2] = clampDevice(1.164f * C + 2.017f * U);
}

std::vector<std::uint8_t> convertOnCpu(
    const DecodedFrame& frame,
    int outputWidth,
    int outputHeight) {
  if (frame.yData.empty() || frame.uvData.empty()) {
    throw std::runtime_error("CUDA preprocessor requires NV12 frame data");
  }

  std::vector<std::uint8_t> output(static_cast<std::size_t>(outputWidth * outputHeight * 3));

  const int srcWidth = frame.width;
  const int srcHeight = frame.height;
  const int srcStride = frame.horizontalStride;
  const int uvStride = frame.chromaStride > 0 ? frame.chromaStride : srcStride;

  for (int y = 0; y < outputHeight; ++y) {
    for (int x = 0; x < outputWidth; ++x) {
      const float srcX = static_cast<float>(x) * static_cast<float>(srcWidth) / static_cast<float>(outputWidth);
      const float srcY = static_cast<float>(y) * static_cast<float>(srcHeight) / static_cast<float>(outputHeight);

      const int x0 = std::clamp(static_cast<int>(srcX), 0, srcWidth - 1);
      const int y0 = std::clamp(static_cast<int>(srcY), 0, srcHeight - 1);

      const float Y = static_cast<float>(frame.yData[static_cast<std::size_t>(y0 * srcStride + x0)]);
      const int uvX = (x0 / 2) * 2;
      const int uvY = y0 / 2;
      const std::size_t uvIndex = static_cast<std::size_t>(uvY * uvStride + uvX);
      if (uvIndex + 1 >= frame.uvData.size()) {
        throw std::runtime_error("CUDA preprocessor received truncated UV data");
      }

      const float U = static_cast<float>(frame.uvData[uvIndex]) - 128.0f;
      const float V = static_cast<float>(frame.uvData[uvIndex + 1]) - 128.0f;
      const float C = std::max(0.0f, Y - 16.0f);

      const std::size_t outputIndex = static_cast<std::size_t>((y * outputWidth + x) * 3);
      output[outputIndex + 0] = clampToByte(1.164f * C + 1.596f * V);
      output[outputIndex + 1] = clampToByte(1.164f * C - 0.392f * U - 0.813f * V);
      output[outputIndex + 2] = clampToByte(1.164f * C + 2.017f * U);
    }
  }

  return output;
}

}  // namespace

CudaPreprocessor::CudaPreprocessor() = default;
CudaPreprocessor::~CudaPreprocessor() = default;

void CudaPreprocessor::setGpuId(int gpu_id) {
  gpu_id_ = gpu_id;
}

RgbImage CudaPreprocessor::convertAndResize(
    const DecodedFrame& frame,
    int outputWidth,
    int outputHeight,
    const PreprocessOptions& options) {
  checkCudaStatus(cudaSetDevice(gpu_id_), "Failed to set CUDA device");

  if (frame.width <= 0 || frame.height <= 0) {
    throw std::runtime_error("CUDA preprocessor received an invalid frame size");
  }
  if (frame.horizontalStride < frame.width) {
    throw std::runtime_error("CUDA preprocessor received an invalid horizontal stride");
  }

  const ResizePlan plan = makeResizePlan(frame.width, frame.height, outputWidth, outputHeight, options.letterbox);

  RgbImage output;
  output.width = outputWidth;
  output.height = outputHeight;
  output.format = PixelFormat::kRgb888;
  output.letterbox = plan.letterbox;
  output.data.resize(static_cast<std::size_t>(outputWidth * outputHeight * 3));

  std::vector<std::uint8_t> resizedRgb;
  if (!frame.isOnDevice) {
    resizedRgb = convertOnCpu(frame, plan.resizedWidth, plan.resizedHeight);
  } else {
    if (frame.deviceY == 0 || frame.deviceUv == 0) {
      throw std::runtime_error("CUDA preprocessor received an invalid device NV12 frame");
    }

    resizedRgb.resize(static_cast<std::size_t>(plan.resizedWidth * plan.resizedHeight * 3));
    std::uint8_t* deviceOutput = nullptr;
    const std::size_t outputBytes = resizedRgb.size();
    checkCudaStatus(cudaMalloc(&deviceOutput, outputBytes), "Failed to allocate CUDA RGB output buffer");

    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((plan.resizedWidth + block.x - 1) / block.x),
        static_cast<unsigned int>((plan.resizedHeight + block.y - 1) / block.y));
    nv12ToRgbResizeKernel<<<grid, block>>>(
        reinterpret_cast<const std::uint8_t*>(frame.deviceY),
        reinterpret_cast<const std::uint8_t*>(frame.deviceUv),
        frame.width,
        frame.height,
        frame.horizontalStride,
        frame.chromaStride > 0 ? frame.chromaStride : frame.horizontalStride,
        deviceOutput,
        plan.resizedWidth,
        plan.resizedHeight);
    checkCudaStatus(cudaGetLastError(), "Failed to launch NV12->RGB CUDA kernel");
    checkCudaStatus(cudaDeviceSynchronize(), "NV12->RGB CUDA kernel failed");
    checkCudaStatus(
        cudaMemcpy(resizedRgb.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost),
        "Failed to copy CUDA RGB output to host");
    checkCudaStatus(cudaFree(deviceOutput), "Failed to release CUDA RGB output buffer");
  }

  if (plan.letterbox.enabled) {
    copyIntoLetterboxedOutput(resizedRgb, plan.resizedWidth, plan.resizedHeight, plan, options.paddingValue, output);
  } else {
    output.data = std::move(resizedRgb);
  }

  return output;
}
