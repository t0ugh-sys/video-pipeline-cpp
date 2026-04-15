#include "backends/rga_preprocessor.hpp"

extern "C" {
#include <im2d.h>
#include <im2d_buffer.h>
}

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {

void checkRgaStatus(IM_STATUS status, const char* message) {
  if (status != IM_STATUS_SUCCESS) {
    throw std::runtime_error(message);
  }
}

LetterboxInfo buildLetterboxInfo(
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    bool enabled) {
  LetterboxInfo info;
  if (!enabled) {
    return info;
  }

  const float scale = std::min(
      static_cast<float>(dstWidth) / static_cast<float>(srcWidth),
      static_cast<float>(dstHeight) / static_cast<float>(srcHeight));
  info.enabled = true;
  info.scale = scale;
  info.resizedWidth = std::max(1, static_cast<int>(srcWidth * scale));
  info.resizedHeight = std::max(1, static_cast<int>(srcHeight * scale));
  info.padLeft = (dstWidth - info.resizedWidth) / 2;
  info.padTop = (dstHeight - info.resizedHeight) / 2;
  info.padRight = dstWidth - info.resizedWidth - info.padLeft;
  info.padBottom = dstHeight - info.resizedHeight - info.padTop;
  return info;
}

void blitIntoLetterboxedOutput(
    const std::vector<std::uint8_t>& resized,
    int resizedWidth,
    int resizedHeight,
    std::uint8_t paddingValue,
    const LetterboxInfo& info,
    std::vector<std::uint8_t>& output,
    int outputWidth) {
  output.assign(output.size(), paddingValue);
  const std::size_t resizedStride = static_cast<std::size_t>(resizedWidth * 3);
  const std::size_t dstStride = static_cast<std::size_t>(outputWidth * 3);
  for (int y = 0; y < resizedHeight; ++y) {
    std::memcpy(
        output.data() + static_cast<std::size_t>(y + info.padTop) * dstStride +
            static_cast<std::size_t>(info.padLeft * 3),
        resized.data() + static_cast<std::size_t>(y) * resizedStride,
        resizedStride);
  }
}

int bufferBytesForNv12(const DecodedFrame& frame) {
  return frame.horizontalStride * frame.verticalStride * 3 / 2;
}

}  // namespace

void RgaPreprocessor::ensureWorkspace(
    std::size_t intermediateBytes,
    std::size_t resizedBytes,
    std::size_t outputBytes) {
  if (intermediateRgb_.size() < intermediateBytes) {
    intermediateRgb_.resize(intermediateBytes);
  }
  if (resizedRgb_.size() < resizedBytes) {
    resizedRgb_.resize(resizedBytes);
  }
  if (outputRgb_.size() < outputBytes) {
    outputRgb_.resize(outputBytes);
  }
}

RgbImage RgaPreprocessor::convertAndResize(
    const DecodedFrame& frame,
    int outputWidth,
    int outputHeight,
    const PreprocessOptions& options) {
  if (frame.dmaFd < 0) {
    throw std::runtime_error("Decoded frame does not provide a valid dma fd");
  }
  if (frame.format != PixelFormat::kUnknown && frame.format != PixelFormat::kNv12) {
    throw std::runtime_error("RGA preprocessor currently expects NV12 decoded frames");
  }

  RgbImage output;
  output.width = outputWidth;
  output.height = outputHeight;
  output.format = PixelFormat::kRgb888;
  output.letterbox = buildLetterboxInfo(frame.width, frame.height, outputWidth, outputHeight, options.letterbox);

  const int resizedWidth = output.letterbox.enabled ? output.letterbox.resizedWidth : outputWidth;
  const int resizedHeight = output.letterbox.enabled ? output.letterbox.resizedHeight : outputHeight;
  const std::size_t intermediateBytes = static_cast<std::size_t>(frame.width * frame.height * 3);
  const std::size_t resizedBytes = static_cast<std::size_t>(resizedWidth * resizedHeight * 3);
  const std::size_t outputBytes = static_cast<std::size_t>(outputWidth * outputHeight * 3);
  ensureWorkspace(intermediateBytes, resizedBytes, outputBytes);

  rga_buffer_handle_t srcHandle = importbuffer_fd(frame.dmaFd, bufferBytesForNv12(frame));
  rga_buffer_handle_t intermediateHandle = importbuffer_virtualaddr(intermediateRgb_.data(), intermediateBytes);
  rga_buffer_handle_t resizedHandle = importbuffer_virtualaddr(resizedRgb_.data(), resizedBytes);
  if (srcHandle == 0 || intermediateHandle == 0 || resizedHandle == 0) {
    if (srcHandle != 0) {
      releasebuffer_handle(srcHandle);
    }
    if (intermediateHandle != 0) {
      releasebuffer_handle(intermediateHandle);
    }
    if (resizedHandle != 0) {
      releasebuffer_handle(resizedHandle);
    }
    throw std::runtime_error("Failed to import RGA buffers");
  }

  rga_buffer_t src = wrapbuffer_handle(
      srcHandle,
      frame.width,
      frame.height,
      RK_FORMAT_YCbCr_420_SP,
      frame.horizontalStride,
      frame.verticalStride);
  rga_buffer_t intermediate = wrapbuffer_handle(
      intermediateHandle,
      frame.width,
      frame.height,
      RK_FORMAT_RGB_888,
      frame.width,
      frame.height);
  rga_buffer_t resized = wrapbuffer_handle(
      resizedHandle,
      resizedWidth,
      resizedHeight,
      RK_FORMAT_RGB_888,
      resizedWidth,
      resizedHeight);

  checkRgaStatus(
      imcvtcolor(src, intermediate, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
      "RGA color conversion failed");
  if (resizedWidth != frame.width || resizedHeight != frame.height) {
    checkRgaStatus(imresize(intermediate, resized), "RGA resize failed");
  } else {
    std::memcpy(resizedRgb_.data(), intermediateRgb_.data(), resizedBytes);
  }

  releasebuffer_handle(srcHandle);
  releasebuffer_handle(intermediateHandle);
  releasebuffer_handle(resizedHandle);

  if (output.letterbox.enabled) {
    blitIntoLetterboxedOutput(
        resizedRgb_,
        resizedWidth,
        resizedHeight,
        options.paddingValue,
        output.letterbox,
        outputRgb_,
        outputWidth);
  } else {
    std::memcpy(outputRgb_.data(), resizedRgb_.data(), resizedBytes);
  }

  output.data.assign(outputRgb_.begin(), outputRgb_.begin() + static_cast<std::ptrdiff_t>(outputBytes));
  return output;
}
