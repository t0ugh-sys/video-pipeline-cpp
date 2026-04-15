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
  std::fill(output.begin(), output.end(), paddingValue);
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

int bufferBytesForNv12(int width, int height) {
  return width * height * 3 / 2;
}

int bufferBytesForNv12(const DecodedFrame& frame) {
  return frame.horizontalStride * frame.verticalStride * 3 / 2;
}

void releaseHandle(rga_buffer_handle_t& handle) {
  if (handle != 0) {
    releasebuffer_handle(handle);
    handle = 0;
  }
}

}  // namespace

void RgaPreprocessor::ensureWorkspace(
    std::size_t resizedNv12Bytes,
    std::size_t resizedRgbBytes,
    std::size_t outputBytes) {
  if (resizedNv12_.size() < resizedNv12Bytes) {
    resizedNv12_.resize(resizedNv12Bytes);
  }
  if (resizedRgb_.size() < resizedRgbBytes) {
    resizedRgb_.resize(resizedRgbBytes);
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
  const bool needsResize = frame.width != resizedWidth || frame.height != resizedHeight;
  const std::size_t resizedNv12Bytes = needsResize ? static_cast<std::size_t>(bufferBytesForNv12(resizedWidth, resizedHeight)) : 0;
  const std::size_t resizedRgbBytes = static_cast<std::size_t>(resizedWidth * resizedHeight * 3);
  const std::size_t outputBytes = static_cast<std::size_t>(outputWidth * outputHeight * 3);
  ensureWorkspace(resizedNv12Bytes, resizedRgbBytes, outputBytes);

  rga_buffer_handle_t srcHandle = 0;
  rga_buffer_handle_t resizedNv12Handle = 0;
  rga_buffer_handle_t rgbHandle = 0;

  try {
    srcHandle = importbuffer_fd(frame.dmaFd, bufferBytesForNv12(frame));
    if (srcHandle == 0) {
      throw std::runtime_error("Failed to import source RGA buffer");
    }

    std::vector<std::uint8_t>* rgbBuffer = output.letterbox.enabled ? &resizedRgb_ : &outputRgb_;
    rgbHandle = importbuffer_virtualaddr(rgbBuffer->data(), resizedRgbBytes);
    if (rgbHandle == 0) {
      throw std::runtime_error("Failed to import RGB RGA buffer");
    }

    rga_buffer_t src = wrapbuffer_handle(
        srcHandle,
        frame.width,
        frame.height,
        RK_FORMAT_YCbCr_420_SP,
        frame.horizontalStride,
        frame.verticalStride);

    if (needsResize) {
      resizedNv12Handle = importbuffer_virtualaddr(resizedNv12_.data(), resizedNv12Bytes);
      if (resizedNv12Handle == 0) {
        throw std::runtime_error("Failed to import resized NV12 RGA buffer");
      }

      rga_buffer_t resizedNv12 = wrapbuffer_handle(
          resizedNv12Handle,
          resizedWidth,
          resizedHeight,
          RK_FORMAT_YCbCr_420_SP,
          resizedWidth,
          resizedHeight);
      checkRgaStatus(imresize(src, resizedNv12), "RGA NV12 resize failed");

      rga_buffer_t resizedRgb = wrapbuffer_handle(
          rgbHandle,
          resizedWidth,
          resizedHeight,
          RK_FORMAT_RGB_888,
          resizedWidth,
          resizedHeight);
      checkRgaStatus(
          imcvtcolor(resizedNv12, resizedRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
          "RGA color conversion failed");
    } else {
      rga_buffer_t outputRgb = wrapbuffer_handle(
          rgbHandle,
          resizedWidth,
          resizedHeight,
          RK_FORMAT_RGB_888,
          resizedWidth,
          resizedHeight);
      checkRgaStatus(
          imcvtcolor(src, outputRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
          "RGA color conversion failed");
    }

    releaseHandle(srcHandle);
    releaseHandle(resizedNv12Handle);
    releaseHandle(rgbHandle);
  } catch (...) {
    releaseHandle(srcHandle);
    releaseHandle(resizedNv12Handle);
    releaseHandle(rgbHandle);
    throw;
  }

  if (output.letterbox.enabled) {
    blitIntoLetterboxedOutput(
        resizedRgb_,
        resizedWidth,
        resizedHeight,
        options.paddingValue,
        output.letterbox,
        outputRgb_,
        outputWidth);
  }

  output.data.assign(outputRgb_.begin(), outputRgb_.begin() + static_cast<std::ptrdiff_t>(outputBytes));
  return output;
}
