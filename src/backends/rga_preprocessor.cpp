#include "backends/rga_preprocessor.hpp"

extern "C" {
#include <mpp_buffer.h>
#include <mpp_frame.h>
}
#include <im2d.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

// RGA hardware requires aligned stride for correct operation. 16 is a safe
// universal choice covering RGA2/RGA3 and both NV12 (min 2) and RGB_888 (min 4).
constexpr int kRgaStrideAlign = 16;

inline int alignUp(int value, int alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// imresize / imcvtcolor return IM_STATUS_SUCCESS (=1) on success.
void checkRgaOp(IM_STATUS status, const char* stage) {
  if (status != IM_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("RGA ") + stage + " failed: " + imStrError_t(status));
  }
}

// imcheck returns IM_STATUS_NOERROR (=2) when parameters are valid.
void checkRgaVerify(int ret, const char* stage) {
  if (ret != IM_STATUS_NOERROR) {
    throw std::runtime_error(
        std::string("RGA imcheck before ") + stage + " failed: " +
        imStrError_t(static_cast<IM_STATUS>(ret)));
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

// Row-copy from an aligned-stride source into a packed destination.
void copyStridedToPacked(
    const std::uint8_t* src,
    int srcRowBytes,
    std::uint8_t* dst,
    int dstRowBytes,
    int height) {
  for (int y = 0; y < height; ++y) {
    std::memcpy(dst + y * dstRowBytes, src + y * srcRowBytes, dstRowBytes);
  }
}

int nv12BytesForStride(int wstride, int hstride) {
  return wstride * hstride * 3 / 2;
}

int rgb888BytesForStride(int wstride, int hstride) {
  return wstride * hstride * 3;
}

void releaseHandle(rga_buffer_handle_t& handle) {
  if (handle != 0) {
    releasebuffer_handle(handle);
    handle = 0;
  }
}

std::shared_ptr<void> makeMppBufferHandle(MppBuffer buffer) {
  return std::shared_ptr<void>(buffer, [](void* opaque) {
    if (opaque != nullptr) {
      MppBuffer buffer = opaque;
      mpp_buffer_put(buffer);
    }
  });
}

MppBuffer allocateBuffer(MppBufferGroup group, std::size_t size, const char* stage) {
  MppBuffer buffer = nullptr;
  if (mpp_buffer_get(group, &buffer, size) != MPP_OK || buffer == nullptr) {
    throw std::runtime_error(std::string("Failed to allocate DRM buffer for ") + stage);
  }
  return buffer;
}

void fillPackedRgbData(
    MppBuffer buffer,
    int srcWstride,
    int width,
    int height,
    std::vector<std::uint8_t>& output) {
  output.resize(static_cast<std::size_t>(width * height * 3));
  const auto* src = static_cast<const std::uint8_t*>(mpp_buffer_get_ptr(buffer));
  if (src == nullptr) {
    throw std::runtime_error("Failed to map RGA output DRM buffer");
  }
  copyStridedToPacked(src, srcWstride * 3, output.data(), width * 3, height);
}

}  // namespace

RgaPreprocessor::~RgaPreprocessor() {
  releasePersistentBuffers();
  if (bufferGroup_ != nullptr) {
    mpp_buffer_group_put(static_cast<MppBufferGroup>(bufferGroup_));
    bufferGroup_ = nullptr;
  }
}

void RgaPreprocessor::setMaxInflightFrames(std::size_t maxInflightFrames) {
  maxInflightFrames_ = std::max<std::size_t>(1, maxInflightFrames);
}

void RgaPreprocessor::ensureBufferGroup(std::size_t maxBufferBytes) {
  if (maxBufferBytes == 0) {
    throw std::runtime_error("RGA buffer group requires a non-zero max buffer size");
  }

  const auto requestedLimit = static_cast<RK_S32>(std::min<std::size_t>(
      maxInflightFrames_,
      static_cast<std::size_t>(std::numeric_limits<RK_S32>::max())));

  if (bufferGroup_ != nullptr &&
      configuredMaxBufferBytes_ >= maxBufferBytes &&
      configuredMaxInflightFrames_ == static_cast<std::size_t>(requestedLimit)) {
    return;
  }

  MppBufferGroup group = nullptr;
  if (mpp_buffer_group_get_internal(
          &group,
          static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)) != MPP_OK) {
    throw std::runtime_error("mpp_buffer_group_get_internal for RGA output failed");
  }

  if (mpp_buffer_group_limit_config(group, maxBufferBytes, requestedLimit) != MPP_OK) {
    mpp_buffer_group_put(group);
    throw std::runtime_error("mpp_buffer_group_limit_config for RGA output failed");
  }

  if (bufferGroup_ != nullptr) {
    releasePersistentBuffers();
    mpp_buffer_group_put(static_cast<MppBufferGroup>(bufferGroup_));
  }
  bufferGroup_ = group;
  configuredMaxBufferBytes_ = maxBufferBytes;
  configuredMaxInflightFrames_ = static_cast<std::size_t>(requestedLimit);
}

void RgaPreprocessor::releasePersistentBuffers() {
  releaseHandle(resizedNv12Handle_);
  releaseHandle(resizedRgbHandle_);

  if (resizedNv12Buffer_ != nullptr) {
    mpp_buffer_put(static_cast<MppBuffer>(resizedNv12Buffer_));
    resizedNv12Buffer_ = nullptr;
  }
  if (resizedRgbBuffer_ != nullptr) {
    mpp_buffer_put(static_cast<MppBuffer>(resizedRgbBuffer_));
    resizedRgbBuffer_ = nullptr;
  }

  resizedNv12Width_ = 0;
  resizedNv12Height_ = 0;
  resizedNv12Wstride_ = 0;
  resizedNv12Hstride_ = 0;
  resizedNv12Bytes_ = 0;
  resizedNv12Fd_ = -1;

  resizedRgbWidth_ = 0;
  resizedRgbHeight_ = 0;
  resizedRgbWstride_ = 0;
  resizedRgbHstride_ = 0;
  resizedRgbBytes_ = 0;
  resizedRgbFd_ = -1;
}

void RgaPreprocessor::ensureResizedNv12Buffer(
    int width,
    int height,
    int wstride,
    int hstride,
    std::size_t bytes) {
  if (bufferGroup_ == nullptr) {
    throw std::runtime_error("RGA resized NV12 buffer group is not initialized");
  }

  if (resizedNv12Buffer_ != nullptr &&
      resizedNv12Width_ == width &&
      resizedNv12Height_ == height &&
      resizedNv12Wstride_ == wstride &&
      resizedNv12Hstride_ == hstride &&
      resizedNv12Bytes_ == bytes &&
      resizedNv12Fd_ >= 0 &&
      resizedNv12Handle_ != 0) {
    return;
  }

  releaseHandle(resizedNv12Handle_);
  if (resizedNv12Buffer_ != nullptr) {
    mpp_buffer_put(static_cast<MppBuffer>(resizedNv12Buffer_));
    resizedNv12Buffer_ = nullptr;
  }

  MppBuffer buffer = allocateBuffer(static_cast<MppBufferGroup>(bufferGroup_), bytes, "RGA resized NV12");
  const int fd = mpp_buffer_get_fd(buffer);
  if (fd < 0) {
    mpp_buffer_put(buffer);
    throw std::runtime_error("mpp_buffer_get_fd failed for resized NV12 buffer");
  }

  const rga_buffer_handle_t handle = importbuffer_fd(fd, static_cast<int>(bytes));
  if (handle == 0) {
    mpp_buffer_put(buffer);
    throw std::runtime_error("Failed to import resized NV12 RGA buffer");
  }

  resizedNv12Buffer_ = buffer;
  resizedNv12Width_ = width;
  resizedNv12Height_ = height;
  resizedNv12Wstride_ = wstride;
  resizedNv12Hstride_ = hstride;
  resizedNv12Bytes_ = bytes;
  resizedNv12Fd_ = fd;
  resizedNv12Handle_ = handle;
}

void RgaPreprocessor::ensureResizedRgbBuffer(
    int width,
    int height,
    int wstride,
    int hstride,
    std::size_t bytes) {
  if (bufferGroup_ == nullptr) {
    throw std::runtime_error("RGA resized RGB buffer group is not initialized");
  }

  if (resizedRgbBuffer_ != nullptr &&
      resizedRgbWidth_ == width &&
      resizedRgbHeight_ == height &&
      resizedRgbWstride_ == wstride &&
      resizedRgbHstride_ == hstride &&
      resizedRgbBytes_ == bytes &&
      resizedRgbFd_ >= 0 &&
      resizedRgbHandle_ != 0) {
    return;
  }

  releaseHandle(resizedRgbHandle_);
  if (resizedRgbBuffer_ != nullptr) {
    mpp_buffer_put(static_cast<MppBuffer>(resizedRgbBuffer_));
    resizedRgbBuffer_ = nullptr;
  }

  MppBuffer buffer = allocateBuffer(static_cast<MppBufferGroup>(bufferGroup_), bytes, "RGA resized RGB");
  const int fd = mpp_buffer_get_fd(buffer);
  if (fd < 0) {
    mpp_buffer_put(buffer);
    throw std::runtime_error("mpp_buffer_get_fd failed for resized RGB buffer");
  }

  const rga_buffer_handle_t handle = importbuffer_fd(fd, static_cast<int>(bytes));
  if (handle == 0) {
    mpp_buffer_put(buffer);
    throw std::runtime_error("Failed to import resized RGB RGA buffer");
  }

  resizedRgbBuffer_ = buffer;
  resizedRgbWidth_ = width;
  resizedRgbHeight_ = height;
  resizedRgbWstride_ = wstride;
  resizedRgbHstride_ = hstride;
  resizedRgbBytes_ = bytes;
  resizedRgbFd_ = fd;
  resizedRgbHandle_ = handle;
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
  if (frame.nativeFormat != 0 && frame.nativeFormat != MPP_FMT_YUV420SP) {
    throw std::runtime_error(
        "RGA preprocessor expects linear MPP_FMT_YUV420SP frames, got native format=" +
        std::to_string(frame.nativeFormat));
  }

  RgbImage output;
  output.width = outputWidth;
  output.height = outputHeight;
  output.format = PixelFormat::kRgb888;
  output.wstride = outputWidth;
  output.hstride = outputHeight;
  output.dmaFd = -1;
  output.dmaSize = 0;
  output.letterbox = buildLetterboxInfo(frame.width, frame.height, outputWidth, outputHeight, options.letterbox);

  const int resizedWidth = output.letterbox.enabled ? output.letterbox.resizedWidth : outputWidth;
  const int resizedHeight = output.letterbox.enabled ? output.letterbox.resizedHeight : outputHeight;
  const bool needsResize = frame.width != resizedWidth || frame.height != resizedHeight;
  const std::size_t outputRgbBytes = static_cast<std::size_t>(outputWidth * outputHeight * 3);

  if (!options.needsCpuData) {
    rga_buffer_handle_t srcHandle = 0;
    rga_buffer_handle_t outputHandle = 0;
    MppBuffer outputBuffer = nullptr;

    try {
      const int resizedNv12Wstride = alignUp(resizedWidth, kRgaStrideAlign);
      const int resizedNv12Hstride = alignUp(resizedHeight, 2);
      const int resizedRgbWstride = alignUp(resizedWidth, kRgaStrideAlign);
      const int resizedRgbHstride = resizedHeight;
      const int outputRgbWstride = alignUp(outputWidth, kRgaStrideAlign);
      const int outputRgbHstride = outputHeight;
      const std::size_t resizedNv12Bytes =
          needsResize ? static_cast<std::size_t>(nv12BytesForStride(resizedNv12Wstride, resizedNv12Hstride)) : 0;
      const std::size_t resizedRgbBytes =
          static_cast<std::size_t>(rgb888BytesForStride(resizedRgbWstride, resizedRgbHstride));
      const std::size_t outputAlignedRgbBytes =
          static_cast<std::size_t>(rgb888BytesForStride(outputRgbWstride, outputRgbHstride));
      const std::size_t maxBufferBytes =
          std::max(outputAlignedRgbBytes, std::max(resizedNv12Bytes, resizedRgbBytes));
      ensureBufferGroup(maxBufferBytes);

      outputBuffer = allocateBuffer(static_cast<MppBufferGroup>(bufferGroup_), outputAlignedRgbBytes, "RGA output");
      output.dmaFd = mpp_buffer_get_fd(outputBuffer);
      if (output.dmaFd < 0) {
        throw std::runtime_error("mpp_buffer_get_fd failed for RGA output buffer");
      }
      output.wstride = outputRgbWstride;
      output.hstride = outputRgbHstride;
      output.dmaSize = outputAlignedRgbBytes;
      output.nativeHandle = makeMppBufferHandle(outputBuffer);
      outputBuffer = nullptr;

      outputHandle = importbuffer_fd(output.dmaFd, static_cast<int>(outputAlignedRgbBytes));
      if (outputHandle == 0) {
        throw std::runtime_error("Failed to import RGA output DRM buffer");
      }

      srcHandle = importbuffer_fd(
          frame.dmaFd,
          frame.horizontalStride,
          frame.verticalStride,
          RK_FORMAT_YCbCr_420_SP);
      if (srcHandle == 0) {
        throw std::runtime_error("RGA importbuffer_fd failed for decoded frame");
      }

      const im_rect emptyRect = {};
      const rga_buffer_t emptyPat = {};
      rga_buffer_t src = wrapbuffer_handle(
          srcHandle,
          frame.width,
          frame.height,
          RK_FORMAT_YCbCr_420_SP,
          frame.horizontalStride,
          frame.verticalStride);
      rga_buffer_t outputRgb = wrapbuffer_handle(
          outputHandle,
          outputWidth,
          outputHeight,
          RK_FORMAT_RGB_888,
          outputRgbWstride,
          outputRgbHstride);

      if (needsResize) {
        ensureResizedNv12Buffer(
            resizedWidth,
            resizedHeight,
            resizedNv12Wstride,
            resizedNv12Hstride,
            resizedNv12Bytes);

        rga_buffer_t resizedNv12 = wrapbuffer_handle(
            resizedNv12Handle_,
            resizedWidth,
            resizedHeight,
            RK_FORMAT_YCbCr_420_SP,
            resizedNv12Wstride,
            resizedNv12Hstride);

        checkRgaVerify(imcheck_t(src, resizedNv12, emptyPat, emptyRect, emptyRect, emptyRect, 0), "imresize(NV12)");
        checkRgaOp(imresize(src, resizedNv12), "imresize(NV12)");

        if (output.letterbox.enabled) {
          ensureResizedRgbBuffer(
              resizedWidth,
              resizedHeight,
              resizedRgbWstride,
              resizedRgbHstride,
              resizedRgbBytes);

          rga_buffer_t resizedRgb = wrapbuffer_handle(
              resizedRgbHandle_,
              resizedWidth,
              resizedHeight,
              RK_FORMAT_RGB_888,
              resizedRgbWstride,
              resizedRgbHstride);

          checkRgaOp(
              imcvtcolor(resizedNv12, resizedRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
              "imcvtcolor(NV12->RGB)");

          const int top = output.letterbox.padTop;
          const int bottom = output.letterbox.padBottom;
          const int left = output.letterbox.padLeft;
          const int right = output.letterbox.padRight;

          const IM_STATUS borderStatus = immakeBorder(
              resizedRgb,
              outputRgb,
              top,
              bottom,
              left,
              right,
              IM_BORDER_CONSTANT,
              static_cast<int>(options.paddingValue),
              1,
              -1,
              nullptr);
          if (borderStatus != IM_STATUS_SUCCESS) {
            throw std::runtime_error(
                std::string("RGA immakeBorder failed: ") + imStrError_t(borderStatus));
          }
        } else {
          checkRgaOp(
              imcvtcolor(resizedNv12, outputRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
              "imcvtcolor(NV12->RGB resized direct)");
        }
      } else {
        checkRgaOp(
            imcvtcolor(src, outputRgb, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
            "imcvtcolor(NV12->RGB direct)");
      }

      releaseHandle(srcHandle);
      releaseHandle(outputHandle);
      return output;
    } catch (const std::exception& error) {
      static bool loggedZeroCopyFallback = false;
      if (!loggedZeroCopyFallback) {
        loggedZeroCopyFallback = true;
        std::cerr << "[RGA] zero-copy preprocess fallback to host-copy: "
                  << error.what() << "\n";
      }
      releaseHandle(srcHandle);
      releaseHandle(outputHandle);
      output.nativeHandle.reset();
      output.dmaFd = -1;
      output.dmaSize = 0;
      output.wstride = outputWidth;
      output.hstride = outputHeight;
      if (outputBuffer != nullptr) {
        mpp_buffer_put(outputBuffer);
      }
    }
  }

  output.data.resize(outputRgbBytes);

  rga_buffer_handle_t srcHandle = 0;
  rga_buffer_handle_t resizedNv12Handle = 0;
  rga_buffer_handle_t resizedRgbHandle = 0;
  rga_buffer_handle_t outputHandle = 0;

  std::vector<std::uint8_t> resizedNv12;
  std::vector<std::uint8_t> resizedRgb;

  try {
    srcHandle = importbuffer_fd(
        frame.dmaFd,
        frame.horizontalStride,
        frame.verticalStride,
        RK_FORMAT_YCbCr_420_SP);
    if (srcHandle == 0) {
      throw std::runtime_error("RGA importbuffer_fd failed for decoded frame");
    }

    rga_buffer_t src = wrapbuffer_handle(
        srcHandle,
        frame.width,
        frame.height,
        RK_FORMAT_YCbCr_420_SP,
        frame.horizontalStride,
        frame.verticalStride);

    if (!needsResize && !output.letterbox.enabled) {
      const int outputRgbWstride = alignUp(outputWidth, kRgaStrideAlign);
      const int outputRgbHstride = outputHeight;
      const std::size_t outputAlignedRgbBytes =
          static_cast<std::size_t>(rgb888BytesForStride(outputRgbWstride, outputRgbHstride));
      ensureBufferGroup(outputAlignedRgbBytes);
      ensureResizedRgbBuffer(
          outputWidth,
          outputHeight,
          outputRgbWstride,
          outputRgbHstride,
          outputAlignedRgbBytes);

      rga_buffer_t dst = wrapbuffer_handle(
          resizedRgbHandle_,
          outputWidth,
          outputHeight,
          RK_FORMAT_RGB_888,
          outputRgbWstride,
          outputRgbHstride);
      // This path intentionally converts into a DRM-backed RGB buffer first and
      // only then copies out row-by-row. Direct RGA writes into host RGB memory
      // looked simpler, but produced visible line artifacts on the target board.
      checkRgaOp(
          imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
          "imcvtcolor(NV12->RGB direct drm)");
      fillPackedRgbData(
          static_cast<MppBuffer>(resizedRgbBuffer_),
          outputRgbWstride,
          outputWidth,
          outputHeight,
          output.data);
    } else {
      const std::size_t resizedNv12Bytes =
          static_cast<std::size_t>(resizedWidth * resizedHeight * 3 / 2);
      const std::size_t resizedRgbBytes =
          static_cast<std::size_t>(resizedWidth * resizedHeight * 3);

      resizedRgb.resize(resizedRgbBytes);
      resizedRgbHandle = importbuffer_virtualaddr(resizedRgb.data(), static_cast<int>(resizedRgbBytes));
      if (resizedRgbHandle == 0) {
        throw std::runtime_error("RGA importbuffer_virtualaddr failed for resized RGB buffer");
      }

      rga_buffer_t resizedRgbBuf = wrapbuffer_handle(
          resizedRgbHandle,
          resizedWidth,
          resizedHeight,
          RK_FORMAT_RGB_888);

      if (needsResize) {
        resizedNv12.resize(resizedNv12Bytes);
        resizedNv12Handle = importbuffer_virtualaddr(resizedNv12.data(), static_cast<int>(resizedNv12Bytes));
        if (resizedNv12Handle == 0) {
          throw std::runtime_error("RGA importbuffer_virtualaddr failed for resized NV12 buffer");
        }

        rga_buffer_t resizedNv12Buf = wrapbuffer_handle(
            resizedNv12Handle,
            resizedWidth,
            resizedHeight,
            RK_FORMAT_YCbCr_420_SP);
        checkRgaOp(imresize(src, resizedNv12Buf), "imresize(NV12 host)");
        checkRgaOp(
            imcvtcolor(resizedNv12Buf, resizedRgbBuf, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
            "imcvtcolor(NV12->RGB resized host)");
      } else {
        checkRgaOp(
            imcvtcolor(src, resizedRgbBuf, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
            "imcvtcolor(NV12->RGB host)");
      }

      std::memset(output.data.data(), options.paddingValue, outputRgbBytes);
      const int left = output.letterbox.enabled ? output.letterbox.padLeft : 0;
      const int top = output.letterbox.enabled ? output.letterbox.padTop : 0;
      for (int y = 0; y < resizedHeight; ++y) {
        std::memcpy(
            output.data.data() +
                static_cast<std::size_t>(((y + top) * outputWidth + left) * 3),
            resizedRgb.data() + static_cast<std::size_t>(y * resizedWidth * 3),
            static_cast<std::size_t>(resizedWidth * 3));
      }
    }
  } catch (...) {
    releaseHandle(outputHandle);
    releaseHandle(resizedRgbHandle);
    releaseHandle(resizedNv12Handle);
    releaseHandle(srcHandle);
    throw;
  }

  releaseHandle(outputHandle);
  releaseHandle(resizedRgbHandle);
  releaseHandle(resizedNv12Handle);
  releaseHandle(srcHandle);
  return output;
}
