#include "backends/rga_preprocessor.hpp"

extern "C" {
#include <im2d.h>
#include <im2d_buffer.h>
}

#include <vector>
#include <stdexcept>

namespace {

void checkRgaStatus(IM_STATUS status, const char* message) {
  if (status != IM_STATUS_SUCCESS) {
    throw std::runtime_error(message);
  }
}

}  // namespace

RgbImage RgaPreprocessor::convertAndResize(
    const DecodedFrame& frame,
    int outputWidth,
    int outputHeight) const {
  if (frame.dmaFd < 0) {
    throw std::runtime_error("Decoded frame does not provide a valid dma fd");
  }

  RgbImage output;
  output.width = outputWidth;
  output.height = outputHeight;
  output.data.resize(static_cast<std::size_t>(outputWidth * outputHeight * 3));
  std::vector<std::uint8_t> intermediateRgb(
      static_cast<std::size_t>(frame.width * frame.height * 3));

  rga_buffer_t src = wrapbuffer_fd(
      frame.dmaFd,
      frame.width,
      frame.height,
      frame.horizontalStride,
      frame.verticalStride,
      RK_FORMAT_YCbCr_420_SP);

  // 先做颜色转换，再做 resize
  rga_buffer_t intermediate = wrapbuffer_virtualaddr(
      intermediateRgb.data(),
      frame.width,
      frame.height,
      RK_FORMAT_RGB_888);

  rga_buffer_t dst = wrapbuffer_virtualaddr(
      output.data.data(),
      outputWidth,
      outputHeight,
      RK_FORMAT_RGB_888);

  im_rect srcRect{};
  srcRect.x = 0;
  srcRect.y = 0;
  srcRect.width = frame.width;
  srcRect.height = frame.height;

  im_rect dstRect{};
  dstRect.x = 0;
  dstRect.y = 0;
  dstRect.width = outputWidth;
  dstRect.height = outputHeight;

  checkRgaStatus(
      imcheck(src, intermediate, srcRect, srcRect),
      "RGA NV12->RGB parameter check failed");
  checkRgaStatus(
      imcvtcolor(src, intermediate, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888),
      "RGA color conversion failed");
  checkRgaStatus(
      imcheck(intermediate, dst, srcRect, dstRect),
      "RGA resize parameter check failed");
  checkRgaStatus(imresize(intermediate, dst), "RGA resize failed");
  return output;
}
