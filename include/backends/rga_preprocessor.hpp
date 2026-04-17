#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

#include <im2d.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

class RgaPreprocessor : public IPreprocessorBackend {
 public:
  RgaPreprocessor() = default;
  ~RgaPreprocessor() override;

  RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight,
      const PreprocessOptions& options = {}) override;
  void setMaxInflightFrames(std::size_t maxInflightFrames) override;

  std::string name() const override { return "Rockchip RGA"; }

 private:
  void ensureBufferGroup(std::size_t maxBufferBytes);
  void ensureResizedNv12Buffer(int width, int height, int wstride, int hstride, std::size_t bytes);
  void ensureResizedRgbBuffer(int width, int height, int wstride, int hstride, std::size_t bytes);
  void releasePersistentBuffers();
  void* bufferGroup_ = nullptr;
  std::size_t configuredMaxBufferBytes_ = 0;
  std::size_t configuredMaxInflightFrames_ = 0;
  std::size_t maxInflightFrames_ = 4;
  void* resizedNv12Buffer_ = nullptr;
  int resizedNv12Width_ = 0;
  int resizedNv12Height_ = 0;
  int resizedNv12Wstride_ = 0;
  int resizedNv12Hstride_ = 0;
  std::size_t resizedNv12Bytes_ = 0;
  int resizedNv12Fd_ = -1;
  rga_buffer_handle_t resizedNv12Handle_ = 0;
  void* resizedRgbBuffer_ = nullptr;
  int resizedRgbWidth_ = 0;
  int resizedRgbHeight_ = 0;
  int resizedRgbWstride_ = 0;
  int resizedRgbHstride_ = 0;
  std::size_t resizedRgbBytes_ = 0;
  int resizedRgbFd_ = -1;
  rga_buffer_handle_t resizedRgbHandle_ = 0;
};
