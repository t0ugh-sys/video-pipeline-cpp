#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

#include <vector>

class RgaPreprocessor : public IPreprocessorBackend {
 public:
  RgaPreprocessor() = default;
  ~RgaPreprocessor() override = default;

  RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight,
      const PreprocessOptions& options = {}) override;

  std::string name() const override { return "Rockchip RGA"; }

 private:
  void ensureWorkspace(std::size_t resizedNv12Bytes, std::size_t resizedRgbBytes, std::size_t outputBytes);

  std::vector<std::uint8_t> resizedNv12_;
  std::vector<std::uint8_t> resizedRgb_;
  std::vector<std::uint8_t> outputRgb_;
};
