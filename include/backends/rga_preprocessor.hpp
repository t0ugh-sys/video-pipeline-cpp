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
  void ensureWorkspace(std::size_t intermediateBytes, std::size_t resizedBytes, std::size_t outputBytes);

  std::vector<std::uint8_t> intermediateRgb_;
  std::vector<std::uint8_t> resizedRgb_;
  std::vector<std::uint8_t> outputRgb_;
};
