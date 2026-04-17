#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

#include <cstddef>
#include <cstdint>
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

  std::string name() const override { return "Rockchip RGA"; }

 private:
  void ensureBufferGroup();
  void* bufferGroup_ = nullptr;
};
