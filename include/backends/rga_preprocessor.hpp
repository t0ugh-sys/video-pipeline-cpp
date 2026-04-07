#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

/**
 * Rockchip RGA 硬件预处理器
 * 支持 NV12/BGR/RGB 格式转换 + 缩放
 */
class RgaPreprocessor : public IPreprocessorBackend {
 public:
  RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight) const override;

  std::string name() const override { return "Rockchip RGA"; }
};
