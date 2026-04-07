#pragma once

#include "pipeline_types.hpp"

#include <memory>
#include <string>

/**
 * 预处理后端抽象接口
 * 支持：Rockchip RGA | NVIDIA CUDA/nvPIPI | CPU (OpenCV/libyuv)
 */
class IPreprocessorBackend {
 public:
  virtual ~IPreprocessorBackend() = default;

  /**
   * 转换并缩放帧
   * @param frame 解码后的帧
   * @param outputWidth 输出宽度
   * @param outputHeight 输出高度
   * @return RGB 图像
   */
  virtual RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight) = 0;

  /** 获取后端名称 */
  virtual std::string name() const = 0;
};

/**
 * 后端类型枚举
 */
enum class PreprocBackendType {
  kAuto,        ///< 自动选择
  kRockchipRga, ///< Rockchip RGA 硬件加速
  kNvidiaCuda,  ///< NVIDIA CUDA/nvPIPI
  kCpu,         ///< CPU 软件转换
};

/**
 * 创建预处理后端实例
 */
std::unique_ptr<IPreprocessorBackend> createPreprocBackend(PreprocBackendType type = PreprocBackendType::kAuto);

/**
 * 检测当前平台可用的预处理后端
 */
PreprocBackendType detectAvailablePreprocBackend();
