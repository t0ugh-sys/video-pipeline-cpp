#pragma once

#include "pipeline_types.hpp"

#include <memory>
#include <optional>
#include <string>

/**
 * 解码器后端抽象接口
 * 支持：Rockchip MPP | NVIDIA NVDEC (FFmpeg) | CPU (FFmpeg 软件解码)
 */
class IDecoderBackend {
 public:
  virtual ~IDecoderBackend() = default;

  /** 初始化解码器 */
  virtual void open(VideoCodec codec) = 0;

  /** 解码一帧 */
  virtual std::optional<DecodedFrame> decode(const EncodedPacket& packet) = 0;

  /** 获取解码器名称 */
  virtual std::string name() const = 0;
};

/**
 * 后端类型枚举
 */
enum class DecoderBackendType {
  kAuto,        ///< 自动选择
  kRockchipMpp, ///< Rockchip MPP 硬件解码
  kNvidiaNvdec, ///< NVIDIA NVDEC 硬件解码 (FFmpeg + cuvid)
  kCpu,         ///< CPU 软件解码 (FFmpeg)
};

/**
 * 创建解码器后端实例
 *
 * @param type 后端类型，kAuto 时自动检测
 * @return 解码器后端智能指针
 */
std::unique_ptr<IDecoderBackend> createDecoderBackend(DecoderBackendType type = DecoderBackendType::kAuto);

/**
 * 检测当前平台可用的解码器后端
 */
DecoderBackendType detectAvailableDecoderBackend();
