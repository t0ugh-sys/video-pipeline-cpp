#pragma once

#include "decoder_interface.hpp"
#include "pipeline_types.hpp"

#include <optional>
#include <vector>
#include <cstdint>

// FFmpeg forward declarations
struct AVCodecContext;
struct AVPacket;
struct AVFrame;
struct AVBufferRef;

/**
 * NVIDIA NVDEC 硬件解码器
 * 使用 FFmpeg + CUDA/cuvid 进行硬件解码
 * 适用于 NVIDIA GPU 平台
 */
class NvdecDecoder : public IDecoderBackend {
 public:
  NvdecDecoder() = default;
  ~NvdecDecoder() override;

  NvdecDecoder(const NvdecDecoder&) = delete;
  NvdecDecoder& operator=(const NvdecDecoder&) = delete;

  void open(VideoCodec codec) override;
  std::optional<DecodedFrame> decode(const EncodedPacket& packet) override;
  std::string name() const override { return "NVIDIA NVDEC"; }

  /** 设置 GPU 设备 ID */
  void setGpuId(int gpu_id) { gpu_id_ = gpu_id; }

 private:
  int toCudaCodec(VideoCodec codec) const;
  void close();
  void submitPacket(const EncodedPacket& packet);
  std::optional<DecodedFrame> receiveFrame();
  static int toAVCodec(VideoCodec codec);

  AVCodecContext* codec_ctx_ = nullptr;
  AVBufferRef* hw_device_ctx_ = nullptr;
  int gpu_id_ = 0;
  int width_ = 0;
  int height_ = 0;
  bool eos_sent_ = false;
};
