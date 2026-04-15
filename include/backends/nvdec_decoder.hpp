#pragma once

#include "decoder_interface.hpp"
#include "pipeline_types.hpp"

#include <optional>
#include <vector>
#include <cstdint>

struct AVCodecContext;
struct AVFrame;
struct AVBufferRef;

class NvdecDecoder : public IDecoderBackend {
 public:
  NvdecDecoder() = default;
  ~NvdecDecoder() override;

  NvdecDecoder(const NvdecDecoder&) = delete;
  NvdecDecoder& operator=(const NvdecDecoder&) = delete;

  void open(VideoCodec codec) override;
  void submitPacket(const EncodedPacket& packet) override;
  std::optional<DecodedFrame> receiveFrame() override;
  std::string name() const override { return "NVIDIA NVDEC"; }

  void setGpuId(int gpu_id) { gpu_id_ = gpu_id; }

 private:
  void close();
  static int toAVCodec(VideoCodec codec);

  AVCodecContext* codec_ctx_ = nullptr;
  AVBufferRef* hw_device_ctx_ = nullptr;
  int gpu_id_ = 0;
  int width_ = 0;
  int height_ = 0;
  bool eos_sent_ = false;
};
