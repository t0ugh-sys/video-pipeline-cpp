#pragma once

#include "decoder_interface.hpp"
#include "pipeline_types.hpp"

#include <optional>

// MPP forward declarations
typedef struct MppCtxImpl* MppCtx;
typedef struct MppApi_t MppApi;

/**
 * Rockchip MPP 硬件解码器
 * 适用于 RK3588/RK3568 等 Rockchip 平台
 */
class MppDecoder : public IDecoderBackend {
 public:
  MppDecoder() = default;
  ~MppDecoder() override;

  MppDecoder(const MppDecoder&) = delete;
  MppDecoder& operator=(const MppDecoder&) = delete;

  void open(VideoCodec codec) override;
  std::optional<DecodedFrame> decode(const EncodedPacket& packet) override;
  std::string name() const override { return "Rockchip MPP"; }

 private:
  int toMppCodec(VideoCodec codec) const;
  void close();
  void submitPacket(const EncodedPacket& packet);
  std::optional<DecodedFrame> receiveFrame();

  MppCtx context_ = nullptr;
  MppApi* api_ = nullptr;
};
