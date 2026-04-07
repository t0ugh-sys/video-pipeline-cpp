#include "backends/mpp_decoder.hpp"

extern "C" {
#include <rk_mpi.h>
#include <mpp_buffer.h>
#include <mpp_err.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
}

#include <stdexcept>

namespace {

constexpr MppCodingType kCodingAvc = MPP_VIDEO_CodingAVC;
constexpr MppCodingType kCodingHevc = MPP_VIDEO_CodingHEVC;

void checkMppStatus(MPP_RET status, const char* message) {
  if (status != MPP_OK) {
    throw std::runtime_error(message);
  }
}

}  // namespace

MppDecoder::~MppDecoder() {
  close();
}

void MppDecoder::open(VideoCodec codec) {
  close();

  checkMppStatus(mpp_create(&context_, &api_), "mpp_create failed");
  checkMppStatus(
      mpp_init(context_, MPP_CTX_DEC, static_cast<MppCodingType>(toMppCodec(codec))),
      "mpp_init failed");

  // 让 MPP 按 Annex-B 方式切包，处理 H.264/H.265 码流时更稳一些。
  RK_U32 splitMode = 1;
  checkMppStatus(
      api_->control(context_, MPP_DEC_SET_PARSER_SPLIT_MODE, &splitMode),
      "MPP_DEC_SET_PARSER_SPLIT_MODE failed");
}

std::optional<DecodedFrame> MppDecoder::decode(const EncodedPacket& packet) {
  submitPacket(packet);
  return receiveFrame();
}

int MppDecoder::toMppCodec(VideoCodec codec) const {
  switch (codec) {
    case VideoCodec::kH264:
      return kCodingAvc;
    case VideoCodec::kH265:
      return kCodingHevc;
    default:
      throw std::runtime_error("Unsupported codec for MPP");
  }
}

void MppDecoder::close() {
  if (context_ != nullptr) {
    mpp_destroy(context_);
  }
  context_ = nullptr;
  api_ = nullptr;
}

void MppDecoder::submitPacket(const EncodedPacket& packet) {
  MppPacket mppPacket = nullptr;
  checkMppStatus(
      mpp_packet_init(&mppPacket,
                      packet.endOfStream ? nullptr : const_cast<std::uint8_t*>(packet.data.data()),
                      packet.endOfStream ? 0 : packet.data.size()),
      "mpp_packet_init failed");

  if (packet.endOfStream) {
    mpp_packet_set_eos(mppPacket);
  }

  // 时间戳往后传，便于后续做日志和结果对齐。
  mpp_packet_set_pts(mppPacket, packet.pts);
  const MPP_RET status = api_->decode_put_packet(context_, mppPacket);
  mpp_packet_deinit(&mppPacket);
  checkMppStatus(status, "decode_put_packet failed");
}

std::optional<DecodedFrame> MppDecoder::receiveFrame() {
  MppFrame frame = nullptr;
  const MPP_RET status = api_->decode_get_frame(context_, &frame);
  checkMppStatus(status, "decode_get_frame failed");

  if (frame == nullptr) {
    return std::nullopt;
  }

  if (mpp_frame_get_errinfo(frame) != 0) {
    mpp_frame_deinit(&frame);
    throw std::runtime_error("MPP decoder returned frame with error info");
  }

  DecodedFrame output;
  output.width = mpp_frame_get_width(frame);
  output.height = mpp_frame_get_height(frame);
  output.horizontalStride = mpp_frame_get_hor_stride(frame);
  output.verticalStride = mpp_frame_get_ver_stride(frame);
  output.pts = mpp_frame_get_pts(frame);

  MppBuffer buffer = mpp_frame_get_buffer(frame);
  if (buffer == nullptr) {
    mpp_frame_deinit(&frame);
    throw std::runtime_error("MPP frame buffer is null");
  }

  // RGA 后续直接消费 dma fd，避免先拷回 CPU 普通内存。
  output.dmaFd = mpp_buffer_get_fd(buffer);
  mpp_frame_deinit(&frame);
  return output;
}
