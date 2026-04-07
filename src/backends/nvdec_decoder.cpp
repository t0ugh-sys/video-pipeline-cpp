#include "backends/nvdec_decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
#include <libavutil/opt.h>
}

#include <stdexcept>
#include <cstring>

namespace {

void checkAvStatus(int status, const char* message) {
  if (status < 0) {
    char err[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(status, err, sizeof(err));
    throw std::runtime_error(std::string(message) + ": " + err);
  }
}

}  // namespace

NvdecDecoder::~NvdecDecoder() {
  close();
}

void NvdecDecoder::open(VideoCodec codec) {
  close();

  // 1. 创建 CUDA 硬件设备上下文
  int ret = av_hwdevice_ctx_create(
      &hw_device_ctx_,
      AV_HWDEVICE_TYPE_CUDA,
      nullptr,
      nullptr,
      0);
  checkAvStatus(ret, "Failed to create CUDA hardware device context");

  // 2. 查找解码器
  const AVCodec* av_codec = avcodec_find_decoder(toAVCodec(codec));
  if (!av_codec) {
    throw std::runtime_error("Codec not found");
  }

  // 3. 创建编解码器上下文
  codec_ctx_ = avcodec_alloc_context3(av_codec);
  if (!codec_ctx_) {
    throw std::runtime_error("Failed to allocate codec context");
  }

  // 4. 设置硬件加速
  codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  if (!codec_ctx_->hw_device_ctx) {
    throw std::runtime_error("Failed to reference hardware device context");
  }

  // 5. 设置 CUDA 设备 ID
  if (gpu_id_ >= 0) {
    av_opt_set_int(codec_ctx_->hw_device_ctx->data, "cuda_device", gpu_id_, 0);
  }

  // 6. 设置输出格式为 NV12 (NVDEC 原生输出)
  codec_ctx_->get_format = [](AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
      if (*p == AV_PIX_FMT_CUDA) {
        return *p;
      }
    }
    return AV_PIX_FMT_NONE;
  };

  // 7. 打开编解码器
  ret = avcodec_open2(codec_ctx_, av_codec, nullptr);
  checkAvStatus(ret, "Failed to open codec");

  // 8. 获取视频尺寸 (如果有)
  if (codec_ctx_->width > 0 && codec_ctx_->height > 0) {
    width_ = codec_ctx_->width;
    height_ = codec_ctx_->height;
  }
}

std::optional<DecodedFrame> NvdecDecoder::decode(const EncodedPacket& packet) {
  submitPacket(packet);
  return receiveFrame();
}

int NvdecDecoder::toAVCodec(VideoCodec codec) {
  switch (codec) {
    case VideoCodec::kH264:
      return AV_CODEC_ID_H264;
    case VideoCodec::kH265:
      return AV_CODEC_ID_HEVC;
    default:
      throw std::runtime_error("Unsupported codec for NVDEC");
  }
}

void NvdecDecoder::close() {
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
  }
  if (hw_device_ctx_) {
    av_buffer_unref(&hw_device_ctx_);
  }
  eos_sent_ = false;
  width_ = 0;
  height_ = 0;
}

void NvdecDecoder::submitPacket(const EncodedPacket& packet) {
  if (!codec_ctx_) {
    throw std::runtime_error("Decoder not initialized");
  }

  AVPacket* av_packet = av_packet_alloc();
  if (!av_packet) {
    throw std::runtime_error("Failed to allocate AVPacket");
  }

  if (!packet.endOfStream) {
    av_packet->data = const_cast<uint8_t*>(packet.data.data());
    av_packet->size = static_cast<int>(packet.data.size());
    av_packet->pts = packet.pts;
    av_packet->flags = packet.keyFrame ? AV_PKT_FLAG_KEY : 0;
  } else {
    // EOF 包
    av_packet->data = nullptr;
    av_packet->size = 0;
    eos_sent_ = true;
  }

  // 发送包到解码器
  int ret = avcodec_send_packet(codec_ctx_, av_packet);
  av_packet_free(&av_packet);

  if (ret < 0 && ret != AVERROR(EAGAIN)) {
    checkAvStatus(ret, "Failed to send packet to decoder");
  }
}

std::optional<DecodedFrame> NvdecDecoder::receiveFrame() {
  if (!codec_ctx_) {
    throw std::runtime_error("Decoder not initialized");
  }

  AVFrame* frame = av_frame_alloc();
  if (!frame) {
    throw std::runtime_error("Failed to allocate AVFrame");
  }

  int ret = avcodec_receive_frame(codec_ctx_, frame);
  if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
    av_frame_free(&frame);
    return std::nullopt;
  }

  if (ret < 0) {
    av_frame_free(&frame);
    checkAvStatus(ret, "Failed to receive frame from decoder");
  }

  // 从 CUDA 硬件帧获取数据
  DecodedFrame output;
  output.width = frame->width;
  output.height = frame->height;
  output.horizontalStride = frame->linesize[0];
  output.verticalStride = frame->height;
  output.pts = frame->pts;

  // 如果是硬件帧，获取 DMA FD
  if (frame->hw_frames_ctx) {
    AVHWFramesContext* hw_frames_ctx = reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
    AVFrame* hw_frame = frame;

    // 创建软件帧来下载数据
    AVFrame* sw_frame = av_frame_alloc();
    if (sw_frame) {
      sw_frame->format = AV_PIX_FMT_NV12;
      ret = av_hwframe_transfer_data(sw_frame, hw_frame, 0);
      if (ret >= 0) {
        // 成功下载，但我们需要 FD 用于零拷贝
        // 这里简化处理，实际使用时可能需要 CUDA IPC
        output.dmaFd = -1;  // CPU 内存路径
      }
      av_frame_free(&sw_frame);
    }
  } else {
    // 软件帧 (不应该发生，但保留处理)
    output.dmaFd = -1;
  }

  av_frame_free(&frame);
  return output;
}
