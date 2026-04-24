#include "backends/nvdec_decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
}

#include <cstring>
#include <memory>
#include <stdexcept>

namespace {

void checkAvStatus(int status, const char* message) {
  if (status < 0) {
    char err[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(status, err, sizeof(err));
    throw std::runtime_error(std::string(message) + ": " + err);
  }
}

void copyNv12Plane(
    std::vector<std::uint8_t>& destination,
    const std::uint8_t* source,
    int sourceStride,
    int widthInBytes,
    int rows) {
  destination.resize(static_cast<std::size_t>(widthInBytes * rows));
  for (int row = 0; row < rows; ++row) {
    std::memcpy(
        destination.data() + static_cast<std::size_t>(row * widthInBytes),
        source + static_cast<std::size_t>(row * sourceStride),
        static_cast<std::size_t>(widthInBytes));
  }
}

}  // namespace

NvdecDecoder::~NvdecDecoder() {
  close();
}

void NvdecDecoder::open(VideoCodec codec) {
  close();

#ifndef _WIN32
  int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
  checkAvStatus(ret, "Failed to create CUDA hardware device context");
#endif

#ifdef _WIN32
  const AVCodec* avCodec = avcodec_find_decoder_by_name(codec == VideoCodec::kH265 ? "hevc_cuvid" : "h264_cuvid");
#else
  const AVCodec* avCodec = avcodec_find_decoder(toAVCodec(codec));
#endif
  if (!avCodec) {
    throw std::runtime_error("Codec not found");
  }

  codec_ctx_ = avcodec_alloc_context3(avCodec);
  if (!codec_ctx_) {
    throw std::runtime_error("Failed to allocate codec context");
  }

#ifndef _WIN32
  codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  if (!codec_ctx_->hw_device_ctx) {
    throw std::runtime_error("Failed to reference hardware device context");
  }

  if (gpu_id_ >= 0) {
    av_opt_set_int(codec_ctx_->hw_device_ctx->data, "cuda_device", gpu_id_, 0);
  }
#endif

  codec_ctx_->get_format = [](AVCodecContext*, const AVPixelFormat* pixelFormats) {
    for (const AVPixelFormat* current = pixelFormats; *current != AV_PIX_FMT_NONE; ++current) {
#ifdef _WIN32
      if (*current == AV_PIX_FMT_NV12) {
        return *current;
      }
#else
      if (*current == AV_PIX_FMT_CUDA) {
        return *current;
      }
#endif
    }
    return AV_PIX_FMT_NONE;
  };

  const int ret = avcodec_open2(codec_ctx_, avCodec, nullptr);
  checkAvStatus(ret, "Failed to open codec");
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

  AVPacket* avPacket = av_packet_alloc();
  if (!avPacket) {
    throw std::runtime_error("Failed to allocate AVPacket");
  }

  if (!packet.endOfStream) {
    avPacket->data = const_cast<std::uint8_t*>(packet.data.data());
    avPacket->size = static_cast<int>(packet.data.size());
    avPacket->pts = packet.pts;
    avPacket->flags = packet.keyFrame ? AV_PKT_FLAG_KEY : 0;
  } else {
    avPacket->data = nullptr;
    avPacket->size = 0;
    eos_sent_ = true;
  }

  const int ret = avcodec_send_packet(codec_ctx_, avPacket);
  av_packet_free(&avPacket);
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

  DecodedFrame output;
  output.width = frame->width;
  output.height = frame->height;
  output.horizontalStride = frame->linesize[0];
  output.verticalStride = frame->height;
  output.chromaStride = frame->linesize[1] > 0 ? frame->linesize[1] : frame->linesize[0];
  output.format = PixelFormat::kNv12;
  output.nativeFormat = frame->format;
  output.pts = frame->pts;
  output.dmaFd = -1;

  if (frame->format == AV_PIX_FMT_CUDA) {
    if (frame->hw_frames_ctx == nullptr) {
      av_frame_free(&frame);
      throw std::runtime_error("NVDEC returned a CUDA frame without hw_frames_ctx");
    }
    const auto* framesCtx =
        reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
    if (framesCtx == nullptr || framesCtx->sw_format != AV_PIX_FMT_NV12) {
      av_frame_free(&frame);
      throw std::runtime_error("NVDEC returned an unsupported CUDA surface format");
    }
    if (frame->data[0] == nullptr || frame->data[1] == nullptr) {
      av_frame_free(&frame);
      throw std::runtime_error("NVDEC returned an incomplete CUDA NV12 frame");
    }

    AVFrame* retainedFrame = av_frame_clone(frame);
    if (!retainedFrame) {
      av_frame_free(&frame);
      throw std::runtime_error("Failed to clone CUDA frame");
    }
    output.isOnDevice = true;
    output.deviceY = reinterpret_cast<std::uintptr_t>(retainedFrame->data[0]);
    output.deviceUv = reinterpret_cast<std::uintptr_t>(retainedFrame->data[1]);
    output.nativeHandle = std::shared_ptr<void>(
        retainedFrame,
        [](void* handle) {
          AVFrame* ownedFrame = reinterpret_cast<AVFrame*>(handle);
          av_frame_free(&ownedFrame);
        });
  } else if (frame->format == AV_PIX_FMT_NV12) {
    copyNv12Plane(output.yData, frame->data[0], frame->linesize[0], frame->width, frame->height);
    copyNv12Plane(output.uvData, frame->data[1], frame->linesize[1], frame->width, frame->height / 2);
  } else {
    av_frame_free(&frame);
    throw std::runtime_error("NVDEC decoder currently expects CUDA or NV12 output");
  }

  width_ = output.width;
  height_ = output.height;
  av_frame_free(&frame);
  return output;
}
