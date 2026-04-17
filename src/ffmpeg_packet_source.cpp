#include "ffmpeg_packet_source.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
}

#include <cstring>
#include <stdexcept>

namespace {

[[noreturn]] void throwFfmpegError(const std::string& message, int errorCode) {
  char errorBuffer[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(errorCode, errorBuffer, sizeof(errorBuffer));
  throw std::runtime_error(message + ": " + errorBuffer);
}

bool containerNameContains(const AVInputFormat* format, const char* token) {
  return format != nullptr && format->name != nullptr && std::strstr(format->name, token) != nullptr;
}

}  // namespace

FFmpegPacketSource::~FFmpegPacketSource() {
  close();
}

void FFmpegPacketSource::open(const InputSourceConfig& config) {
  close();
  avformat_network_init();

  int result = avformat_open_input(&formatContext_, config.uri.c_str(), nullptr, nullptr);
  if (result < 0) {
    throwFfmpegError("Failed to open input", result);
  }

  result = avformat_find_stream_info(formatContext_, nullptr);
  if (result < 0) {
    throwFfmpegError("Failed to find stream info", result);
  }

  videoStreamIndex_ = av_find_best_stream(formatContext_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (videoStreamIndex_ < 0) {
    throw std::runtime_error("Failed to find video stream");
  }

  const AVStream* videoStream = formatContext_->streams[videoStreamIndex_];
  codec_ = toVideoCodec(videoStream->codecpar->codec_id);
  if (codec_ == VideoCodec::kUnknown) {
    throw std::runtime_error("Unsupported video codec for hardware decoder");
  }

  initBitstreamFilter();
}

EncodedPacket FFmpegPacketSource::readPacket() {
  EncodedPacket filteredOutput;
  if (tryReceiveFilteredPacket(filteredOutput)) {
    return filteredOutput;
  }

  AVPacket packet{};

  while (true) {
    const int result = av_read_frame(formatContext_, &packet);
    if (result == AVERROR_EOF) {
      if (bsfContext_ != nullptr && !bsfFlushed_) {
        const int flushResult = av_bsf_send_packet(bsfContext_, nullptr);
        if (flushResult < 0) {
          throwFfmpegError("Failed to flush bitstream filter", flushResult);
        }
        bsfFlushed_ = true;
        if (tryReceiveFilteredPacket(filteredOutput)) {
          return filteredOutput;
        }
      }
      EncodedPacket eosPacket;
      eosPacket.endOfStream = true;
      return eosPacket;
    }
    if (result < 0) {
      throwFfmpegError("Failed to read frame", result);
    }

    if (packet.stream_index != videoStreamIndex_) {
      av_packet_unref(&packet);
      continue;
    }

    if ((packet.flags & AV_PKT_FLAG_CORRUPT) != 0 || packet.size <= 0 || packet.data == nullptr) {
      av_packet_unref(&packet);
      continue;
    }

    if (bsfContext_ != nullptr) {
      int filterResult = av_bsf_send_packet(bsfContext_, &packet);
      av_packet_unref(&packet);
      if (filterResult < 0) {
        throwFfmpegError("Failed to send packet to bitstream filter", filterResult);
      }
      bsfFlushed_ = false;

      if (tryReceiveFilteredPacket(filteredOutput)) {
        return filteredOutput;
      }
      continue;
    }

    EncodedPacket output = copyPacket(&packet);
    av_packet_unref(&packet);
    return output;
  }
}

VideoCodec FFmpegPacketSource::codec() const {
  return codec_;
}

VideoCodec FFmpegPacketSource::toVideoCodec(int codecId) {
  switch (codecId) {
    case AV_CODEC_ID_H264:
      return VideoCodec::kH264;
    case AV_CODEC_ID_HEVC:
      return VideoCodec::kH265;
    default:
      return VideoCodec::kUnknown;
  }
}

void FFmpegPacketSource::close() {
  if (bsfContext_ != nullptr) {
    av_bsf_free(&bsfContext_);
  }
  if (formatContext_ != nullptr) {
    avformat_close_input(&formatContext_);
  }
  videoStreamIndex_ = -1;
  codec_ = VideoCodec::kUnknown;
  bsfFlushed_ = false;
}

bool FFmpegPacketSource::needsAnnexBFilter() const {
  if (formatContext_ == nullptr || videoStreamIndex_ < 0) {
    return false;
  }
  if (codec_ != VideoCodec::kH264 && codec_ != VideoCodec::kH265) {
    return false;
  }
  return containerNameContains(formatContext_->iformat, "mp4") ||
         containerNameContains(formatContext_->iformat, "mov");
}

void FFmpegPacketSource::initBitstreamFilter() {
  if (!needsAnnexBFilter()) {
    return;
  }

  const char* filterName = codec_ == VideoCodec::kH264 ? "h264_mp4toannexb" : "hevc_mp4toannexb";
  const AVBitStreamFilter* filter = av_bsf_get_by_name(filterName);
  if (filter == nullptr) {
    throw std::runtime_error(std::string("Failed to find FFmpeg bitstream filter: ") + filterName);
  }

  int result = av_bsf_alloc(filter, &bsfContext_);
  if (result < 0) {
    throwFfmpegError("Failed to allocate bitstream filter", result);
  }

  AVStream* stream = formatContext_->streams[videoStreamIndex_];
  result = avcodec_parameters_copy(bsfContext_->par_in, stream->codecpar);
  if (result < 0) {
    throwFfmpegError("Failed to copy codec parameters to bitstream filter", result);
  }
  bsfContext_->time_base_in = stream->time_base;

  result = av_bsf_init(bsfContext_);
  if (result < 0) {
    throwFfmpegError("Failed to initialize bitstream filter", result);
  }
}

EncodedPacket FFmpegPacketSource::copyPacket(const void* packetPtr) const {
  const auto* packet = static_cast<const AVPacket*>(packetPtr);
  EncodedPacket output;
  output.data.assign(packet->data, packet->data + packet->size);
  output.pts = packet->pts;
  output.keyFrame = (packet->flags & AV_PKT_FLAG_KEY) != 0;
  return output;
}

bool FFmpegPacketSource::tryReceiveFilteredPacket(EncodedPacket& output) {
  if (bsfContext_ == nullptr) {
    return false;
  }

  while (true) {
    AVPacket filteredPacket{};
    const int filterResult = av_bsf_receive_packet(bsfContext_, &filteredPacket);
    if (filterResult == AVERROR(EAGAIN) || filterResult == AVERROR_EOF) {
      return false;
    }
    if (filterResult < 0) {
      throwFfmpegError("Failed to receive packet from bitstream filter", filterResult);
    }

    if ((filteredPacket.flags & AV_PKT_FLAG_CORRUPT) != 0 ||
        filteredPacket.size <= 0 ||
        filteredPacket.data == nullptr) {
      av_packet_unref(&filteredPacket);
      continue;
    }

    output = copyPacket(&filteredPacket);
    av_packet_unref(&filteredPacket);
    return true;
  }
}
