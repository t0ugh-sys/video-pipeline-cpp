#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DEFAULT="${ROOT_DIR}/build-rockchip/video_pipeline"
INPUT_DEFAULT="/edge/workspace/2_h264_clean.mp4"
MODEL_DEFAULT="/edge/workspace/rk-video-pipeline-cpp/models/stall_int8.rknn"
OUTPUT_H264_DEFAULT="/edge/workspace/vis_modelzoo_full.h264"
OUTPUT_MP4_DEFAULT="/edge/workspace/vis_modelzoo_full.mp4"
OUTPUT_OVERLAY_MODE="${OUTPUT_OVERLAY_MODE:-cpu}"

BIN_PATH="${1:-$BIN_DEFAULT}"
INPUT_PATH="${2:-$INPUT_DEFAULT}"
MODEL_PATH="${3:-$MODEL_DEFAULT}"
OUTPUT_H264_PATH="${4:-$OUTPUT_H264_DEFAULT}"
OUTPUT_MP4_PATH="${5:-$OUTPUT_MP4_DEFAULT}"

echo "bin=${BIN_PATH}"
echo "input=${INPUT_PATH}"
echo "model=${MODEL_PATH}"
echo "output_h264=${OUTPUT_H264_PATH}"
echo "output_mp4=${OUTPUT_MP4_PATH}"
echo "output_overlay=${OUTPUT_OVERLAY_MODE}"

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "error: video_pipeline binary not found or not executable: ${BIN_PATH}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "error: input video not found: ${INPUT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "error: model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

rm -f "${OUTPUT_H264_PATH}" "${OUTPUT_MP4_PATH}"

"${BIN_PATH}" \
  --backend rockchip \
  --infer-workers 2 \
  --rknn-zero-copy false \
  --progress-every 300 \
  --encoder-fps 30 \
  --encoder-bitrate 20000000 \
  --output-overlay "${OUTPUT_OVERLAY_MODE}" \
  --output-video "${OUTPUT_H264_PATH}" \
  "${INPUT_PATH}" \
  "${MODEL_PATH}" \
  640 640

if [[ ! -s "${OUTPUT_H264_PATH}" ]]; then
  echo "error: output bitstream was not generated: ${OUTPUT_H264_PATH}" >&2
  exit 1
fi

ffmpeg -y -framerate 30 -i "${OUTPUT_H264_PATH}" -c copy "${OUTPUT_MP4_PATH}"

if [[ ! -s "${OUTPUT_MP4_PATH}" ]]; then
  echo "error: remuxed mp4 was not generated: ${OUTPUT_MP4_PATH}" >&2
  exit 1
fi

ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 "${OUTPUT_MP4_PATH}"

echo "rockchip output regression completed"
