#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DEFAULT="${ROOT_DIR}/build-rockchip/video_pipeline"
MODEL_DEFAULT="/edge/workspace/rk-video-pipeline-cpp/models/stall_int8.rknn"
INPUT_RTSP_DEFAULT="${INPUT_RTSP_URL:-}"
OUTPUT_RTSP_DEFAULT="${OUTPUT_RTSP_URL:-}"
INPUT_WIDTH_DEFAULT="${INPUT_WIDTH:-640}"
INPUT_HEIGHT_DEFAULT="${INPUT_HEIGHT:-640}"
PROGRESS_EVERY_DEFAULT="${PROGRESS_EVERY:-120}"
MAX_FRAMES_DEFAULT="${MAX_FRAMES:-180}"
FFPROBE_TIMEOUT_SEC="${FFPROBE_TIMEOUT_SEC:-15}"

BIN_PATH="${1:-$BIN_DEFAULT}"
INPUT_RTSP_URL="${2:-$INPUT_RTSP_DEFAULT}"
MODEL_PATH="${3:-$MODEL_DEFAULT}"
OUTPUT_RTSP_URL="${4:-$OUTPUT_RTSP_DEFAULT}"
INPUT_WIDTH="${5:-$INPUT_WIDTH_DEFAULT}"
INPUT_HEIGHT="${6:-$INPUT_HEIGHT_DEFAULT}"
PROGRESS_EVERY="${7:-$PROGRESS_EVERY_DEFAULT}"
MAX_FRAMES="${8:-$MAX_FRAMES_DEFAULT}"

echo "bin=${BIN_PATH}"
echo "input_rtsp=${INPUT_RTSP_URL}"
echo "model=${MODEL_PATH}"
echo "output_rtsp=${OUTPUT_RTSP_URL}"
echo "input_size=${INPUT_WIDTH}x${INPUT_HEIGHT}"
echo "progress_every=${PROGRESS_EVERY}"
echo "max_frames=${MAX_FRAMES}"
echo "ffprobe_timeout_sec=${FFPROBE_TIMEOUT_SEC}"

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "error: video_pipeline binary not found or not executable: ${BIN_PATH}" >&2
  exit 1
fi

if [[ -z "${INPUT_RTSP_URL}" ]]; then
  echo "error: input RTSP URL is required as arg2 or INPUT_RTSP_URL" >&2
  exit 1
fi

if [[ -z "${OUTPUT_RTSP_URL}" ]]; then
  echo "error: output RTSP URL is required as arg4 or OUTPUT_RTSP_URL" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "error: model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "error: ffprobe is required but was not found in PATH" >&2
  exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "error: timeout is required but was not found in PATH" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${PIPELINE_PID:-}" ]]; then
    kill "${PIPELINE_PID}" >/dev/null 2>&1 || true
    wait "${PIPELINE_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"${BIN_PATH}" \
  --backend rockchip \
  --infer-workers 2 \
  --rknn-zero-copy false \
  --progress-every "${PROGRESS_EVERY}" \
  --max-frames "${MAX_FRAMES}" \
  --encoder-fps 30 \
  --encoder-bitrate 20000000 \
  --output-rtsp "${OUTPUT_RTSP_URL}" \
  "${INPUT_RTSP_URL}" \
  "${MODEL_PATH}" \
  "${INPUT_WIDTH}" "${INPUT_HEIGHT}" &
PIPELINE_PID=$!

sleep 3

if ! kill -0 "${PIPELINE_PID}" >/dev/null 2>&1; then
  echo "error: video_pipeline exited before RTSP probe" >&2
  wait "${PIPELINE_PID}"
  exit 1
fi

timeout "${FFPROBE_TIMEOUT_SEC}" \
  ffprobe -v error -rtsp_transport tcp \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 "${OUTPUT_RTSP_URL}"

wait "${PIPELINE_PID}"
PIPELINE_PID=""

echo "rockchip RTSP regression completed"
