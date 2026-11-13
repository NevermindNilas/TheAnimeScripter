#!/usr/bin/env bash
# Smoke-test a TAS Linux portable archive the way a user receives it: extract,
# --help, install the CPU profile, run one real dedup encode. Needs no GPU.
# The workflows call this file, so running it locally runs the same checks.
#
# Usage: smoke-linux-bundle.sh <archive.tar.gz> <work-dir>
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <archive.tar.gz> <work-dir>" >&2
    exit 2
fi

archive=$(realpath "$1")
work=$2
bundle="$work/bundle"

export CUDA_VISIBLE_DEVICES=

mkdir -p "$bundle"
tar -xzf "$archive" -C "$bundle"
cd "$bundle"
bundle=$(pwd)

echo "== Executable bits survived the archive"
test -x run.sh
test -x bin/python3

echo "== Base bundle --help"
./run.sh --help > "$work/help.txt"
grep -q -- "--download_requirements" "$work/help.txt"

echo "== Install the linux-lite profile"
df -h "$bundle"
./run.sh --download_requirements linux-lite
df -h "$bundle"
du -sh "$bundle"

echo "== Generate a clip where every frame is doubled"
input="$work/smoke-input.mp4"
output="$work/smoke-output.mp4"
bin/python3 -I - "$input" <<'PY'
import sys

import cv2
import numpy as np

height, width = 180, 320
writer = cv2.VideoWriter(
    sys.argv[1], cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
)
assert writer.isOpened(), "cv2 could not open an mp4v writer"
rng = np.random.default_rng(0)
for _ in range(30):
    frame = cv2.resize(
        rng.integers(0, 256, (9, 16, 3), dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    writer.write(frame)
    writer.write(frame)
writer.release()
PY

echo "== Dedup encode"
./run.sh --input "$input" --output "$output" --dedup --dedup_method ssim

echo "== Check the output"
bin/python3 -I - "$input" "$output" <<'PY'
import sys

import cv2


def frameCount(path):
    capture = cv2.VideoCapture(path)
    assert capture.isOpened(), f"cv2 cannot open {path}"
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


source, written = frameCount(sys.argv[1]), frameCount(sys.argv[2])
print(f"source frames: {source}, written frames: {written}")
assert source == 60, "the input clip came out the wrong length"
assert 20 <= written <= 40, "dedup did not drop the doubled frames"
PY

echo "== NCNN upscale (Vulkan if the machine has it, else NCNN's CPU path)"
upscaled="$work/smoke-upscaled.mp4"
./run.sh --input "$output" --output "$upscaled" --upscale span-ncnn

bin/python3 -I - "$output" "$upscaled" <<'PY'
import sys

import cv2

source, written = cv2.VideoCapture(sys.argv[1]), cv2.VideoCapture(sys.argv[2])
assert written.isOpened(), "cv2 cannot open the upscaled output"
ok, frame = written.read()
assert ok, "the upscaled output has no frames"
assert frame.shape[:2] == (360, 640), f"wrong size: {frame.shape}"
assert frame.mean() > 5, "the upscaled output is black"
assert written.get(cv2.CAP_PROP_FRAME_COUNT) == source.get(cv2.CAP_PROP_FRAME_COUNT)
PY

echo "== Smoke test passed"
