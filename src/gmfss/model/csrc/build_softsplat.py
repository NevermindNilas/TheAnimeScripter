"""One-shot build script for the softsplat CUDA extension.

Compiles `softsplat_kernel.cu` with torch.utils.cpp_extension and copies the
resulting binary into `src/gmfss/model/` next to `softsplat.py`. Run once on a
build machine; the binary is committed to the repo so end users do not need a
CUDA toolchain.

Usage:
    python src/gmfss/model/csrc/build_softsplat.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# CUDA 13.x dropped support for sm < 75. Min target: Turing (RTX 20 / GTX 16).
# Bake SASS for Turing -> Blackwell, PTX forward-compat for future archs.
os.environ.setdefault(
    "TORCH_CUDA_ARCH_LIST",
    "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX",
)

from torch.utils.cpp_extension import load  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
BUILD_DIR = HERE / "build"
BUILD_DIR.mkdir(exist_ok=True)

EXT_NAME = "_gmfss_softsplat_ext"

print(f"[build] sources: {HERE / 'softsplat_kernel.cu'}")
print(f"[build] build_directory: {BUILD_DIR}")
print(f"[build] arch list: {os.environ['TORCH_CUDA_ARCH_LIST']}")

ext = load(
    name=EXT_NAME,
    sources=[str(HERE / "softsplat_kernel.cu")],
    build_directory=str(BUILD_DIR),
    extra_cflags=["/Zc:preprocessor"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-Xcompiler",
        "/Zc:preprocessor",
    ],
    verbose=True,
)

# Locate the produced binary (.pyd on Windows, .so on Linux).
suffix = ".pyd" if sys.platform == "win32" else ".so"
candidates = list(BUILD_DIR.glob(f"{EXT_NAME}*{suffix}"))
if not candidates:
    raise SystemExit(f"[build] no {suffix} produced under {BUILD_DIR}")
src_bin = candidates[0]
dst_bin = MODEL_DIR / src_bin.name
shutil.copy2(src_bin, dst_bin)
print(f"[build] copied {src_bin.name} -> {dst_bin}")

# Smoke test: round-trip a tiny tensor through the extension.
import torch  # noqa: E402

x = torch.randn(1, 3, 4, 4, device="cuda", dtype=torch.float16)
f = torch.zeros(1, 2, 4, 4, device="cuda", dtype=torch.float16)
y = ext.forward(x, f)
assert y.shape == x.shape and y.dtype == x.dtype
print(f"[build] smoke ok: out shape={tuple(y.shape)} dtype={y.dtype}")
print("[build] done")
