"""Export raw Limbo V1/V2 safetensors for TensorRT multi-frame inference."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, choices=(280, 378), default=280)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    from src.depth.streaming_export import exportStreamingDepth

    print(
        exportStreamingDepth(
            args.checkpoint, args.output_dir, args.height, 504, not args.fp32
        )
    )


if __name__ == "__main__":
    main()
