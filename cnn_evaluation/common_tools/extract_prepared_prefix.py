#!/usr/bin/env python3

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract the first N bytes from a prepared images.bin")
    parser.add_argument("--input", required=True, help="input images.bin path")
    parser.add_argument("--output", required=True, help="output prefix file path")
    parser.add_argument("--bytes", type=int, required=True, help="number of bytes to copy")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    num_bytes = int(args.bytes)
    if num_bytes <= 0:
        raise ValueError("--bytes must be positive")

    with input_path.open("rb") as src, output_path.open("wb") as dst:
        remaining = num_bytes
        while remaining > 0:
            chunk = src.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)

    written = output_path.stat().st_size
    print(f"[EXTRACT_PREFIX] input={input_path}")
    print(f"[EXTRACT_PREFIX] output={output_path}")
    print(f"[EXTRACT_PREFIX] requested_bytes={num_bytes} written_bytes={written}")


if __name__ == "__main__":
    main()
