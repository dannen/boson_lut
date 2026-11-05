#!/usr/bin/env python3
"""Makes a colour copy of a grayscale recording from boson_lut.py, using any of its LUTs."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.dont_write_bytecode = True # Importing boson_lut shouldn't leave a __pycache__ beside the scripts
from boson_lut import LUTS, apply_lut, load_custom_luts

# Decoded grayscale video can differ slightly between channels; a coloured picture differs much more.
MAX_GRAY_CHANNEL_DIFFERENCE = 8
FRAMES_TO_CHECK = 5


def looks_grayscale(frame):
    channel_difference = frame.max(axis=2).astype(np.int16) - frame.min(axis=2).astype(np.int16)
    return int(channel_difference.max()) <= MAX_GRAY_CHANNEL_DIFFERENCE


def main():
    load_custom_luts(os.path.dirname(os.path.abspath(__file__)), quiet=True)
    load_custom_luts('.', quiet=True)
    lut_names = sorted(LUTS.keys())

    parser = argparse.ArgumentParser(description="Apply a LUT to a grayscale recording.")
    parser.add_argument('input', help="Grayscale recording.")
    parser.add_argument('lut', choices=lut_names, metavar='LUT', help="LUT to apply. Available: " + ", ".join(lut_names) + ".")
    parser.add_argument('-o', '--output', help="Output file. Default: the input's name plus the LUT's, as .avi.")
    parser.add_argument('--lossless', action='store_true', help="Write FFV1 instead of XVID.")
    parser.add_argument('--overwrite', action='store_true', help="Replace the output file if it exists.")
    parser.add_argument('--force', action='store_true', help="Process a video that doesn't look grayscale.")
    args = parser.parse_args()

    output = args.output or f"{os.path.splitext(args.input)[0]}-{args.lut}.avi"
    if os.path.abspath(output) == os.path.abspath(args.input):
        sys.exit("Error: the output file can't be the input file.")
    if os.path.exists(output) and not args.overwrite:
        sys.exit(f"Error: {output} already exists. Use --overwrite to replace it.")

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        sys.exit(f"Error: could not open {args.input}.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        sys.exit(f"Error: {args.input} doesn't report a frame rate.")
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    writer = None
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frames < FRAMES_TO_CHECK and not args.force and not looks_grayscale(frame):
                sys.exit(f"Error: {args.input} doesn't look grayscale, so a LUT was probably already applied. Use --force to process it anyway.")
            if writer is None:
                # Opened only now, so a rejected input never leaves an empty output file
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*('FFV1' if args.lossless else 'XVID')), fps, size)
                if not writer.isOpened():
                    sys.exit(f"Error: could not create {output}. This OpenCV build may lack the codec.")
            writer.write(apply_lut(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args.lut))
            frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames == 0:
        sys.exit(f"Error: {args.input} has no readable frames.")
    print(f"Wrote {frames} frames to {output}")


if __name__ == "__main__":
    main()
