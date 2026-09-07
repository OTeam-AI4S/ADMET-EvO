#!/usr/bin/env python3
"""Apply two tightly scoped edits to the supplied TDC comparison PNG.

1. Remove the metric name and direction arrow after every task label.
2. Right-align the remaining task labels beside the rank panel.
3. Swap the original ADMET-EvO teal and MapLight+GNN coral colors.

All rank-marker colors and all other pixels are retained from the source image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "tdc_admetevo_maplight_comparison_final.png"

# Exact solid colors sampled from the source artwork.
TEAL = np.array([24, 137, 119], dtype=np.float32)
CORAL = np.array([207, 101, 89], dtype=np.float32)

# Row centers and the x coordinate where each metric suffix starts.
# These coordinates apply to the supplied 1118 x 932 source PNG.
METRIC_SUFFIXES = [
    (80, 184),
    (119, 184),
    (158, 184),
    (198, 185),
    (237, 185),
    (276, 185),
    (316, 186),
    (355, 184),
    (394, 186),
    (433, 205),
    (473, 170),
    (512, 170),
    (551, 184),
    (590, 184),
    (630, 170),
    (669, 206),
    (708, 205),
    (747, 206),
    (787, 184),
    (826, 205),
    (865, 170),
    (905, 184),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized HSV channels in [0, 1] for an RGB float array."""
    scaled = rgb / 255.0
    maximum = scaled.max(axis=2)
    minimum = scaled.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    hue = np.zeros_like(maximum)
    nonzero = delta > 0

    red = nonzero & (maximum == scaled[:, :, 0])
    green = nonzero & (maximum == scaled[:, :, 1])
    blue = nonzero & (maximum == scaled[:, :, 2])
    hue[red] = ((scaled[:, :, 1][red] - scaled[:, :, 2][red]) / delta[red]) % 6
    hue[green] = (scaled[:, :, 2][green] - scaled[:, :, 0][green]) / delta[green] + 2
    hue[blue] = (scaled[:, :, 0][blue] - scaled[:, :, 1][blue]) / delta[blue] + 4
    hue /= 6.0
    return hue, saturation, maximum


def recolor_antialiased_pixels(
    array: np.ndarray,
    mask: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    """Swap a solid color and its white-antialiased edge shades."""
    pixels = array[mask].astype(np.float32)
    denominator = 255.0 - source
    alpha = np.median((255.0 - pixels) / denominator, axis=1)
    alpha = np.clip(alpha, 0.0, 1.0)[:, None]
    replacement = (1.0 - alpha) * 255.0 + alpha * target
    array[mask] = np.rint(replacement).clip(0, 255).astype(np.uint8)


def right_align_task_labels(image: Image.Image, right_edge: int = 246) -> None:
    """Move the existing rasterized labels without changing their typography."""
    for row_y, _ in METRIC_SUFFIXES:
        row_box = (0, row_y - 10, 253, row_y + 11)
        row = np.asarray(image.crop(row_box))
        ink = np.any(row < 254, axis=2)
        yy, xx = np.where(ink)
        if len(xx) == 0:
            raise ValueError(f"No task-label pixels found at row y={row_y}")

        left = int(xx.min())
        right = int(xx.max()) + 1
        top = int(yy.min())
        bottom = int(yy.max()) + 1
        label = image.crop((left, row_y - 10 + top, right, row_y - 10 + bottom))

        ImageDraw.Draw(image).rectangle(row_box, fill=(255, 255, 255))
        destination_x = right_edge - label.width + 1
        image.paste(label, (destination_x, row_y - 10 + top))


def edit_figure(source_path: Path, output_path: Path) -> None:
    image = Image.open(source_path).convert("RGB")
    if image.size != (1118, 932):
        raise ValueError(f"Expected source size 1118 x 932, found {image.size}")

    array = np.asarray(image).copy()
    hue, saturation, value = rgb_to_hsv(array.astype(np.float32))
    yy, xx = np.indices(hue.shape)

    # Restrict recoloring to the comparison bars and the two method headers.
    method_regions = (
        ((xx >= 530) & (xx <= 918) & (yy >= 45) & (yy <= 925))
        | ((xx >= 940) & (xx <= 1110) & (yy >= 5) & (yy <= 45))
    )
    teal_mask = method_regions & (hue >= 0.43) & (hue <= 0.52) & (saturation >= 0.08) & (value >= 0.25)
    coral_mask = method_regions & ((hue <= 0.04) | (hue >= 0.98)) & (saturation >= 0.08) & (value >= 0.25)

    recolor_antialiased_pixels(array, teal_mask, TEAL, CORAL)
    recolor_antialiased_pixels(array, coral_mask, CORAL, TEAL)

    edited = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(edited)
    for row_y, suffix_x in METRIC_SUFFIXES:
        draw.rectangle((suffix_x, row_y - 10, 252, row_y + 10), fill=(255, 255, 255))
    right_align_task_labels(edited)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited.save(output_path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    edit_figure(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
