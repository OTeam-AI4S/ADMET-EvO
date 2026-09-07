#!/usr/bin/env python3
"""Increase typography in eight supplied figure panels without restyling them.

The original raster panels are retained as the visual base. Only the existing
text regions are cleared and redrawn in Microsoft YaHei. This keeps the data
geometry, colors, fills, strokes, axes, and panel dimensions unchanged while
making the labels easier to read. PDF and SVG exports retain the replacement
text as editable text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "figures" / "text_enlarged" / "source"
OUTPUT_DIR = ROOT / "outputs" / "figures" / "text_enlarged"
FONT_REGULAR = ROOT / "assets" / "fonts" / "msyh.ttc"
FONT_BOLD = ROOT / "assets" / "fonts" / "msyhbd.ttc"

TEXT = "#405361"
MUTED = "#657483"
CORAL = "#CF6559"
RADAR_CORAL = "#DE674E"
TEAL = "#188977"


@dataclass(frozen=True)
class Label:
    x: float
    y: float
    text: str
    size: float
    color: str = TEXT
    bold: bool = False
    ha: str = "center"
    va: str = "top"
    linespacing: float = 1.18


@dataclass(frozen=True)
class FigureSpec:
    source: str
    output: str
    masks: tuple[tuple[int, int, int, int], ...]
    labels: tuple[Label, ...]


def configure_font() -> None:
    for path in (FONT_REGULAR, FONT_BOLD):
        if not path.is_file():
            raise FileNotFoundError(f"Required Microsoft YaHei font is missing: {path}")
        fm.fontManager.addfont(path)
    resolved = Path(fm.findfont(fm.FontProperties(family="Microsoft YaHei"))).resolve()
    if resolved not in {FONT_REGULAR.resolve(), FONT_BOLD.resolve()}:
        raise RuntimeError(f"Microsoft YaHei resolved unexpectedly: {resolved}")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def radar_spec(index: int, labels: tuple[str, str, str, str, str, str]) -> FigureSpec:
    source = SOURCE_DIR / f"radar_{index}_original.png"
    with Image.open(source) as image:
        width, height = image.size
    top, upper_right, lower_right, bottom, lower_left, upper_left = labels
    return FigureSpec(
        source=source.name,
        output=f"radar_{index}_large_text",
        masks=(
            (78, 0, width - 78, 28),
            (207, 49, width, 94),
            (206, 132, width, 181),
            (85, 190, width - 85, height),
            (0, 132, 67, 181),
            (0, 49, 66, 94),
        ),
        labels=(
            Label(width / 2, 1.5, top.split("\n", 1)[0], 7.8, RADAR_CORAL, True),
            Label(width / 2, 13.5, top.split("\n", 1)[1], 7.2, RADAR_CORAL, True),
            Label(width - 37.0, 57.0, upper_right, 6.8),
            Label(width - 36.5, 143.0, lower_right, 6.8),
            Label(width / 2, 198.0, bottom, 6.8),
            Label(36.5, 143.0, lower_left, 6.8),
            Label(37.0, 57.0, upper_left, 6.8),
        ),
    )


def specs() -> tuple[FigureSpec, ...]:
    radars = (
        radar_spec(
            1,
            (
                "ADMET-EvO\n0.874",
                "ZairaChem\n0.871",
                "ADMETrix\n0.870",
                "MapLight+\nGNN\n0.869",
                "MapLight\n0.868",
                "CFA\n0.852",
            ),
        ),
        radar_spec(
            2,
            (
                "ADMET-EvO\n0.590",
                "CFA\n0.576",
                "MapLight\n0.562",
                "MapLight+\nGNN\n0.557",
                "Euclia ML\n0.547",
                "Voting\nRegressor\n0.544",
            ),
        ),
        radar_spec(
            3,
            (
                "ADMET-EvO\n0.392",
                "MiniMol\n0.456",
                "Chemprop-\nRDKit\n0.467",
                "Chemprop\n0.470",
                "BaseBoosting\n0.479",
                "ADMETrix\n0.512",
            ),
        ),
        radar_spec(
            4,
            (
                "ADMET-EvO\n7.302",
                "Gradient\nBoost\n7.440",
                "MapLight+\nGNN\n7.526",
                "MapLight\n7.660",
                "MiniMol\n7.696",
                "Chemprop\n7.788",
            ),
        ),
        radar_spec(
            5,
            (
                "ADMET-EvO\n0.739",
                "MiniMol\n0.741",
                "Chemprop-\nRDKit\n0.761",
                "DeepMol\nAutoML\n0.775",
                "AttentiveFP\n0.776",
                "MapLight+\nGNN\n0.789",
            ),
        ),
    )

    cost = FigureSpec(
        source="cost_original.png",
        output="cost_large_text",
        masks=(
            (0, 6, 101, 34),
            (0, 67, 101, 96),
            (463, 6, 531, 34),
            (204, 67, 276, 96),
            (278, 63, 462, 96),
            (85, 102, 531, 124),
        ),
        labels=(
            Label(94, 20, "Score-only", 8.8, TEXT, False, "right", "center"),
            Label(94, 82, "Compute-aware", 8.8, TEXT, False, "right", "center"),
            Label(468, 20, "2.247 h", 8.7, TEXT, True, "left", "center"),
            Label(211, 82, "0.625 h", 8.7, CORAL, True, "left", "center"),
            Label(289, 82, "0.278× of score-only", 8.7, TEAL, True, "left", "center"),
            Label(102, 105, "0.0", 7.9, MUTED),
            Label(182, 105, "0.5", 7.9, MUTED),
            Label(262, 105, "1.0", 7.9, MUTED),
            Label(342, 105, "1.5", 7.9, MUTED),
            Label(422, 105, "2.0", 7.9, MUTED),
            Label(502, 105, "2.5", 7.9, MUTED),
        ),
    )

    gain = FigureSpec(
        source="gain_distribution_original.png",
        output="gain_distribution_large_text",
        masks=((0, 113, 379, 132),),
        labels=(
            Label(28, 114, "−5", 8.8, MUTED),
            Label(52, 114, "0", 8.8, MUTED),
            Label(150, 114, "20", 8.8, MUTED),
            Label(249, 114, "40", 8.8, MUTED),
            Label(348, 114, "60", 8.8, MUTED),
        ),
    )

    coverage = FigureSpec(
        source="coverage_original.png",
        output="coverage_large_text",
        masks=(
            (0, 3, 23, 29),
            (0, 48, 23, 73),
            (0, 78, 23, 115),
            (20, 99, 358, 116),
        ),
        labels=(
            Label(19, 15, "100", 7.9, MUTED, False, "right", "center"),
            Label(19, 60, "80", 7.9, MUTED, False, "right", "center"),
            Label(19, 102, "60", 7.9, MUTED, False, "right", "center"),
            Label(28, 100, "25", 7.9, MUTED),
            Label(87, 100, "40", 7.9, MUTED),
            Label(166, 100, "60", 7.9, MUTED),
            Label(245, 100, "80", 7.9, MUTED),
            Label(342, 100, "100", 7.9, MUTED),
        ),
    )
    return (*radars, cost, gain, coverage)


def render(spec: FigureSpec) -> None:
    source_path = SOURCE_DIR / spec.source
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    background = Image.open(source_path).convert("RGBA")
    width, height = background.size
    white = (255, 255, 255, 255)
    draw = ImageDraw.Draw(background)
    for rectangle in spec.masks:
        draw.rectangle(rectangle, fill=white)

    figure = plt.figure(figsize=(width / 100, height / 100), dpi=400, facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(background, interpolation="lanczos", extent=(-0.5, width - 0.5, height - 0.5, -0.5))
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_axis_off()

    for label in spec.labels:
        font_path = FONT_BOLD if label.bold else FONT_REGULAR
        axis.text(
            label.x,
            label.y,
            label.text,
            ha=label.ha,
            va=label.va,
            color=label.color,
            fontproperties=fm.FontProperties(fname=str(font_path), size=label.size),
            linespacing=label.linespacing,
            clip_on=False,
        )

    prefix = OUTPUT_DIR / spec.output
    figure.savefig(prefix.with_suffix(".png"), dpi=400, pad_inches=0)
    figure.savefig(prefix.with_suffix(".pdf"), dpi=300, pad_inches=0)
    figure.savefig(prefix.with_suffix(".svg"), dpi=300, pad_inches=0)
    figure.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def main() -> None:
    configure_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs():
        render(spec)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
