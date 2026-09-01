from __future__ import annotations

import argparse
import base64
import io
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[4]
FIGURE_DIR = ROOT / "Main_manuscript" / "figures"
MAP_ASSET_DIR = FIGURE_DIR / "map_assets"
STANDARD_MAP_DIR = Path(
    os.environ.get("MNR_STANDARD_MAP_DIR", ROOT / "restricted_inputs" / "standard_map_2023")
)

SOURCE_PDF = FIGURE_DIR / "Figure1.pdf"
SOURCE_PNG = FIGURE_DIR / "Figure1_microtuned.png"
BASE_PDF = FIGURE_DIR / "Figure1_before_south_china_sea.pdf"
BASE_PNG = FIGURE_DIR / "Figure1_microtuned_before_south_china_sea.png"
OUTPUT_PDF = FIGURE_DIR / "Figure1_south_china_sea.pdf"
OUTPUT_PNG = FIGURE_DIR / "Figure1_south_china_sea.png"
INSET_PNG = MAP_ASSET_DIR / "south_china_sea_inset_clean.png"
STANDALONE_PDF = FIGURE_DIR / "SouthChinaSeaInset.pdf"

# Coordinates use the native PDF page system (points, top-left origin).
INSET_RECT_PT = (31.5, 158.0, 72.8, 215.8)
LABEL_RECT_PT = (29.5, 151.2, 74.8, 157.8)


def extract_embedded_tiff(eps_path: Path, tiff_path: Path) -> Path:
    """Extract the TIFF preview embedded in the official binary EPS."""
    with eps_path.open("rb") as handle:
        header = handle.read(30)
        if len(header) != 30 or header[:4] != bytes.fromhex("c5d0d3c6"):
            raise ValueError(f"Unsupported EPS preview header: {eps_path}")
        _, _, _, _, _, tiff_offset, tiff_length, _ = struct.unpack(
            "<7IH", header
        )
        handle.seek(tiff_offset)
        preview = handle.read(tiff_length)
    tiff_path.parent.mkdir(parents=True, exist_ok=True)
    tiff_path.write_bytes(preview)
    return tiff_path


def build_clean_inset(tiff_path: Path, output_path: Path) -> Path:
    """Retain official coastline/island geometry while suppressing tiny labels."""
    image = Image.open(tiff_path).convert("RGB")
    crop = image.crop((2900, 1740, 3438, 2365))

    rgb = np.asarray(crop, dtype=np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(
        delta,
        maximum,
        out=np.zeros_like(delta),
        where=maximum > 0,
    )

    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-8
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red_max = (maximum == red) & nonzero
    green_max = (maximum == green) & nonzero
    blue_max = (maximum == blue) & nonzero
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4
    hue /= 6.0

    coloured = (saturation > 0.16) & (maximum < 0.995)
    cool = coloured & (hue > 0.43) & (hue < 0.84)
    warm = coloured & (hue >= 0.84)

    cool_mask = Image.fromarray((cool * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(3)
    )
    warm_mask = Image.fromarray((warm * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(3)
    )

    clean = Image.new("RGBA", crop.size, "white")
    clean.paste((96, 123, 120, 255), mask=cool_mask)
    clean.paste((179, 83, 67, 255), mask=warm_mask)

    # Keep a quiet cartographic frame; the label is added as editable text.
    draw = ImageDraw.Draw(clean)
    draw.rectangle(
        (1, 1, clean.width - 2, clean.height - 2),
        outline=(123, 139, 136, 255),
        width=2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(output_path, dpi=(600, 600), optimize=True)
    return output_path


def add_inset_to_pdf(source: Path, inset: Path, output: Path) -> None:
    document = pymupdf.open(source)
    page = document[0]
    inset_rect = pymupdf.Rect(*INSET_RECT_PT)
    page.draw_rect(
        inset_rect,
        color=(0.48, 0.55, 0.54),
        fill=(1, 1, 1),
        width=0.35,
        overlay=True,
    )
    page.insert_image(inset_rect, filename=str(inset), keep_proportion=False, overlay=True)
    label = "South China Sea"
    font_size = 4.8
    label_width = pymupdf.get_text_length(label, fontname="helv", fontsize=font_size)
    label_center = (LABEL_RECT_PT[0] + LABEL_RECT_PT[2]) / 2
    page.insert_text(
        pymupdf.Point(label_center - label_width / 2, 157.0),
        label,
        fontname="helv",
        fontsize=font_size,
        color=(0.16, 0.21, 0.21),
        overlay=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output, garbage=4, deflate=True)
    document.close()


def export_standalone_pdf(inset: Path, output: Path) -> None:
    width, height = 45.3, 66.5
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    label = "South China Sea"
    font_size = 4.8
    label_width = pymupdf.get_text_length(label, fontname="helv", fontsize=font_size)
    page.insert_text(
        pymupdf.Point((width - label_width) / 2, 6.0),
        label,
        fontname="helv",
        fontsize=font_size,
        color=(0.16, 0.21, 0.21),
    )
    inset_rect = pymupdf.Rect(2.0, 8.0, width - 2.0, height - 1.0)
    page.draw_rect(
        inset_rect,
        color=(0.48, 0.55, 0.54),
        fill=(1, 1, 1),
        width=0.35,
    )
    page.insert_image(inset_rect, filename=str(inset), keep_proportion=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output, garbage=4, deflate=True)
    document.close()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    arial = Path(r"C:\Windows\Fonts\arial.ttf")
    if arial.exists():
        return ImageFont.truetype(str(arial), size=size)
    return ImageFont.load_default()


def add_inset_to_png(
    source: Path, reference_pdf: Path, inset: Path, output: Path
) -> None:
    figure = Image.open(source).convert("RGBA")
    inset_image = Image.open(inset).convert("RGBA")
    reference = pymupdf.open(reference_pdf)
    page_rect = reference[0].rect
    scale_x = figure.width / page_rect.width
    scale_y = figure.height / page_rect.height
    reference.close()

    left, top, right, bottom = INSET_RECT_PT
    box = (
        round(left * scale_x),
        round(top * scale_y),
        round(right * scale_x),
        round(bottom * scale_y),
    )
    inset_image = inset_image.resize(
        (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS
    )
    figure.alpha_composite(inset_image, dest=(box[0], box[1]))

    draw = ImageDraw.Draw(figure)
    label_left, label_top, label_right, label_bottom = LABEL_RECT_PT
    label_box = (
        round(label_left * scale_x),
        round(label_top * scale_y),
        round(label_right * scale_x),
        round(label_bottom * scale_y),
    )
    font = _font(max(11, round(4.8 * scale_y)))
    label = "South China Sea"
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = (label_box[0] + label_box[2] - text_width) / 2
    y = (label_box[1] + label_box[3] - text_height) / 2 - bounds[1]
    draw.text((x, y), label, font=font, fill=(41, 54, 54, 255))

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output, dpi=(600, 600), optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace the active Figure 1 PDF/PNG after writing versioned outputs.",
    )
    args = parser.parse_args()

    preview_tiff = MAP_ASSET_DIR / "china_standard_map_2023_preview.tif"
    if preview_tiff.exists():
        build_clean_inset(preview_tiff, INSET_PNG)
    elif not INSET_PNG.exists():
        eps_files = sorted(STANDARD_MAP_DIR.glob("*.eps"))
        if not eps_files:
            raise FileNotFoundError(
                "Neither the packaged inset derivative nor its preview source is "
                "available. Set MNR_STANDARD_MAP_DIR to the author-held directory "
                "containing the GS(2023)2767 EPS source."
            )
        extract_embedded_tiff(eps_files[0], preview_tiff)
        build_clean_inset(preview_tiff, INSET_PNG)
    source_pdf = BASE_PDF if BASE_PDF.exists() else SOURCE_PDF
    source_png = BASE_PNG if BASE_PNG.exists() else SOURCE_PNG
    add_inset_to_pdf(source_pdf, INSET_PNG, OUTPUT_PDF)
    add_inset_to_png(source_png, source_pdf, INSET_PNG, OUTPUT_PNG)
    export_standalone_pdf(INSET_PNG, STANDALONE_PDF)

    if args.apply:
        if not BASE_PDF.exists():
            shutil.copy2(SOURCE_PDF, BASE_PDF)
        if not BASE_PNG.exists():
            shutil.copy2(SOURCE_PNG, BASE_PNG)
        shutil.copy2(OUTPUT_PDF, SOURCE_PDF)
        shutil.copy2(OUTPUT_PNG, SOURCE_PNG)

    print(OUTPUT_PDF)
    print(OUTPUT_PNG)
    print(STANDALONE_PDF)


if __name__ == "__main__":
    main()
