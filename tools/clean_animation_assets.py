#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "assets" / "generated_raw"
CLEAN_ROOT = ROOT / "assets" / "clean"
SHEETS_ROOT = ROOT / "assets" / "sheets"
REPORT_PATH = CLEAN_ROOT / "verification_report.json"


@dataclass(frozen=True)
class SheetConfig:
    key: str
    raw_sheet: Path
    clean_dir: Path
    sheet_path: Path
    cols: int
    rows: int
    frame_width: int
    frame_height: int
    safe_margin: int
    anchor: str
    basename: str


@dataclass(frozen=True)
class SingleAssetConfig:
    key: str
    raw_sheet: Path
    clean_path: Path
    width: int
    height: int
    safe_margin: int
    anchor: str


SHEET_CONFIGS = {
    "launch_body": SheetConfig(
        key="launch_body",
        raw_sheet=RAW_ROOT / "launch" / "rocket_launch_body_raw_sheet.png",
        clean_dir=CLEAN_ROOT / "launch",
        sheet_path=SHEETS_ROOT / "rocket_launch_body_sheet.png",
        cols=8,
        rows=2,
        frame_width=160,
        frame_height=256,
        safe_margin=14,
        anchor="bottom",
        basename="launch_body",
    ),
    "flame": SheetConfig(
        key="flame",
        raw_sheet=RAW_ROOT / "launch" / "rocket_flame_raw_sheet.png",
        clean_dir=CLEAN_ROOT / "launch",
        sheet_path=SHEETS_ROOT / "rocket_flame_sheet.png",
        cols=8,
        rows=1,
        frame_width=96,
        frame_height=128,
        safe_margin=12,
        anchor="bottom",
        basename="flame",
    ),
    "smoke": SheetConfig(
        key="smoke",
        raw_sheet=RAW_ROOT / "launch" / "rocket_smoke_raw_sheet.png",
        clean_dir=CLEAN_ROOT / "launch",
        sheet_path=SHEETS_ROOT / "rocket_smoke_sheet.png",
        cols=6,
        rows=2,
        frame_width=160,
        frame_height=96,
        safe_margin=10,
        anchor="bottom",
        basename="smoke",
    ),
    "rocket_trail": SheetConfig(
        key="rocket_trail",
        raw_sheet=RAW_ROOT / "launch" / "rocket_trail_raw_sheet.png",
        clean_dir=CLEAN_ROOT / "launch",
        sheet_path=SHEETS_ROOT / "rocket_trail_sheet.png",
        cols=8,
        rows=1,
        frame_width=96,
        frame_height=160,
        safe_margin=12,
        anchor="top",
        basename="trail",
    ),
    "fireworks": SheetConfig(
        key="fireworks",
        raw_sheet=RAW_ROOT / "fireworks" / "fireworks_raw_sheet.png",
        clean_dir=CLEAN_ROOT / "fireworks",
        sheet_path=SHEETS_ROOT / "fireworks_sheet.png",
        cols=8,
        rows=3,
        frame_width=96,
        frame_height=96,
        safe_margin=10,
        anchor="center",
        basename="fireworks",
    ),
    "ants_celebrate": SheetConfig(
        key="ants_celebrate",
        raw_sheet=RAW_ROOT / "ants_celebrate" / "ant_celebrate_raw_sheet.png",
        clean_dir=CLEAN_ROOT / "ants_celebrate",
        sheet_path=SHEETS_ROOT / "ant_celebrate_sheet.png",
        cols=4,
        rows=3,
        frame_width=64,
        frame_height=64,
        safe_margin=10,
        anchor="bottom",
        basename="ants_celebrate",
    ),
}

SINGLE_ASSET_CONFIGS = {
    "launch_base": SingleAssetConfig(
        key="launch_base",
        raw_sheet=RAW_ROOT / "launch" / "launch_pad_base_core_complete_raw.png",
        clean_path=CLEAN_ROOT / "launch" / "launch_pad_base_core_complete.png",
        width=402,
        height=292,
        safe_margin=8,
        anchor="center",
    ),
}

CHROMA_KEYS = (
    (0, 255, 0),
    (255, 0, 255),
)


def ensure_dirs() -> None:
    for path in [
        RAW_ROOT / "launch",
        RAW_ROOT / "fireworks",
        RAW_ROOT / "ants_celebrate",
        CLEAN_ROOT / "launch",
        CLEAN_ROOT / "fireworks",
        CLEAN_ROOT / "ants_celebrate",
        SHEETS_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def is_close_to_color(pixel: tuple[int, int, int, int], target: tuple[int, int, int], threshold: int) -> bool:
    r, g, b, a = pixel
    if a < 8:
        return True
    return abs(r - target[0]) <= threshold and abs(g - target[1]) <= threshold and abs(b - target[2]) <= threshold


def is_background_pixel(pixel: tuple[int, int, int, int]) -> bool:
    r, g, b, a = pixel
    if a < 8:
        return True
    for key in CHROMA_KEYS:
        if is_close_to_color(pixel, key, 26):
            return True
    if g > 92 and g > r + 28 and g > b + 28:
        return True
    if r > 92 and b > 92 and r > g + 24 and b > g + 24:
        return True
    if r > 245 and g > 245 and b > 245:
        return True
    if abs(r - g) < 12 and abs(g - b) < 12 and r > 228:
        return True
    return False


def scrub_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            if is_background_pixel(pixels[x, y]):
                pixels[x, y] = (0, 0, 0, 0)
    return rgba


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.getchannel("A").getbbox()


def contiguous_ranges(indices: Iterable[int]) -> list[tuple[int, int]]:
    indices = sorted(set(indices))
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def separator_ranges(image: Image.Image, axis: str, threshold: float = 0.92) -> list[tuple[int, int]]:
    spans: list[int] = []
    if axis == "x":
        for x in range(image.width):
            white_count = 0
            for y in range(image.height):
                pixel = image.getpixel((x, y))
                if pixel[3] < 8:
                    continue
                if pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240:
                    white_count += 1
            if white_count >= image.height * threshold:
                spans.append(x)
    else:
        for y in range(image.height):
            white_count = 0
            for x in range(image.width):
                pixel = image.getpixel((x, y))
                if pixel[3] < 8:
                    continue
                if pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240:
                    white_count += 1
            if white_count >= image.width * threshold:
                spans.append(y)
    return contiguous_ranges(spans)


def content_spans(length: int, separators: list[tuple[int, int]], expected_count: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in separators:
        if cursor < start:
            spans.append((cursor, start))
        cursor = end + 1
    if cursor < length:
        spans.append((cursor, length))
    spans = [(start, end) for start, end in spans if end - start > 4]
    if len(spans) == expected_count:
        return spans
    cell = length / expected_count
    fallback: list[tuple[int, int]] = []
    for index in range(expected_count):
        start = round(index * cell)
        end = round((index + 1) * cell)
        fallback.append((start, end))
    return fallback


def slice_sheet_frames(raw_sheet: Path, cols: int, rows: int) -> list[tuple[str, Image.Image, tuple[int, int, int, int]]]:
    image = Image.open(raw_sheet).convert("RGBA")
    x_spans = content_spans(image.width, separator_ranges(image, "x"), cols)
    y_spans = content_spans(image.height, separator_ranges(image, "y"), rows)
    frames: list[tuple[str, Image.Image, tuple[int, int, int, int]]] = []
    index = 0
    for row, (top, bottom) in enumerate(y_spans):
        for col, (left, right) in enumerate(x_spans):
            box = (left, top, right, bottom)
            frames.append((f"{raw_sheet.stem}_{index:02d}.png", image.crop(box), box))
            index += 1
    return frames


def measure_source_edge_contact(frame: Image.Image) -> dict[str, int]:
    bbox = alpha_bbox(frame)
    if bbox is None:
        return {"left": frame.width, "right": frame.width, "top": frame.height, "bottom": frame.height}
    return {
        "left": bbox[0],
        "right": frame.width - bbox[2],
        "top": bbox[1],
        "bottom": frame.height - bbox[3],
    }


def detect_edge_flags(image: Image.Image, bbox: tuple[int, int, int, int] | None) -> tuple[bool, bool, bool]:
    if bbox is None:
        return False, False, False
    left, top, right, bottom = bbox
    white_flag = False
    gray_flag = False
    checker_score = 0
    for y in range(top, bottom):
        for x in range(left, right):
            pixel = image.getpixel((x, y))
            if pixel[3] < 8:
                continue
            near_edge = x - left <= 1 or right - x <= 2 or y - top <= 1 or bottom - y <= 2
            if not near_edge:
                continue
            r, g, b, _ = pixel
            if r > 244 and g > 244 and b > 244:
                white_flag = True
            if abs(r - g) < 10 and abs(g - b) < 10 and 150 <= r <= 225:
                gray_flag = True
            if (r > 210 and g > 210 and b > 210) and ((x + y) % 2 == 0):
                checker_score += 1
    return white_flag, gray_flag, checker_score >= 8 and white_flag and gray_flag


def paste_with_anchor(image: Image.Image, size: tuple[int, int], anchor: str) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    bbox = alpha_bbox(image)
    if bbox is None:
        return canvas, None
    content = image.crop(bbox)
    available_width = max(1, size[0] - 4)
    available_height = max(1, size[1] - 4)
    scale = min(1, available_width / content.width, available_height / content.height)
    if scale < 1:
        resized_width = max(1, round(content.width * scale))
        resized_height = max(1, round(content.height * scale))
        content = content.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    if anchor == "bottom":
        offset_x = max(0, (size[0] - content.width) // 2)
        offset_y = max(0, size[1] - content.height - 2)
    elif anchor == "top":
        offset_x = max(0, (size[0] - content.width) // 2)
        offset_y = 2
    elif anchor == "center":
        offset_x = max(0, (size[0] - content.width) // 2)
        offset_y = max(0, (size[1] - content.height) // 2)
    else:
        offset_x = max(0, (size[0] - content.width) // 2)
        offset_y = max(0, (size[1] - content.height) // 2)
    canvas.alpha_composite(content, (offset_x, offset_y))
    return canvas, alpha_bbox(canvas)


def build_sheet(config: SheetConfig, frame_paths: Iterable[Path]) -> None:
    sheet = Image.new(
        "RGBA",
        (config.frame_width * config.cols, config.frame_height * config.rows),
        (0, 0, 0, 0),
    )
    for index, frame_path in enumerate(frame_paths):
        frame = Image.open(frame_path).convert("RGBA")
        col = index % config.cols
        row = index // config.cols
        if row >= config.rows:
            break
        sheet.alpha_composite(frame, (col * config.frame_width, row * config.frame_height))
    sheet.save(config.sheet_path)


def clean_frame(
    frame: Image.Image,
    source_name: str,
    source_box: tuple[int, int, int, int],
    output_path: Path,
    frame_width: int,
    frame_height: int,
    safe_margin: int,
    anchor: str,
) -> dict:
    scrubbed = scrub_background(frame)
    source_bbox = alpha_bbox(scrubbed)
    source_margins = measure_source_edge_contact(scrubbed)
    white_edge, gray_edge, checker_flag = detect_edge_flags(scrubbed, source_bbox)
    warnings: list[str] = []
    if source_bbox is None:
        warnings.append("empty_frame")
        clean = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
        final_bbox = None
    else:
        expand = safe_margin
        left = max(0, source_bbox[0] - expand)
        top = max(0, source_bbox[1] - expand)
        right = min(scrubbed.width, source_bbox[2] + expand)
        bottom = min(scrubbed.height, source_bbox[3] + expand)
        expanded = scrubbed.crop((left, top, right, bottom))
        clean, final_bbox = paste_with_anchor(expanded, (frame_width, frame_height), anchor)
        if min(source_margins.values()) <= 0:
            warnings.append("touches_source_edge")
    final_bbox = alpha_bbox(clean)
    if final_bbox is None:
        margins = {"left": frame_width, "right": frame_width, "top": frame_height, "bottom": frame_height}
        touches_edge = False
    else:
        margins = {
            "left": final_bbox[0],
            "right": frame_width - final_bbox[2],
            "top": final_bbox[1],
            "bottom": frame_height - final_bbox[3],
        }
        touches_edge = min(margins.values()) <= 0
        if touches_edge:
            warnings.append("touches_edge")
        if min(margins.values()) < 2:
            warnings.append("insufficient_margin")
    frame_bleed = min(source_margins.values()) <= 0 or (final_bbox is not None and min(margins.values()) < 2)
    if white_edge:
        warnings.append("suspected_white_edge")
    if gray_edge:
        warnings.append("suspected_gray_edge")
    if checker_flag:
        warnings.append("suspected_checkerboard")
    if frame_bleed:
        warnings.append("suspected_frame_bleed")
    clean.save(output_path)
    return {
        "source": source_name,
        "source_box": list(source_box),
        "clean": str(output_path.relative_to(ROOT)),
        "bbox": list(final_bbox) if final_bbox else None,
        "touches_edge": touches_edge,
        "margins": margins,
        "source_margins": source_margins,
        "suspected_white_edge": white_edge,
        "suspected_gray_edge": gray_edge,
        "suspected_checkerboard": checker_flag,
        "suspected_frame_bleed": frame_bleed,
        "warnings": warnings,
    }


def clean_group(config: SheetConfig) -> dict:
    config.clean_dir.mkdir(parents=True, exist_ok=True)
    frames = slice_sheet_frames(config.raw_sheet, config.cols, config.rows)
    reports: list[dict] = []
    clean_paths: list[Path] = []
    for index, (source_name, frame, source_box) in enumerate(frames[: config.cols * config.rows]):
        output_path = config.clean_dir / f"{config.basename}_{index:02d}.png"
        report = clean_frame(
            frame=frame,
            source_name=str(config.raw_sheet.relative_to(ROOT)),
            source_box=source_box,
            output_path=output_path,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
            safe_margin=config.safe_margin,
            anchor=config.anchor,
        )
        reports.append(report)
        clean_paths.append(output_path)
    build_sheet(config, clean_paths)
    return {
        "sheet": str(config.sheet_path.relative_to(ROOT)),
        "expected_frames": config.cols * config.rows,
        "cleaned_frames": len(clean_paths),
        "frames": reports,
        "warnings": [frame["warnings"] for frame in reports if frame["warnings"]],
    }


def clean_single_asset(config: SingleAssetConfig) -> dict:
    config.clean_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(config.raw_sheet).convert("RGBA")
    report = clean_frame(
        frame=image,
        source_name=str(config.raw_sheet.relative_to(ROOT)),
        source_box=(0, 0, image.width, image.height),
        output_path=config.clean_path,
        frame_width=config.width,
        frame_height=config.height,
        safe_margin=config.safe_margin,
        anchor=config.anchor,
    )
    return {
        "asset": str(config.clean_path.relative_to(ROOT)),
        "frames": [report],
        "warnings": [report["warnings"]] if report["warnings"] else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw animation assets into verified sprite sheets.")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=list(SHEET_CONFIGS.keys()) + list(SINGLE_ASSET_CONFIGS.keys()),
        choices=list(SHEET_CONFIGS.keys()) + list(SINGLE_ASSET_CONFIGS.keys()),
        help="Asset groups to process.",
    )
    args = parser.parse_args()

    ensure_dirs()

    report = {"groups": {}, "warnings": []}
    for key in args.groups:
        if key in SHEET_CONFIGS:
            group_report = clean_group(SHEET_CONFIGS[key])
        else:
            group_report = clean_single_asset(SINGLE_ASSET_CONFIGS[key])
        report["groups"][key] = group_report
        for frame in group_report["frames"]:
            if frame["warnings"]:
                report["warnings"].append({
                    "group": key,
                    "frame": frame["clean"],
                    "warnings": frame["warnings"],
                })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verification report written to {REPORT_PATH}")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"WARNING {warning['group']} {warning['frame']}: {', '.join(warning['warnings'])}")


if __name__ == "__main__":
    main()
