#!/usr/bin/env python3
"""Create overlapping image tiles and a JSON manifest for semantic editing."""

import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

from apple_preflight import require_apple_silicon


def positions(length: int, count: int, overlap: int) -> tuple[list[int], int]:
    tile = (length + overlap * (count - 1) + count - 1) // count
    if tile >= length:
        return [0], length
    starts = [round(i * (length - tile) / (count - 1)) for i in range(count)]
    starts[-1] = length - tile
    return starts, tile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=192)
    args = parser.parse_args()

    if args.cols < 1 or args.rows < 1 or args.overlap < 1:
        parser.error("rows/cols and overlap must be positive")

    require_apple_silicon([args.image, args.output_dir.parent])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.image)
    if source.mode not in {"RGB", "RGBA", "L", "LA"}:
        parser.error(
            f"Unsupported source mode {source.mode!r}; convert explicitly with a color-managed "
            "workflow instead of silently discarding high bit depth or indexed color"
        )
    original_mode = source.mode
    alpha_name = None
    if "A" in original_mode:
        alpha_name = "source_alpha.png"
        source.getchannel("A").save(args.output_dir / alpha_name, compress_level=2)

    icc_name = None
    icc_profile = source.info.get("icc_profile")
    if icc_profile:
        icc_name = "source_profile.icc"
        (args.output_dir / icc_name).write_bytes(icc_profile)

    if original_mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", source.size, (255, 255, 255, 255))
        background.alpha_composite(source.convert("RGBA"))
        image = background.convert("RGB")
    else:
        image = source.convert("RGB")
    width, height = image.size
    xs, tile_width = positions(width, args.cols, args.overlap)
    ys, tile_height = positions(height, args.rows, args.overlap)
    tiles = []

    for row, y in enumerate(ys, 1):
        for col, x in enumerate(xs, 1):
            name = f"r{row}c{col}"
            filename = f"{name}_input.png"
            image.crop((x, y, x + tile_width, y + tile_height)).save(
                args.output_dir / filename, compress_level=2
            )
            tiles.append({
                "name": name,
                "input": filename,
                "structure_output": f"{name}_structure.png",
                "output": f"{name}_output.png",
                "row": row,
                "col": col,
                "x": x,
                "y": y,
                "width": tile_width,
                "height": tile_height,
                "structural_status": "pending",
                "structure_prompt": None,
                "structure_review_note": None,
                "structure_output_sha256": None,
                "semantic_status": "pending",
                "semantic_prompt": None,
                "semantic_review_note": None,
                "output_sha256": None,
            })

    manifest = {
        "source": str(args.image.resolve()),
        "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "source_mode": original_mode,
        "source_icc": icc_name,
        "source_alpha": alpha_name,
        "width": width,
        "height": height,
        "rows": len(ys),
        "cols": len(xs),
        "tiles": tiles,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    anchor_path = args.output_dir / "scene_anchor.json"
    if not anchor_path.exists():
        anchor_path.write_text(
            json.dumps(
                {
                    "visual_domain": "",
                    "global_subject_identity": "",
                    "composition_and_geometry": "",
                    "lighting_and_color": "",
                    "focus_motion_and_atmosphere": "",
                    "style_language": "",
                    "projection_and_line_families": "",
                    "repeated_systems": [],
                    "emitters_reflections_and_bloom": "",
                    "organic_support_systems": [],
                    "far_field_behavior": "",
                    "occlusion_and_edge_ownership": "",
                    "high_risk_regions": [],
                    "protected_content": [],
                    "forbidden_changes": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(path)


if __name__ == "__main__":
    main()
