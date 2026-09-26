#!/usr/bin/env python3
"""Build a native-pixel subject contact sheet and flag blur or likely duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from apple_preflight import require_apple_silicon


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("subjects", type=Path, help="JSON list containing id and face_box")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-edit-face-width", type=int, default=80)
    parser.add_argument("--min-sharpness", type=float, default=45.0)
    parser.add_argument("--max-hash-distance", type=int, default=7)
    parser.add_argument("--cell-size", type=int, default=256)
    args = parser.parse_args()

    require_apple_silicon([args.image, args.subjects, args.output_dir.parent])
    image = Image.open(args.image).convert("RGB")
    manifest = json.loads(args.subjects.read_text(encoding="utf-8"))
    subjects = manifest.get("subjects") if isinstance(manifest, dict) else manifest
    if not isinstance(subjects, list) or not subjects:
        parser.error("Subject manifest must be a non-empty JSON list")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    hashes = {}
    seen = set()
    for index, subject in enumerate(subjects):
        subject_id = str(subject.get("id", "")).strip()
        box = subject.get("face_box")
        if not subject_id or subject_id in seen:
            parser.error(f"Subject {index} needs a unique non-empty id")
        seen.add(subject_id)
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, int) for value in box):
            parser.error(f"Subject {subject_id!r} needs integer face_box [x,y,width,height]")
        x, y, width, height = box
        if x < 0 or y < 0 or width < 1 or height < 1 or x + width > image.width or y + height > image.height:
            parser.error(f"Subject {subject_id!r} face_box exceeds the image")
        crop = image.crop((x, y, x + width, y + height))
        crop_path = args.output_dir / f"{subject_id}.png"
        crop.save(crop_path, compress_level=2)
        gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hashes[subject_id] = difference_hash(crop)
        entries.append({
            "id": subject_id,
            "face_box": box,
            "crop": str(crop_path),
            "face_width": width,
            "requires_local_repair": width < args.min_edit_face_width,
            "sharpness": sharpness,
            "sharpness_pass": sharpness >= args.min_sharpness,
            "identity": subject.get("identity", {}),
        })

    duplicate_pairs = []
    ids = list(hashes)
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            distance = bin(hashes[left] ^ hashes[right]).count("1")
            if distance <= args.max_hash_distance:
                duplicate_pairs.append({"subjects": [left, right], "hash_distance": distance})

    columns = min(5, len(entries))
    rows = math.ceil(len(entries) / columns)
    label_height = 40
    sheet = Image.new("RGB", (columns * args.cell_size, rows * (args.cell_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        crop = Image.open(entry["crop"]).convert("RGB")
        crop.thumbnail((args.cell_size, args.cell_size), Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        left = column * args.cell_size + (args.cell_size - crop.width) // 2
        top = row * (args.cell_size + label_height) + (args.cell_size - crop.height) // 2
        sheet.paste(crop, (left, top))
        label = f"{entry['id']}  w={entry['face_width']}  sharp={entry['sharpness']:.1f}"
        draw.text((column * args.cell_size + 6, row * (args.cell_size + label_height) + args.cell_size + 8), label, fill="black")
    sheet_path = args.output_dir / "contact-sheet.png"
    sheet.save(sheet_path, compress_level=2)

    failures = []
    failures.extend(f"{entry['id']}: face width below {args.min_edit_face_width}" for entry in entries if entry["requires_local_repair"])
    failures.extend(f"{entry['id']}: sharpness below {args.min_sharpness}" for entry in entries if not entry["sharpness_pass"])
    failures.extend(f"likely duplicate: {pair['subjects'][0]} / {pair['subjects'][1]}" for pair in duplicate_pairs)
    report = {
        "image": str(args.image.resolve()),
        "image_sha256": sha256(args.image),
        "subject_manifest": str(args.subjects.resolve()),
        "subject_manifest_sha256": sha256(args.subjects),
        "contact_sheet": str(sheet_path.resolve()),
        "subjects": entries,
        "duplicate_pairs": duplicate_pairs,
        "pass": not failures,
        "failures": failures,
    }
    report_path = args.output_dir / "contact-sheet-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(sheet_path)
    print(report_path)
    if failures:
        parser.exit(2, "CONTACT_SHEET_REJECTED: " + "; ".join(failures) + "\n")


if __name__ == "__main__":
    main()
