"""Shared byte, geometry and file contracts; no inference or visual verdicts."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile

from PIL import Image

APPEARANCE_CHECKS = {"style_consistency", "material_construction", "spatial_detail_hierarchy", "optics_or_markmaking"}
PEOPLE_CHECKS = {"eyes_and_gaze", "mouth_and_teeth", "skin_texture", "identity_and_distinctness",
                 "hair_hands_and_anatomy", "wardrobe_and_material_integration"}
WEIGHTS_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


def validate_resolution(approval: dict, size: tuple[int, int]) -> None:
    """Enforce the approved target style's long-edge limit without resizing."""
    look = approval["review"]["visual_contract"]["target_look"]
    if look not in {"camera_photo", "render", "illustration", "graphic", "mixed"}:
        raise ValueError("Unknown approved target look")
    limit = 6144 if look == "camera_photo" else 12288
    if min(size) < 1 or max(size) > limit:
        raise ValueError(f"{look} output long edge must be at most {limit} pixels")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local_file(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Names must be a single non-empty filename component")
    path = directory / name
    if path.resolve().parent != directory.resolve():
        raise ValueError("File path escapes its task directory")
    return path


def new_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")


def new_task_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f"Task directory is not empty: {path}; continue using its existing manifest")
    path.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".",
                                         suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def immutable_artifact(directory: Path, name: str, source: Path) -> Path:
    """Install a content-addressed revision; failed copies cannot damage accepted bytes."""
    digest = sha256(source)
    destination = local_file(directory, f"{name}-{digest}.png")
    if destination.is_symlink():
        raise ValueError("Artifact destination cannot be a symlink")
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError("Existing immutable artifact changed")
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="artifact-", suffix=".tmp", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(source, temporary)
        if sha256(temporary) != digest:
            raise ValueError("Artifact changed during copy")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


@contextmanager
def locked_manifest(path: Path):
    """Keep a stable sidecar lock across atomic inode replacement."""
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            yield data
            atomic_json(path, data)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def box(value, size):
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError("Boxes require integer [x,y,width,height]")
    x, y, w, h = value
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > size[0] or y + h > size[1]:
        raise ValueError("Box exceeds its image canvas")
    return value


def contains(outer, inner):
    x, y, w, h = outer
    ix, iy, iw, ih = inner
    return x <= ix and y <= iy and ix + iw <= x + w and iy + ih <= y + h


def load_approval(path: Path, source: Path = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("approval_version") != 4 or data.get("approved") is not True:
        raise ValueError("Require a version 4 semantic approval")
    master = Path(data["master"])
    if source is not None and master.resolve() != source.resolve():
        raise ValueError("Approval belongs to a different source")
    if sha256(master) != data["master_sha256"] or data["review"].get("image_sha256") != data["master_sha256"]:
        raise ValueError("Approval/review source hash changed")
    for field in ("subject_report", "repair_manifest"):
        bound = data.get(field)
        if bound and sha256(Path(bound["path"])) != bound["sha256"]:
            raise ValueError(f"Approval {field} changed")
    return data


def verify_subject_report(report, image: Path, stage: str, expected_ids=None):
    if (report.get("schema_version") != 2 or report.get("stage") != stage
            or report.get("image_sha256") != sha256(image)
            or report.get("manual_semantic_review_required") is not True):
        raise ValueError("Subject report stage or image hash does not match")
    entries = report.get("subjects")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Subject report requires non-empty subjects")
    ids = [item.get("id") for item in entries]
    if any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("Subject IDs must be unique and non-empty")
    if expected_ids is not None and set(ids) != set(expected_ids):
        raise ValueError("Subject IDs do not match the required people")
    with Image.open(image) as pixels:
        for item in entries:
            face = box(item.get("face_box"), pixels.size)
            if stage in {"repair", "final"} and (face[2] < 120 or item.get("automated_failures")):
                raise ValueError("Subject repair/final precheck failed or face density below 120px")
    return {entry["id"]: entry for entry in entries}


def verify_repairs(path: Path, master: Path, subjects=None, target_size=None):
    data = json.loads(path.read_text(encoding="utf-8"))
    if Path(data["source"]).resolve() != master.resolve() or data["source_sha256"] != sha256(master):
        raise ValueError("Repair manifest source hash or path changed")
    with Image.open(master) as pixels:
        size = pixels.size
    if data.get("source_size") != list(size):
        raise ValueError("Repair manifest source_size does not match the master")
    covered = set()
    if not data.get("regions"):
        raise ValueError("Repair manifest needs regions")
    for item in data["regions"]:
        target = box(item["target_box"], size)
        crop = box(item["crop_box"], size)
        if not contains(crop, target):
            raise ValueError("Repair target exceeds its crop")
        output = local_file(path.parent, item["output"])
        if item.get("status") != "accepted" or sha256(output) != item.get("output_sha256"):
            raise ValueError("Repair output is not accepted or changed")
        with Image.open(output) as pixels:
            native = pixels.size
        if list(native) != item.get("output_size") or native[0] < crop[2] or native[1] < crop[3]:
            raise ValueError("Repair native density is insufficient")
        if abs((native[0] / native[1]) / (crop[2] / crop[3]) - 1) > 0.002:
            raise ValueError("Repair native aspect does not match the crop")
        if target_size:
            scale = target_size[0] / size[0]
            if abs(scale - target_size[1] / size[1]) > 1e-8:
                raise ValueError("Repair target canvas must preserve aspect")
            needed = [round((crop[i] + crop[i + 2]) * scale) - round(crop[i] * scale) for i in (0, 1)]
            if native[0] < needed[0] or native[1] < needed[1]:
                raise ValueError("Repair native output cannot support final canvas; generate denser detail")
        ids = item.get("subject_ids", [])
        if not isinstance(ids, list) or len(set(ids)) != len(ids):
            raise ValueError("Invalid repair subject IDs")
        if ids:
            report = item.get("subject_report")
            if not isinstance(report, dict):
                raise ValueError("People repairs require output subject evidence")
            outputs = verify_subject_report(report, output, "repair", ids)
            for sid in ids:
                if subjects is not None and (sid not in subjects or not contains(target, subjects[sid]["face_box"])):
                    raise ValueError(f"Repair target does not cover source face {sid}")
                face = outputs[sid]["face_box"]
                mapped = [crop[0] + face[0] * crop[2] / native[0], crop[1] + face[1] * crop[3] / native[1],
                          face[2] * crop[2] / native[0], face[3] * crop[3] / native[1]]
                if not contains(target, mapped):
                    raise ValueError(f"Repaired face {sid} is outside the replacement target")
        covered.update(ids)
    return covered


def neural_contract_sha256(approval):
    """Repair revisions do not alter the neural input or its semantic review."""
    content = {key: approval.get(key) for key in ("master_sha256", "master_size", "master_mode", "review", "subject_report")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_neural_report(report, approval_path, approval, output_hash):
    approval_matches = report.get("approval_sha256") == sha256(approval_path)
    repair_revision_matches = report.get("neural_contract_sha256") == neural_contract_sha256(approval)
    if (report.get("source_sha256") != approval["master_sha256"]
            or not (approval_matches or repair_revision_matches)
            or report.get("output_sha256") != output_hash):
        raise ValueError("Neural report source/approval/output hash chain does not match")
    if report.get("device") == "mps":
        if report.get("mps_fallback") is not False or report.get("weights_sha256") != WEIGHTS_SHA256 or report.get("trial_box"):
            raise ValueError("Invalid full-run MPS execution evidence")
        visual = approval["review"]
        needs_trial = visual["visual_contract"]["target_look"] == "camera_photo" and (
            report.get("delivery_scale", 0) > 2 or visual.get("subject_report_required") is True)
        if needs_trial and not report.get("trial_review_sha256"):
            raise ValueError("Photographic neural execution has no passing trial evidence")
    elif report.get("route", "").startswith("MPS VAE encode + MPS DINOv2"):
        if (report.get("run_kind") != "full" or not report.get("trial_review_sha256")
                or not report.get("verified_package_plan_sha256") or not report.get("conversion_sha256")):
            raise ValueError("ANE full run lacks crop/package/conversion evidence")
    else:
        raise ValueError("Unsupported neural execution route")


def preserve_mode(result: Image.Image, source: Image.Image) -> Image.Image:
    if source.mode in {"L", "LA"}:
        result = result.convert("L")
    if "A" in source.mode:
        result.putalpha(source.getchannel("A").resize(result.size, Image.Resampling.LANCZOS))
    return result
