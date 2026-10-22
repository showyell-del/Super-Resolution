"""Small, source-bound regression fixtures; never evidence of photographic quality."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
validate_native_review = importlib.import_module("finalize_delivery").validate_native_review
semantic = importlib.import_module("approve_semantic_master")
REQUIRED_CHECKS = semantic.REQUIRED_CHECKS
REQUIRED_PEOPLE_CHECKS = semantic.REQUIRED_PEOPLE_CHECKS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(image):
    return {
        "image_sha256": digest(image), "review_scale": "100% native pixels",
        "visual_contract": {"source_look": "graphic", "target_look": "graphic",
                            "truth_mode": "evidence_preserving", "source_evidence": "Fixture",
                            "target_reason": "Preserve fixture", "detail_policy": "Keep edges"},
        "appearance_checks": {k: "pass" for k in (
            "style_consistency", "material_construction", "spatial_detail_hierarchy", "optics_or_markmaking")},
        "global_checks": {k: "pass" for k in REQUIRED_CHECKS},
        "subject_report_required": False,
        "regions": [{"box": [0, 0, 128, 128], "name": "fixture", "status": "pass", "note": "Fixture only"}],
        "reviewer_note": "Numeric fixture; not an appearance benchmark"}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("SR_TEST_ROOT", "/Volumes/work/Ai/Super-Resolution-tests"))
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.source = self.dir / "source.png"
        rng = np.random.default_rng(42)
        self.pixels = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(self.pixels).save(self.source)
        self.prompt = self.write("prompt.txt", "Keep the fixture")

    def write(self, name, content):
        path = self.dir / name
        path.write_text(content if isinstance(content, str) else json.dumps(content))
        return path

    def run_cli(self, script, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                              capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

    def assert_rejected(self, result, text=None):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        if text:
            self.assertIn(text.lower(), result.stderr.lower())

    def prepare(self, name="patch", subjects=None):
        definition = self.write("regions.json", [{"name": name, "x": 16, "y": 16, "width": 96,
                                                   "height": 96, "subject_ids": subjects or []}])
        result = self.run_cli("prepare_regions.py", self.source, definition, self.dir / "repair", "--context", 0)
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.dir / "repair/regions-manifest.json", definition

    def test_final_explicit_people_failure_is_rejected(self):
        value = review(self.source)
        value["people_review"] = {"checks": {k: "pass" for k in REQUIRED_PEOPLE_CHECKS},
                                  "subjects": [{"id": "a", "status": "fail", "note": "Broken eye"}]}
        value["people_review"]["checks"]["eyes_and_gaze"] = "fail"
        with self.assertRaises(ValueError):
            validate_native_review(value, digest(self.source), (128, 128))

    def test_final_protected_failure_is_rejected(self):
        value = review(self.source)
        value["global_checks"]["protected_content"] = "fail"
        with self.assertRaises(ValueError):
            validate_native_review(value, digest(self.source), (128, 128))

    def test_delivery_directory_rejects_second_image(self):
        from finalize_delivery import validate_delivery_directory
        validate_delivery_directory(self.source)
        Image.fromarray(self.pixels).save(self.dir / "other-candidate.png")
        with self.assertRaisesRegex(ValueError, "one"):
            validate_delivery_directory(self.source)
        (self.dir / "other-candidate.png").unlink()
        Image.fromarray(self.pixels).save(self.dir / "other.gif")
        with self.assertRaisesRegex(ValueError, "one"):
            validate_delivery_directory(self.source)

    def test_downsample_requires_approved_parent_and_new_output(self):
        from downsample_delivery import downsample
        final = self.dir / "smaller.png"
        report = self.write("parent.json", {"delivery_version": 1, "final": {"path": str(self.source)}})
        with self.assertRaises(ValueError):
            downsample(report, final, (64, 64))
        self.assertFalse(final.exists())

    def test_semantic_stale_review_is_rejected(self):
        value = review(self.source)
        value["image_sha256"] = "0" * 64
        path = self.write("review.json", value)
        self.assert_rejected(self.run_cli("approve_semantic_master.py", self.source, path), "hash")

    def test_unsupported_mode_rejected_before_approval(self):
        Image.fromarray(self.pixels).convert("P").save(self.source)
        path = self.write("review.json", review(self.source))
        self.assert_rejected(self.run_cli("approve_semantic_master.py", self.source, path), "mode")

    def test_prepare_regions_cannot_reset_accepted(self):
        manifest, definition = self.prepare()
        data = json.loads(manifest.read_text())
        data["regions"][0]["status"] = "accepted"
        manifest.write_text(json.dumps(data))
        before = manifest.read_bytes()
        self.assert_rejected(self.run_cli("prepare_regions.py", self.source, definition, manifest.parent))
        self.assertEqual(manifest.read_bytes(), before)

    def test_prepare_tiles_cannot_reset_task(self):
        output = self.dir / "tiles"
        args = (self.source, output, "--cols", 2, "--rows", 2, "--overlap", 32, "--context-reviewed")
        first = self.run_cli("prepare_tiles.py", *args)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = (output / "manifest.json").read_bytes()
        self.assert_rejected(self.run_cli("prepare_tiles.py", *args))
        self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_semantic_tiles_reject_one_pixel_overlap(self):
        self.assert_rejected(self.run_cli("prepare_tiles.py", self.source, self.dir / "tiles",
                                          "--cols", 2, "--rows", 2, "--overlap", 1))

    def test_region_name_cannot_escape(self):
        definition = self.write("regions.json", [{"name": "../escape", "x": 0, "y": 0, "width": 32, "height": 32}])
        self.assert_rejected(self.run_cli("prepare_regions.py", self.source, definition, self.dir / "repair"))

    def test_contact_subject_name_cannot_escape(self):
        subjects = self.write("subjects.json", {"subjects": [{"id": "../escaped", "face_box": [0, 0, 120, 120]}]})
        result = self.run_cli("build_contact_sheet.py", self.source, subjects, self.dir / "contacts", "--stage", "final")
        self.assert_rejected(result)
        self.assertFalse((self.dir / "escaped.png").exists())

    def test_verification_region_name_cannot_escape(self):
        regions = self.write("regions.json", [{"name": "../escaped", "x": 0, "y": 0, "width": 64, "height": 64}])
        result = self.run_cli("verify_output.py", self.source, regions, self.dir / "verification",
                              "--width", 128, "--height", 128)
        self.assert_rejected(result)
        self.assertFalse((self.dir / "escaped.png").exists())

    def test_normalization_cannot_overwrite_source(self):
        before = digest(self.source)
        self.assert_rejected(self.run_cli("normalize_canvas.py", self.source, self.source,
                                          "--target-width", 1, "--target-height", 1))
        self.assertEqual(digest(self.source), before)

    def test_duplicate_verification_names_rejected_before_writing(self):
        regions = self.write("regions.json", [{"name": "same", "x": x, "y": 0, "width": 32, "height": 32}
                                               for x in (0, 64)])
        folder = self.dir / "verification"
        self.assert_rejected(self.run_cli("verify_output.py", self.source, regions, folder,
                                          "--width", 128, "--height", 128))
        self.assertFalse(folder.exists())

    def test_untrimmed_subject_id_rejected_before_writing(self):
        subjects = self.write("subjects.json", {"subjects": [{"id": name, "face_box": [0, 0, 120, 120]}
                                                             for name in ("a", " a")]})
        folder = self.dir / "contacts"
        self.assert_rejected(self.run_cli("build_contact_sheet.py", self.source, subjects, folder, "--stage", "final"))
        self.assertFalse(folder.exists())
        self.assertFalse((self.dir / "escape-input.png").exists())

    def test_all_definitions_validated_before_any_write(self):
        definition = self.write("regions.json", [
            {"name": "valid", "x": 0, "y": 0, "width": 32, "height": 32},
            {"name": "invalid", "x": 999, "y": 0, "width": 32, "height": 32}])
        self.assert_rejected(self.run_cli("prepare_regions.py", self.source, definition, self.dir / "repair"))
        self.assertFalse((self.dir / "repair/valid-input.png").exists())

    def test_low_density_patch_is_not_accepted(self):
        manifest, _ = self.prepare()
        tiny = self.dir / "tiny.png"
        Image.new("RGB", (1, 1)).save(tiny)
        self.assert_rejected(self.run_cli("record_region.py", manifest, "patch", tiny,
                                          "--prompt-file", self.prompt, "--review-note", "Inspected"))
        self.assertEqual(json.loads(manifest.read_text())["regions"][0]["status"], "pending")

    def test_people_patch_requires_output_subject_evidence(self):
        manifest, _ = self.prepare(subjects=["a"])
        output = self.dir / "patch.png"
        Image.open(manifest.parent / "patch-input.png").resize((192, 192)).save(output)
        self.assert_rejected(self.run_cli("record_region.py", manifest, "patch", output,
                                          "--prompt-file", self.prompt, "--review-note", "Inspected"))

    def accepted_patch(self):
        manifest, _ = self.prepare()
        output = manifest.parent / "patch-input.png"
        result = self.run_cli("record_region.py", manifest, "patch", output,
                              "--prompt-file", self.prompt, "--review-note", "Fixture inspected")
        self.assertEqual(result.returncode, 0, result.stderr)
        return manifest

    def test_composite_cannot_overwrite_source(self):
        manifest = self.accepted_patch()
        before = self.source.read_bytes()
        self.assert_rejected(self.run_cli("composite_regions.py", manifest, self.source))
        self.assertEqual(self.source.read_bytes(), before)

    def test_composite_preserves_alpha(self):
        alpha = Image.fromarray(self.pixels).convert("RGBA")
        channel = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)
        alpha.putalpha(Image.fromarray(channel))
        alpha.save(self.source)
        manifest = self.accepted_patch()
        output = self.dir / "composite.png"
        result = self.run_cli("composite_regions.py", manifest, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(output) as image:
            self.assertEqual(image.mode, "RGBA")
            np.testing.assert_array_equal(np.asarray(image.getchannel("A")), channel)

    def test_delivery_requires_execution_chain(self):
        path = self.write("final-review.json", review(self.source))
        self.assert_rejected(self.run_cli("finalize_delivery.py", self.source, self.dir / "delivery.json",
                                          "--width", 128, "--height", 128, "--final-review", path))

    def approval(self, repairs=None):
        value = review(self.source)
        path = self.write("semantic-review.json", value)
        args = [self.source, path]
        if repairs:
            args.extend(["--repair-manifest", repairs])
        result = self.run_cli("approve_semantic_master.py", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.source.with_suffix(".png.approval.json")

    def test_wrong_composite_base_evidence_is_rejected(self):
        manifest = self.accepted_patch()
        approval = self.approval(manifest)
        base = self.dir / "base.png"
        Image.open(self.source).save(base)
        report = self.write("wrong-report.json", {"source_sha256": "0" * 64,
                                                  "output_sha256": digest(base), "device": "mps"})
        self.assert_rejected(self.run_cli("composite_regions.py", manifest, self.dir / "result.png",
                                          "--base", base, "--approval", approval, "--base-report", report))

    def test_background_region_cannot_count_as_face_repair(self):
        from pipeline_contract import verify_repairs
        manifest, _ = self.prepare(subjects=["a"])
        data = json.loads(manifest.read_text())
        item = data["regions"][0]
        output = manifest.parent / item["output"]
        Image.open(manifest.parent / item["input"]).resize((192, 192)).save(output)
        subject = {"id": "a", "face_box": [32, 32, 120, 120], "automated_failures": []}
        item.update(status="accepted", output_size=[192, 192], output_sha256=digest(output),
                    subject_report={"schema_version": 2, "stage": "repair", "image_sha256": digest(output),
                                    "manual_semantic_review_required": True, "subjects": [subject]})
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "cover"):
            verify_repairs(manifest, self.source, {"a": {"face_box": [0, 0, 12, 12]}})

    def test_patch_cannot_exceed_its_native_final_density(self):
        from pipeline_contract import verify_repairs
        manifest = self.accepted_patch()
        with self.assertRaisesRegex(ValueError, "native output"):
            verify_repairs(manifest, self.source, target_size=(256, 256))

    def test_correct_execution_chain_passes_and_wrong_final_fails(self):
        from finalize_delivery import validate_execution_chain
        from pipeline_contract import load_approval, WEIGHTS_SHA256
        approval_path = self.approval()
        approval = load_approval(approval_path)
        neural = {"device": "mps", "mps_fallback": False, "weights_sha256": WEIGHTS_SHA256,
                  "source_sha256": digest(self.source), "approval_sha256": digest(approval_path),
                  "output_sha256": "a" * 64, "delivery_scale": 2}
        validate_execution_chain(approval_path, approval, [{"content": neural}], "a" * 64)
        with self.assertRaises(ValueError):
            validate_execution_chain(approval_path, approval, [{"content": neural}], "b" * 64)

    def test_repair_only_revision_reuses_neural_base_but_not_changed_review(self):
        from pipeline_contract import load_approval, verify_neural_report, neural_contract_sha256, WEIGHTS_SHA256
        old_path = self.approval()
        old = load_approval(old_path)
        neural = {"device": "mps", "mps_fallback": False, "weights_sha256": WEIGHTS_SHA256,
                  "source_sha256": digest(self.source), "approval_sha256": digest(old_path),
                  "neural_contract_sha256": neural_contract_sha256(old),
                  "output_sha256": "a" * 64, "delivery_scale": 2}
        revised = json.loads(old_path.read_text())
        revised["approved_at_utc"] = "repair-only-revision"
        revised["repair_manifest"] = {"path": "new-pack.json", "sha256": "b" * 64}
        revised_path = self.write("revised-approval.json", revised)
        verify_neural_report(neural, revised_path, revised, "a" * 64)
        revised["review"]["visual_contract"]["truth_mode"] = "creative_reconstruction"
        with self.assertRaises(ValueError):
            verify_neural_report(neural, revised_path, revised, "a" * 64)

    def test_ane_full_requires_trial_before_runtime_import(self):
        approval = self.approval()
        result = self.run_cli("vosr2_ane_hybrid.py", "--source", self.source, "--approval", approval,
                              "--output", self.dir / "ane.png", "--scratch", self.dir / "scratch")
        self.assert_rejected(result, "Full execution requires")
        self.assertFalse((self.dir / "scratch").exists())

    def test_ane_trial_binds_source_packages_settings_and_pixels(self):
        from ane_contract import validate_ane_trial
        approval = self.approval()
        plan = self.write("plan.json", {"fixture": True})
        image = self.dir / "trial.png"
        Image.fromarray(self.pixels[:64, :64]).resize((256, 256)).save(image)
        trial_review = review(image)
        trial_review["image"] = str(image)
        review_path = self.write("trial-review.json", trial_review)
        conversion = {"fixture": "pinned-package-evidence"}
        self.write("trial-runtime-report.json", {
            "source_sha256": digest(self.source), "approval_sha256": digest(approval),
            "output_sha256": digest(image), "output_size": [256, 256], "run_kind": "diagnostic",
            "seed": 42, "align": "wavelet", "package_plan_sha256": digest(plan),
            "conversion_sha256": conversion, "trial_box": [0, 0, 64, 64]})
        validate_ane_trial(review_path, self.source, approval, plan, 42, "wavelet", conversion)
        with self.assertRaises(ValueError):
            validate_ane_trial(review_path, self.source, approval, plan, 43, "wavelet", conversion)
        with self.assertRaises(ValueError):
            validate_ane_trial(review_path, self.source, approval, plan, 42, "wavelet", {"changed": True})
        Image.new("RGB", (256, 256)).save(image)
        with self.assertRaises(ValueError):
            validate_ane_trial(review_path, self.source, approval, plan, 42, "wavelet", conversion)

    def test_job_duplicate_finished_action_is_not_repeated(self):
        job = self.dir / "job.json"
        result = self.run_cli("job.py", "init", job, "--source", self.source, "--lane", "local", "--target", 256, 256)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = ("start", job, "--kind", "generation", "--label", "face-a", "--input", f"source={self.source}")
        result = self.run_cli("job.py", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        action = result.stdout.strip()
        result = self.run_cli("job.py", "finish", job, action, "--status", "pass", "--output", f"image={self.source}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_rejected(self.run_cli("job.py", *args))
        data = json.loads(job.read_text())
        self.assertEqual(data["summary"]["generation_count"], 1)

    def test_semantic_explicit_people_fail_cannot_be_ignored(self):
        value = review(self.source)
        value["people_review"] = {"checks": {k: "fail" for k in REQUIRED_PEOPLE_CHECKS},
                                  "subjects": [{"id": "a", "status": "fail"}]}
        path = self.write("review.json", value)
        self.assert_rejected(self.run_cli("approve_semantic_master.py", self.source, path))

    def test_changed_input_can_be_recorded_as_true_job_failure(self):
        job = self.dir / "job.json"
        self.run_cli("job.py", "init", job, "--source", self.source, "--lane", "local", "--target", 256, 256)
        prompt = self.write("changed-prompt.txt", "original")
        result = self.run_cli("job.py", "start", job, "--kind", "generation", "--label", "face",
                              "--input", f"prompt={prompt}")
        self.assertEqual(result.returncode, 0, result.stderr)
        prompt.write_text("changed")
        finished = self.run_cli("job.py", "finish", job, result.stdout.strip(), "--status", "fail",
                                "--reason", "Input changed during execution")
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(json.loads(job.read_text())["summary"]["failed_count"], 1)

    def test_repair_manifest_wrong_aspect_is_rejected(self):
        from pipeline_contract import verify_repairs
        manifest = self.accepted_patch()
        data = json.loads(manifest.read_text())
        output = manifest.parent / data["regions"][0]["output"]
        Image.new("RGB", (96, 4096)).save(output)
        data["regions"][0].update(output_size=[96, 4096], output_sha256=digest(output))
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "aspect"):
            verify_repairs(manifest, self.source)


if __name__ == "__main__":
    unittest.main()
