#!/usr/bin/env python3
"""Small job ledger: real action counts, hash-based reuse and wall-time evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

from pipeline_contract import atomic_json, locked_manifest, new_output, sha256


def files(values):
    result = {}
    for value in values:
        role, raw = value.split("=", 1)
        if not role or role in result:
            raise ValueError("File roles must be unique and non-empty")
        path = Path(raw).resolve()
        result[role] = {"path": str(path), "sha256": sha256(path)}
    return result


def summarize(data):
    actions = data["actions"]
    finished = [item for item in actions if "ended_at" in item]
    data["summary"] = {
        "generation_count": sum(item["kind"] == "generation" for item in actions),
        "full_neural_count": sum(item["kind"] == "neural" for item in actions),
        "trial_count": sum(item["kind"] == "trial" for item in actions),
        "failed_count": sum(item.get("status") == "fail" for item in actions),
        "model_seconds": sum(item.get("model_seconds", 0) for item in finished),
        "reported_tokens": sum(item.get("tokens") or 0 for item in finished),
        "token_measurements": sum(item.get("tokens") is not None for item in finished),
        "wall_seconds": round(time.time() - data["created_epoch"], 3)}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("job", type=Path)
    init.add_argument("--source", type=Path, required=True)
    init.add_argument("--lane", choices=("clean", "local", "global", "evidence-preserving"), required=True)
    init.add_argument("--target", nargs=2, type=int, required=True)
    start = commands.add_parser("start")
    start.add_argument("job", type=Path)
    start.add_argument("--kind", choices=("generation", "trial", "neural", "review", "composite"), required=True)
    start.add_argument("--label", required=True)
    start.add_argument("--input", action="append", required=True)
    finish = commands.add_parser("finish")
    finish.add_argument("job", type=Path)
    finish.add_argument("action_id")
    finish.add_argument("--status", choices=("pass", "fail"), required=True)
    finish.add_argument("--output", action="append", default=[])
    finish.add_argument("--reason", default="")
    finish.add_argument("--model-seconds", type=float, default=0)
    finish.add_argument("--tokens", type=int)
    args = parser.parse_args()
    try:
        if args.command == "init":
            new_output(args.job)
            if min(args.target) < 1:
                raise ValueError("Target dimensions must be positive")
            data = {"version": 1, "created_epoch": time.time(), "source": str(args.source.resolve()),
                    "source_sha256": sha256(args.source), "lane": args.lane, "target": args.target, "actions": []}
            summarize(data)
            atomic_json(args.job, data)
            print(args.job)
            return
        with locked_manifest(args.job) as data:
            finishing_failure = args.command == "finish" and args.status == "fail"
            if not finishing_failure and sha256(Path(data["source"])) != data["source_sha256"]:
                raise ValueError("Job source changed; create a new job plan")
            if args.command == "start":
                inputs = files(args.input)
                fingerprint = hashlib.sha256(json.dumps([args.kind, args.label, inputs], sort_keys=True).encode()).hexdigest()
                for item in data["actions"]:
                    if item["fingerprint"] == fingerprint:
                        if item["status"] == "pass":
                            for value in item["outputs"].values():
                                if sha256(Path(value["path"])) != value["sha256"]:
                                    raise ValueError("Accepted output changed; diagnose before repeating work")
                        raise ValueError("Identical action already recorded; reuse it or change the diagnosed method/input")
                action_id = uuid.uuid4().hex[:12]
                data["actions"].append({"id": action_id, "kind": args.kind, "label": args.label,
                                        "fingerprint": fingerprint, "inputs": inputs, "started_at": time.time(), "status": "running"})
            else:
                matches = [item for item in data["actions"] if item["id"] == args.action_id]
                if len(matches) != 1 or matches[0]["status"] != "running":
                    raise ValueError("Action must exist and still be running")
                action = matches[0]
                for value in action["inputs"].values():
                    if args.status == "pass" and sha256(Path(value["path"])) != value["sha256"]:
                        raise ValueError("Action input changed during execution")
                if args.status == "fail" and not args.reason.strip():
                    raise ValueError("Failure needs a concrete reason")
                if args.status == "pass" and not args.output:
                    raise ValueError("Passing action needs an output artifact or review")
                if not 0 <= args.model_seconds < float("inf") or (args.tokens is not None and args.tokens < 0):
                    raise ValueError("Invalid timing/token values")
                end = time.time()
                action.update(status=args.status, outputs=files(args.output), reason=args.reason.strip(), ended_at=end,
                              elapsed_seconds=round(end-action["started_at"], 3), model_seconds=args.model_seconds, tokens=args.tokens)
                action_id = action["id"]
            summarize(data)
        print(action_id)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
