# Delivery evidence

Separate file identity, completed backend execution, and human visual acceptance. None substitutes for another. For a requested nonphotographic 12K-detail result, also require the [effective-detail standard](12k-quality.md). A structurally plausible face does not approve the detail level.

Keep the preflight, source/target contract, subject manifest where needed, relevant trial, semantic approval, accepted repairs, measured job ledger, and final byte-bound review. Source/master approvals use version 4; final delivery reports use version 2. Older reports without these contracts are historical evidence, not valid approvals for new execution.

## Finalize

Use a dedicated delivery directory with exactly one image. Bases, trials, contact sheets and repair revisions stay in the job workspace. Final photographic people need a final-stage subject report containing every approved ID, faces at least 120px wide, and native review regions covering them all. Do not change the approved mode, alpha or ICC.

```bash
python scripts/finalize_delivery.py /work/job/delivery/final.png /work/job/delivery/delivery.json \
  --width 6144 --height 4096 \
  --approval /work/job/master.png.approval.json \
  --final-review /work/job/final-review.json \
  --operation-report /work/job/base.png.mps-report.json \
  --operation-report /work/job/delivery/final.png.composite-report.json
```

Dimensions above are an example, not an automatic crop. Add `--subject-report /work/job/final-people/contact-sheet-report.json` for critical people. Remove the composite report only when no repair pack was approved. For ANE use its `base-runtime-report.json` instead of the MPS report. All operation reports are supplied in execution order, neural first.

Keep hash-bound source/result/half-size comparison evidence and final-scale repair density calculations in the job directory; bind the evidence JSON with `--artifact detail-evidence=/absolute/path/evidence.json`. Human inspection determines effective detail; this flag records evidence and does not automatically grade it. Detail failure must be reflected in the existing appearance/people checks, so it blocks delivery.

The finalizer checks source/approval/output hash lineage, every approved repair, final appearance and protected-content checks, subject coverage, image dimensions/mode/profile, and the final bytes. Review after the last export: a previous checksum or visual approval is stale.

## Multiple requested resolutions

Only for nonphotographic targets, derive a smaller candidate from a version 2 approved parent. Photography is capped at 6K and has no parent-12K workflow:

```bash
python scripts/downsample_delivery.py /work/job/12k/delivery.json /work/job/6k/final.png \
  --width 6144 --height 4096
```

This validates the parent image, review, subject and execution evidence before resampling. Review the new bytes, rebuild final subject evidence when needed, and finalize with the original neural/composite report sequence plus the new `.downsample-report.json`. Each requested resolution has its own delivery directory and approval. Never rerun the neural model just to derive a smaller output.
