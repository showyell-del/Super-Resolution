# Cost and continuation

One job JSON is enough; no scheduler or database. Keep it outside the delivery directory on the work volume. Choose the lane before generating.

```bash
python scripts/job.py init /work/job/job.json \
  --source /work/job/source.png --lane local --target 6144 4096
python scripts/job.py start /work/job/job.json --kind generation --label face-a-v1 \
  --input crop=/work/job/repair/a-input.png --input prompt=/work/job/a-prompt.txt
```

The second command returns an action ID. Run the tool, then record its actual result:

```bash
python scripts/job.py finish /work/job/job.json ACTION_ID --status pass \
  --output repair=/work/job/generated-a.png --output review=/work/job/a-review.json
```

Use `--kind trial|neural|review|composite` for other steps. Include relevant settings, approvals and model/runtime reports as inputs or outputs. `--model-seconds` and `--tokens` are measured values only; absent provider token metrics stay unknown. Wall time begins at initialization and call counts include failed attempts.

On failure, use `--status fail --reason "concrete observed failure"`. Changed/missing inputs must not prevent logging failure. Passing actions require output artifacts and unchanged inputs.

Identical kind/label/input hashes are rejected, not executed again. Reuse accepted artifacts. To retry, diagnose the cause and record a genuinely changed prompt/settings/input, or a narrowly corrected runtime attempt with a distinct label and reason. Do not rename unchanged work to evade reuse. A failed region invalidates only that region; preserve the neural base and other repairs.
