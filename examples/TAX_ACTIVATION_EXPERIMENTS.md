# Running Tax activation experiments

This guide covers the reusable Bash launcher in
`examples/submit_tax_activations.sh`. It submits a Fax/Tax checkpoint inference
job that exports token-aligned activations for analysis by `soft_h`.

The launcher is intentionally a small wrapper around Tax's existing
`ops/kjobs-compute.yaml` and `kjobs fax submit` flow. It does not add a
`SweepConfig`. Use it for one-off integration tests and a small number of
checkpoint runs. Move to the team's existing SweepConfig patterns when the
experiment becomes a checkpoint-by-hook matrix or needs task dependencies.

## Tax ownership and edit policy

Treat Tax/Fax as a shared dependency that this project consumes, not as
project-owned code. Prefer existing Tax configuration, `patch_run_config`,
supported command-line options, and changes to the `soft_h` launcher or
analysis code.

- Do not modify `tax/fax_mono/` to make an experiment run.
- Do not patch existing Tax internals merely to work around a checkpoint,
  topology, or launch configuration.
- Modify existing Tax code only when evidence identifies a genuine internal
  logic bug, no supported configuration solution is suitable, and Michael has
  explicitly approved the proposed Tax diff.
- Ask for that approval before editing Tax, not only before submitting the
  resulting job.
- Keep any approved Tax change minimal, covered by a focused regression test,
  and suitable for review by the Tax owners.

The activation extractor and its tests under `tax/scripts/` and `tax/tests/`
are integration additions made for this project. Do not expand their scope or
use them as justification for unrelated Tax changes.

## Prerequisites

Run the launcher on the Cohere login VM, with these checkouts:

```text
~/repos/soft_h
~/repos/tax
~/repos/kueue-jobs-cli
```

Install the H100 cluster context once:

```bash
kjobs setup cluster cw-us-east-04-prod
```

At the start of a VM session, ensure that OIDC authentication is current:

```bash
kubectl --context cw-us-east-04-prod auth whoami
```

This reuses a valid cached token and opens the OIDC browser flow only when
needed. The local browser callback requires SSH forwarding from local port 8000
to port 8000 on the VM.

`kjobs setup cluster` only installs and validates the presence of the context;
its `Passed!` message does not prove that an API request can authenticate.

## Preview a job

The launcher is non-submitting by default. Always preview first:

```bash
cd ~/repos/soft_h

examples/submit_tax_activations.sh \
  --checkpoint \
    s3://us-east-01a/promoted-checkpoints/post-training/priyanka_cohere_com/sft/c5_babylightspeed_agents_expert/pranka0618161154-limpgods/1/ckpt-536 \
  --run-name limpgods_ckpt536_block0
```

The preview prints the checkpoint, destination, hooks, resource settings,
queue, image behavior, and exact extraction command.

Defaults:

- context: `cw-us-east-04-prod`;
- queue: `post-training-smifs-queue`;
- priority: `dev-low`;
- compute: one worker with eight H100s, using Tax's compute YAML;
- time limit: one hour;
- hook: `block_0_block_output`;
- evaluation batches: one;
- validation horizon: 1,000 steps, passed through `patch_run_config`;
- tensor parallelism: 8, passed through `patch_run_config` before Tax validates
  derived KV-head replication and then reapplied by the sample fragment;
- output: `gs://cohere-dev/michael-rizvi/soft_h_tax/<run-name>`;
- sample fragment: `configs/sample.run.fragment`, which uses TP=8;
- image: build from the current Tax checkout under the timestamp tag generated
  by `kjobs fax submit`.

Run `examples/submit_tax_activations.sh --help` for all options. There is no
catalog of Fax image tags and the launcher does not choose one. Tax's
`kjobs fax submit` flow generates `YYYYMMDDHHMM-$(whoami)`, builds the image,
and fills `head.image.tag` and `worker.image.tag`.

The validation horizon does not make the extractor run 1,000 batches.
`--max-batches` still bounds extraction. The larger horizon only lets
checkpoints with many low-weight evaluation datasets construct their balanced
evaluation loader when the sampling fragment uses `eval_batch_size=2`.

The explicit tensor-parallel patch intentionally duplicates the sample
fragment's TP=8 value. Tax validates derived sharding fields while loading the
checkpoint, before applying the sample fragment. Supplying TP=8 at load time
lets Tax's existing validator enable KV-head replication for models such as
Limpgods, which has four KV heads. This is a configuration workaround and does
not modify Tax.

## Submit

After reviewing the preview and obtaining explicit approval for the displayed
resource and output specification, repeat the command with `--submit`:

```bash
examples/submit_tax_activations.sh \
  --checkpoint \
    s3://us-east-01a/promoted-checkpoints/post-training/priyanka_cohere_com/sft/c5_babylightspeed_agents_expert/pranka0618161154-limpgods/1/ckpt-536 \
  --run-name limpgods_ckpt536_block0 \
  --submit
```

The launcher:

1. runs `kubectl auth whoami` to trigger OIDC if necessary;
2. builds or reuses the requested Fax image;
3. submits through Tax's existing Ray JobSet configuration;
4. requests one eight-H100 worker;
5. emits a heartbeat every 30 seconds during compilation;
6. runs `scripts/extract_activations.py` with `uv run --no-sync`.

Use a new run name for every attempt. The Tax extractor refuses to overwrite
existing shards or manifests.

## Monitor

The submit command prints the job name. Use it directly:

```bash
kjobs list \
  --context cw-us-east-04-prod \
  --queue post-training-smifs-queue

kjobs logs JOB_NAME --context cw-us-east-04-prod
```

Do not treat the Kubernetes `Succeeded` state alone as proof that extraction
completed. Verify that the destination contains both a manifest and at least
one numeric shard:

```bash
gcloud storage ls \
  "gs://cohere-dev/michael-rizvi/soft_h_tax/<run-name>/**"
```

Expected files include `manifest.json` and `batch_00000.npz`. A directory
containing only `__create_dir__` is an incomplete run.

Do not cancel a job without explicit approval:

```bash
kjobs cancel JOB_NAME --context cw-us-east-04-prod
```

## Analyze the exported activations

`analyze_tax_activations.py` currently reads local files, while the launcher
writes durable GCS artifacts. Copy one completed run to the VM:

```bash
gcloud storage cp --recursive \
  gs://cohere-dev/michael-rizvi/soft_h_tax/limpgods_ckpt536_block0 \
  /tmp/limpgods_ckpt536_block0
```

Then run the deterministic soft-entropy consumer:

```bash
cd ~/repos/soft_h

uv run python examples/analyze_tax_activations.py \
  /tmp/limpgods_ckpt536_block0 \
  --n-bins 100 \
  --seed 0 \
  --output /tmp/limpgods_ckpt536_block0/results.json
```

Before interpreting the result, verify that `manifest.json` reports the
expected checkpoint and hook, `total_samples` is positive, and the hook
dimension is 2048 for BLS.

## Authentication recovery

If `kubectl auth whoami` hangs instead of opening or reusing OIDC, clear only
the OIDC token cache and retry:

```bash
kubectl oidc-login clean
kubectl --context cw-us-east-04-prod auth whoami
```

On a headless VM, `kubectl oidc-login clean` may warn that
`org.freedesktop.secrets` is unavailable after successfully deleting
`~/.kube/cache/oidc-login`. That keyring warning is expected. Do not delete the
entire `~/.kube/config`.
