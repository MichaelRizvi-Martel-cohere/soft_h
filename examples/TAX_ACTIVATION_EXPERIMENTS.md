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
  --dataset-parquet \
    gs://cohere-dev/michael-rizvi/soft_h/c4/c4_en_validation_rev1588ec45_seed0_n10000_drydock/documents.parquet \
  --max-batches 1 \
  --run-name limpgods_ckpt536_block0
```

The preview prints the checkpoint, destination, hooks, resource settings,
queue, image behavior, and exact extraction command.

Defaults:

- context: `cw-us-east-04-prod`;
- queue: `post-training-smifs-queue`;
- priority: `dev-medium`;
- compute: one worker with eight H100s, using Tax's compute YAML;
- time limit: one hour;
- hook: `block_0_block_output`;
- input: an explicit frozen C4 Parquet sample through Fax's Drydock eval loader,
  with maximum sequence length 512;
- tensor parallelism: 4;
- FSDP parallelism: automatic, which uses the two remaining ways on an
  eight-GPU worker;
- output: `gs://cohere-dev/michael-rizvi/soft_h_tax/<run-name>`;
- sample fragment: `configs/sample.run.fragment`, with its generic sharding
  replaced by the BLS settings above;
- image: build from the current Tax checkout under the timestamp tag generated
  by `kjobs fax submit`.

Run `examples/submit_tax_activations.sh --help` for all options. There is no
catalog of Fax image tags and the launcher does not choose one. Tax's
`kjobs fax submit` flow generates `YYYYMMDDHHMM-$(whoami)`, builds the image,
and fills `head.image.tag` and `worker.image.tag`.

The C4 path uses Fax's existing balanced evaluation loader but replaces its
checkpoint-specific mixture with one fixed Drydock Parquet source. Tax does not
contain a separate JSONL tokenizer or batcher.

The extractor applies TP to both checkpoint loading and the inference
fragment, then lets Fax assign unused devices to FSDP. TP is a model-sharding
choice, not the number of requested GPUs. The exact checkpoint load path
rejected TP=8 against four expert groups during pre-load validation; TP=4
passes. With one eight-GPU worker, TP=4 and automatic FSDP=2 use all eight
GPUs.

Before submitting, the launcher runs the exact checkpoint and extraction
settings locally in CPU-only preflight mode using the `caios` AWS profile. It
loads and validates the checkpoint, computes the local four-device simulated
mesh, loads the checkpoint tokenizer, and constructs the first C4 batch. This
catches checkpoint/configuration validation, impossible mesh products,
credentials, tokenizer, and basic batch-shape errors without waiting for GPUs.
It does not validate the actual eight-GPU FSDP=2 mesh, load model weights,
execute hooks, or compile a GPU forward pass.

## Configuration evidence and pre-submit rules

This section records a repository-wide search performed on 2026-08-24 after
several BLS smoke-job failures.

### What the kjobs documentation covers

The
[kjobs usage guide](https://cookbook.cohere.com/recipes/00_components/kjobs/02_usage.html)
and `cohere-ai/kueue-jobs-cli/docs/submit.md` describe Kubernetes allocation:
workers, GPUs per worker, CPU, memory, queue, priority, image, command, and
environment variables. They do not define Fax model parallelism or determine
whether a model fits in GPU memory.

`worker.count * worker.gpu` gives Fax the physical device count. TP, FSDP, EP,
SP, DP, batch size, and MoE expert groups are separate model/runtime choices
validated inside Tax.

### Existing Limpgods paths

The exact Limpgods checkpoint appears in many SFT and RLVR configurations, but
the Tax repository contains no established raw-Fax `fax generate` or
activation-extraction recipe for BLS 3A30T. The closest raw-Fax entry points
are:

- `fax_mono/fax/generate.py`;
- `scripts/extract_activations.py`, which uses the same
  `runtime_config.load_generate_config` and `InferenceZord` path;
- generic `configs/sample.run.fragment`; and
- experiment-local `configs/bls_smoke.run.fragment`.

Do not describe `bls_smoke.run.fragment` as a team-known-good configuration.
It was introduced during this integration work and still requires an
end-to-end successful GPU run.

Historical training comments say Limpgods was written with one expert group,
while the exact current generate-config load reported four groups before
inference overrides. Generate later derives `n_expert_groups = EP * TP`.
Therefore, do not copy or hard-code an expert-group value from an unrelated
training file; load the exact checkpoint through the intended runtime path and
let Tax validate it.

There is an established serving path for the same model:

- `workspace/agent_expert/standalone/deploy_baselines.py` points the Limpgods
  baseline at its promoted `hf_export/fp8` artifact;
- `workspace/agent_expert/standalone/hf_export_deploy_eval.py` selects a
  two-H100 node profile for BLS 3A30T; and
- `workspace/agent_expert/standalone/eval_h100.py` lists the deployed Limpgods
  model among prior H100 baselines.

That evidence shows that the FP8 HF export can be served, but it is not a
configuration for this experiment. Deployment uses HF-exported weights and
vLLM/Kinfer rather than the raw Fax/JAX checkpoint, and it cannot expose the
Tax intermediate hooks used here. Likewise, the checked-in RLVR and SFT
configs are training/rollout configurations with many GPUs, data loaders,
sidecars, and loss settings; their resource and sharding values should not be
copied into this forward-only job.

### Authoritative Fax invariants

The enforced rules are in Tax:

1. **Device mesh.** `fax_mono/fax/device/hardware.py` computes
   `PP * DP * FSDP * EP * SP * TP = number of devices`. If FSDP is `None`, it
   absorbs the remaining device axis. Explicit axes may not exceed the
   allocation, TP/EP/SP must divide the device count, and PP must divide the
   worker count.
2. **Data submesh.** The batch axis is sharded over
   `DP * FSDP * EP`. `fax_mono/fax/utils/train_utils.py` defines
   `check_batch_size_with_mesh`, and existing Tax sweep configs explicitly
   require `eval_batch_size` to be divisible by this product.
3. **MoE groups.** `fax_mono/fax/config/config.py` requires
   `EP * TP <= n_expert_groups`, `n_expert_groups` divisible by `EP * TP`,
   and the number of experts divisible by `n_expert_groups`. Generate-time
   inference sets `n_expert_groups = EP * TP` in
   `fax_mono/fax/config/runtime_config.py`.
4. **Attention heads.** Tax validates that the number of attention heads is
   divisible by `TP * SP`.
5. **Inference overrides.** Generate-time inference forces SP=1,
   `scan_forward_pass=False`, `stack_layer_params_at_init=False`,
   megatron-style shardings on, and parameter quantization on. The typed
   `GenerateFragment` rejects unsupported fields.

Inference enables batch padding, but a non-divisible `eval_batch_size` is
still unsafe for this workflow: model construction can use the unpadded shape
before inference input padding runs. The eight-GPU smoke with TP=4, FSDP=2,
and batch size 1 failed in `shard_map` for exactly this reason. Treat
divisibility as a strict pre-submit requirement.

For the activation launcher's current eight-GPU settings:

```text
devices = 8
TP=4, EP=1, SP=1, PP=1, automatic FSDP=2, DP=1
data submesh = DP * FSDP * EP = 2
required eval_batch_size multiple = 2
configs/sample.run.fragment eval_batch_size = 2
```

The static mesh and batch arithmetic therefore agrees for the current
launcher. Fax's existing loader creates the shape-2 batches.

### Which observed failures were preventable

- **TP=8 with BLS MoE groups:** statically preventable by loading the exact
  checkpoint configuration and running Tax's existing validators.
- **TP=4, FSDP=2, batch size 1:** statically preventable from
  `eval_batch_size % (DP * FSDP * EP) == 0`.
- **Free-text split into positional CLI arguments:** eliminated by passing a
  shell-safe Parquet URI rather than document text through the launcher.
- **Checkpoint-specific evaluation mixture:** replaced by one explicit C4
  Drydock source through the native Fax loader.
- **Two- and four-H100 compilation OOM:** not reliably predictable from
  kjobs documentation or CPU config validation. Only a known-good
  model-specific GPU configuration or a real XLA compile/forward probe can
  establish memory fit.

### Required pre-submit review

Before any new model/topology combination:

1. distinguish a raw Fax checkpoint from an `hf_export`/Kinfer deployment;
2. search Tax for the exact checkpoint and model family, but copy only
   settings from the same execution path;
3. preview and record worker count, GPUs per worker, TP, FSDP, EP, SP, PP,
   inferred DP, and eval batch size;
4. require the complete mesh product to equal the requested device count;
5. require `eval_batch_size` to be divisible by `DP * FSDP * EP`;
6. load the exact checkpoint config so Tax validates MoE groups and attention
   heads;
7. run the launcher's CPU preflight; and
8. treat the first full-topology GPU compile as the memory smoke test, not as
   a configuration validator.

The launcher should eventually call Tax's batch/mesh guard explicitly and
print these derived values. Until that is implemented, perform the arithmetic
above during preview. A CPU preflight success is necessary but does not prove
that a raw BLS forward pass or activation capture fits on the requested GPUs.

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

1. runs the checkpoint-aware CPU preflight described above;
2. runs `kubectl auth whoami` to trigger OIDC if necessary;
3. builds a Fax image from the current Tax checkout;
4. submits through Tax's existing Ray JobSet configuration;
5. requests one eight-H100 worker;
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

## Test online accumulation locally

The online integration test runs a native Drydock C4 batch through a
randomly initialized two-layer Fax model, accumulates both block outputs, and
checks the result against the existing NPZ analyzer:

```bash
PYTHONPATH="${HOME}/repos/soft_h:${HOME}/repos/tax" \
  uv run --project "${HOME}/repos/tax" --extra drydock \
  python "${HOME}/repos/soft_h/examples/test_tax_online_entropy_integration.py"
```

This is a CPU-only local Ray test and does not submit a Kubernetes job. Success
requires 128 aligned token rows, 104 positions with complete quadgram context,
finite metrics for both hooks, numerical agreement between online and offline
analysis, and no persistent activation shard in the online output.

Use multiple batches to stress the accumulator checkpoint round trip with real
random-model activations:

```bash
PYTHONPATH="${HOME}/repos/soft_h:${HOME}/repos/tax" \
  uv run --project "${HOME}/repos/tax" --extra drydock \
  python "${HOME}/repos/soft_h/examples/test_tax_online_entropy_integration.py" \
  --n-batches 20
```

With `--online-entropy`, the launcher hashes the local `soft_entropy` Python
source, stages that exact source under Tax's `scripts/` Docker build context,
and sets the image command's `PYTHONPATH` accordingly. It verifies the staged
digest before submission, records the digest in `results.json`, and removes the
temporary source copy after `kjobs fax submit` returns.

Online launcher runs checkpoint every 50 completed batches by default. Override
that with `--checkpoint-every`; zero disables checkpointing. To resume, reuse
the original run name and total `--max-batches`, then pass the complete numbered
checkpoint explicitly:

```bash
examples/submit_tax_activations.sh \
  ... \
  --online-entropy \
  --max-batches 250 \
  --run-name <same-run-name> \
  --resume-from \
    gs://cohere-dev/michael-rizvi/soft_h_tax/<same-run-name>/checkpoints/batch_00100
```

Resume reloads and recompiles the model, but it does not repeat model forwards
for the first 100 batches. It replays those batches through the unshuffled
evaluation loader and verifies their token-prefix hash before continuing.

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
