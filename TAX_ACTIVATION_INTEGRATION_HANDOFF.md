# Tax activation extraction to soft-entropy: project handoff

Last updated: 2026-08-21

## Goal

Use Fax/Tax to load a model checkpoint, run model inference with intermediate
activation hooks, export aligned activation and token-label tensors, and consume
those tensors in `soft_h` to estimate soft entropy and mutual information.

## Native data path decision

Always use Fax's existing evaluation pipeline for dataset-backed activation
experiments. Do not add a parallel JSONL tokenizer or batcher to Tax.

The shortest C4 path is:

1. `soft_h` freezes the selected public C4 documents once.
2. Tax's `scripts/prepare_c4_unpacked.py` tokenizes each document independently
   with the checkpoint tokenizer, retains at most the first 512 tokens, and
   writes exactly one native unpacked TFRecord row per document.
3. Patch the checkpoint run config to use one fixed source:
   `eval_data_dir_dict = {"unpacked": {"gs://.../unpacked": 1.0}}`.
4. Fax's existing `get_data_loaders()` and `GPTBalancedDataLoader` pad, label,
   batch, and shard those rows without packing documents or emitting
   continuation chunks.
5. The Tax extraction script only runs `forward_with_hooks_step` and exports
   aligned activations and loader-provided labels/sequence metadata.

This route requires no changes to `fax_mono`, no new Tax dataloader, no literal
prompt, and no custom model-side batcher. The small preprocessing script writes
the existing Fax unpacked TFRecord contract; dataset selection remains a
run-config override rather than model inference code.

The repositories remain independent. Tax writes a portable directory of numeric
NPZ shards plus a JSON manifest. `soft_h` reads that directory. There is no
Python package dependency between the repositories.

## Repository roles

- `/home/michael_rizvi_martel_cohere_com/repos/tax`
  - Fax model configuration, checkpoint restoration, distributed JAX inference,
    and activation hooks.
- `/home/michael_rizvi_martel_cohere_com/repos/soft_h`
  - Streaming soft-entropy and mutual-information estimation.
- `/home/michael_rizvi_martel_cohere_com/repos/command_data`
  - Dataset processing and Fax training-config generation. It is not required
    for the initial checkpoint inference or activation smoke tests.
- `/home/michael_rizvi_martel_cohere_com/repos/apiary`
  - Evaluation of deployed or otherwise served models. It is not required for
    native Fax checkpoint inference.

No files were changed in `command_data` or `apiary`.

## Implemented data flow

1. Tax restores a checkpoint and obtains one batch from its evaluation loader.
2. `InferenceZord.forward_with_hooks_step` returns selected `[B, S, D]`
   activations.
3. The batch mask is applied identically to activations, input tokens, and
   output tokens.
4. Tax writes:
   - `batch_XXXXX.npz` numeric shards;
   - `manifest.json` describing hooks, dimensions, sample counts, and checkpoint.
5. `soft_h` streams each `[N, D]` activation matrix and aligned `[N]` token
   labels into one `SoftEntropyAccumulator` per hook.
6. The consumer reports `H(Z)`, token mutual information, regularity,
   dimensions, and sample counts.

The exported labels currently mean:

- `input_token`: token presented at a sequence position;
- `output_token`: next-token training label at that position;
- `sequence_id`: included when available, but not currently analyzed.

## Changes in `tax`

### `scripts/extract_activations.py` (new)

- Loads an inference-ready config from a Fax checkpoint.
- Resolves modern checkpoint weights separately from config loading: it reads
  `run_config.json` and `arch_config.json` with `state_ckpt=None`, applies Fax's
  existing generation overrides, and then assigns `config.run.ckpt_dir`. This
  avoids loading large serialized training metadata such as a prefetched batch
  and requires no `fax_mono` change.
- Rejects legacy pre-GDA checkpoints, whose weight path still depends on that
  training metadata.
- Runs selected representation-shaped hooks for a bounded number of batches.
- Masks and flattens activations and token labels consistently.
- Writes numeric NPZ shards and a versioned JSON manifest through `co.fs`, so
  local paths and supported object-storage URIs work.
- Refuses to overwrite existing shards/manifests.
- Validates hook syntax and activation shapes.
- Explicitly imports Fax registrations so the standalone script can construct
  checkpoint dataloaders.

Default hook: `block_0_block_output`.

Hooks require:

- `run.scan_forward_pass = False`;
- `run.performance.stack_layer_params_at_init = False`.

Recommended initial hooks are `embedding_output` and block outputs.
`layernorm_2_act` is accepted by the parser but is not consistently emitted by
all architectures, so it should not be used without an architecture-specific
smoke test.

### `tests/fax/extract_activations_test.py` (new)

Tests hook parsing, aligned masking/flattening, shape rejection, numeric NPZ
round-trips, overwrite protection, and avoidance of object arrays.

### `tests/fax/hooks.py` (modified)

Modernized the old hook integration test:

- disables scanned forward passes and stacked-at-init parameters;
- accesses `ExecutorOutputs.outputs`;
- exercises the currently emitted CPU-model hook types.

## Changes in `soft_h`

### `examples/analyze_tax_activations.py` (new)

- Validates the manifest schema and shard metadata.
- Prevents manifest path traversal outside the activation directory.
- Requires aligned one-dimensional token labels.
- Rejects malformed, non-finite, and zero-norm activations.
- Streams shards without concatenating the full dataset in memory.
- Computes results independently for every exported hook.
- Supports deterministic reference sampling through `--seed`.

Example:

```bash
uv run python examples/analyze_tax_activations.py \
  /path/to/activation_directory \
  --n-bins 100 \
  --seed 0 \
  --output /path/to/results.json
```

### `tests/test_analyze_tax_activations.py` (new)

Tests deterministic finite results and manifest shard path-traversal rejection.

### `soft_entropy/__init__.py` (modified)

Loads `LLMInferrer` lazily. This prevents the optional `tqdm` dependency from
being required when only `SoftEntropyAccumulator` is used.

### `uv.lock` (untracked)

A lockfile was generated while setting up/running the local environment. Review
whether the project wants this file before committing it.

## Verification completed

### Unit and formatting checks

Latest Tax checks:

```bash
uv run pre-commit run --files \
  scripts/extract_activations.py \
  tests/fax/extract_activations_test.py \
  tests/fax/hooks.py

uv run --group test pytest tests/fax/extract_activations_test.py -q
```

Result: pre-commit passed and all 9 extractor unit tests passed.

The corresponding `soft_h` consumer tests also passed during implementation.

All changes are currently uncommitted.

### Tiny random-model activation smoke

Ran locally on the login VM's CPUs with a local Ray cluster. It therefore did
not appear in `kjobs` or the Kubernetes queue.

- Model: randomly initialized two-layer `tiny_mup`, 2.13M parameters.
- Width: 256.
- Data: one tokenized CI batch, not a literal text prompt.
- Hooks: embedding output, first block output, final block output.
- Exported valid token positions: 32,690.
- `soft_h`: successfully computed entropy and input/output-token mutual
  information for all three hooks.

Artifacts were written under:

```text
/tmp/soft_h_tiny_ci_20260821_1534/
```

The numerical values are not scientifically meaningful because the model
weights are random. The smoke validates the technical handoff.

### Tiny checkpoint save/restore smoke

Also ran locally on the VM's CPUs:

1. Initialized the same 2.13M-parameter model.
2. Saved a real Fax checkpoint.
3. Shut down that model/Ray process.
4. Restored the checkpoint through the production checkpoint-loading path.
5. Ran the actual `scripts.extract_activations` implementation.
6. Exported 417 valid token positions for three hooks.
7. Successfully analyzed the shards with `soft_h`.

This additionally tested checkpoint config/metadata, parameter serialization,
restoration, and resharding for inference.

Artifacts:

```text
/tmp/soft_h_tiny_ckpt_20260821_1538/
```

These `/tmp` artifacts are ephemeral.

The CI model can perform autoregressive generation, but its 1,024-token model
vocabulary does not match the normal text tokenizer. A prompt-based tiny smoke
would need either a matching toy tokenizer or a larger random embedding/output
vocabulary. Its generated text would still be gibberish.

## Real BLS checkpoint

The smallest real pretrained model identified was BLS (Baby Light Speed):

- approximately 30B total parameters;
- approximately 3B active parameters;
- promoted agent-expert checkpoint:

```text
s3://us-east-01a/promoted-checkpoints/post-training/tomsherborne_cohere_com/sft/c5_babylightspeed_agents_expert/tmshrb0611125249-peakcoil/1/ckpt-2771
```

The attempted real-model smoke only ran `fax generate` on the literal prompt
`"Hello"` for two generated tokens. It did not request or cache activations.

The real checkpoint restored successfully. Remaining failure is during
GPU compilation for sampling, not checkpoint access.

## Cluster interface

Tax provides `ops/kjobs-compute.yaml`, which defines a Ray JobSet:

- CPU Ray head;
- GPU Ray worker(s);
- small CPU exec pod that submits the command to Ray.

Cluster authentication:

```bash
kjobs setup cluster cw-us-east-04-prod
```

OIDC opens `http://localhost:8000/`. On the VM this may print an `xdg-open`
error; open the URL through Cursor's forwarded port in a local browser.

Useful commands:

```bash
kjobs --context cw-us-east-04-prod list
kjobs --context cw-us-east-04-prod logs JOB_NAME
kjobs --context cw-us-east-04-prod cancel JOB_NAME
```

Submission settings used for the BLS smoke:

- context: `cw-us-east-04-prod`;
- queue: `post-training-capa-agentic-queue`;
- priority: `dev-low`;
- time limit: one hour;
- head: 4 CPU, 64 GiB;
- worker: one node, initially 2 H100s, 14 CPU, 256 GiB;
- image:
  `iad.ocir.io/ax8fanrz7l8n/fax:202608211900-michael-rizvi-bls-smoke`.

The queue was selected explicitly because the `workspace/agent_expert`
post-training scripts use the agentic post-training capacity. It was not the
Fax wrapper's default queue.

Use `uv run --no-sync` inside this image. Running `uv run --frozen` or otherwise
syncing in the pod tries to fetch private Git dependencies such as `jaxpp`
without the required GitHub SSH credentials.

The image tag was built from the current Tax checkout with the canonical
`kjobs fax submit` flow. An assumed `fax:main` tag did not exist and caused an
`ImagePullBackOff`.

Do not pass `FAX_NUMBER_WORKERS` or `FAX_NUMBER_GPUS_PER_WORKER` manually:
`kjobs` derives and injects them from `worker.count` and `worker.gpu`, and Kueue
rejects duplicated environment variables.

## BLS submission failures and lessons

1. **Missing image tag**
   - `fax:main` did not exist in OCIR.
   - Fixed by building and using the timestamped image above.

2. **Dependency sync inside the pod**
   - `uv run --frozen --no-editable` attempted an SSH clone of private `jaxpp`.
   - Fixed by using dependencies already baked into the image with
     `uv run --no-sync`.

3. **Wrong sampling fragment for BLS**
   - `configs/ci/ci_sample.run.fragment` disables parameter/activation
     quantization.
   - This produced float32 values rejected by BLS's sparse scatter kernel.
   - Fixed by using `configs/sample.run.fragment`, changing only generated
     tokens from 20 to 2 and tensor parallelism for the attempted hardware.

4. **Ray log-stream disconnect**
   - One attempt ended when the exec pod's Ray WebSocket closed after a long
     quiet compilation interval.
   - A wrapper now prints a heartbeat every 30 seconds while `fax generate`
     runs. This kept the log/control stream active and exposed the underlying
     compiler failure.

5. **Two H100s are insufficient for this BLS sampling graph**
   - We overrode production tensor parallelism from 8 to 2.
   - During sampling compilation, XLA reported about 69.0 GB of input/output
     arguments against a roughly 68.0 GB compiler memory budget.
   - `xla::Autotuner::ProfileAll()` then segfaulted.
   - This was not a queue, authentication, checkpoint, or Python-level error.

Latest failed heartbeat job:

```text
vm-michael-rizvi-fax-202608211925-5fz3
```

It remains as a failed UI record unless explicitly cancelled. Earlier failed
records were cancelled at the user's request.

## Recommended next steps

### Monday: real BLS generation smoke

Obtain explicit approval, then submit:

- one 8-H100 worker;
- tensor parallelism 8, matching `configs/sample.run.fragment`;
- the existing image tag and checkpoint;
- two generated tokens from `"Hello"`;
- the 30-second heartbeat;
- `dev-low` priority and the established agentic queue.

Eight H100s are the canonical configuration, not merely a performance
optimization. Four may be sufficient based on the observed memory excess, but
that would be another noncanonical experiment. Prefer TP=8 for the next
diagnostic unless cost/capacity requires a staged TP=4 attempt.

The submission is materially more expensive than the approved 2-H100 run and
requires fresh explicit approval.

### C4 activation extraction

The controlled input begins with the pinned English C4 validation JSONL
artifact described by its adjacent manifest. Before model inference,
`scripts/prepare_c4_unpacked.py` uses the checkpoint tokenizer to write one
variable-length native TFRecord row per source document, capped at 512 tokens.
Fax's existing unpacked evaluation path then pads, labels, batches, and shards
the rows. It never combines documents or emits a second chunk for a long
document. With evaluation batch size 2, 10,000 paper samples are exactly 5,000
model batches.

The `soft_h` consumer defaults to collision-free input/output unigram labels,
the token-level backoff used for the paper's primary analyses. Higher-order
backoff is an explicit `label_types` selection. It masks positions without the
requested preceding and trailing context, computes one result per block-output
hook, and averages metrics uniformly across hooks.

The native path has been exercised end to end on CPU with Fax's existing
two-layer `tiny_mup` architecture and random initialization. The test used the
production 50,261-token tokenizer, a 32-token context, four C4 sequences, and
both block-output hooks. Fax/Drydock produced the batch, Tax exported 128 aligned
activation rows of dimension 256 per hook, and `soft_h` analyzed 104 positions
with complete quadgram context. All entropy, MI, regularity, and optimality
metrics were finite. This was an integration test only, not a scientific model
result.

Tax now also has an `--online_entropy=True` mode in
`scripts/extract_activations.py`. The existing hooked forward still produces one
batch at a time, but the main loop immediately passes the aligned arrays to
`soft_h`'s `TaxActivationAccumulator` and then discards them. The accumulator
sums soft-bin assignments globally, including per-ngram conditional counts; it
does not average per-batch entropy values. Online mode writes only
`results.json`, not activation NPZ shards.

`examples/test_tax_online_entropy_integration.py` exercises the paper-aligned
unpacked path with the production tokenizer, four-way simulated CPU mesh, and
two-layer random `tiny_mup`. The latest 20-batch stress run consumed exactly 80
paper-aligned samples and 2,491 valid unigram positions for each 256-dimensional
block output. Every entropy, MI, regularity, and optimality metric was finite,
and online, resumed, and NPZ-based analyses matched within numerical tolerance.

The checkpoint stress test checkpoints after batch 10, restores into new
accumulators, replays the same unpacked data prefix, and matches both
uninterrupted accumulation and the 20-shard offline analyzer. Earlier Drydock
testing exposed that one-token tuple labels were being restored as scalars;
checkpoint schema version 2 preserves scalar-versus-tuple label identity.

For online submissions, the launcher deterministically hashes the Python source
under `soft_entropy`, stages an exact copy under Tax's `scripts/` image context,
and sets the container `PYTHONPATH` to that packaged copy. The Tax Dockerfile
already copies `scripts/` into `/app`, so no runtime network install is needed.
The launcher verifies the staged hash, records it in `results.json`, and removes
the temporary build-context copy after `kjobs fax submit` returns. The first
packaged preflight used digest
`d6a4cbdb299b3df779fbb028d2f08805662d3e23b7201d600d81c9e3d5060841`.
This implementation removes persistent activation storage but still transfers
each batch's raw hook outputs from Fax workers to the Ray head.

Online mode supports numbered accumulator checkpoints through
`--checkpoint_every=N` and explicit resume through `--resume_from=...`. A
checkpoint contains only numeric soft-bin counts, conditional n-gram counts,
reference points, counters, and a run fingerprint; it does not duplicate the
unchanged Fax model checkpoint. Tax writes `COMPLETE` last and retains only the
newest complete accumulator checkpoint. Resume reloads the same Fax model,
restores the accumulator, and replays already-consumed evaluation batches without
model inference. A rolling hash of the aligned token prefix must match before
new forwards run, so changed data order or configuration fails closed.
`max_batches` remains the total target rather than the number of additional
batches after resume.

The first BLS GPU gate succeeded as job
`vm-michael-rizvi-fax-202608241920-ub6y`. It used eight H100s with TP=4,
automatic FSDP=2, a `[2, 512]` C4 batch, and one
`block_0_block_output` hook. Checkpoint step 536 restored, the hooked forward
compiled and executed, and Tax wrote 1,024 float32 activation rows of dimension
2,048 plus aligned token and sequence metadata. The complete Ray job took about
seven minutes and produced:

`gs://cohere-dev/michael-rizvi/soft_h_tax/bls30b_ckpt536_c4_block0_b1_seq512_native_v1`

After staging the manifest and 8 MiB NPZ shard locally, `soft_h` analyzed 1,012
positions with complete quadgram context. All entropy, MI, regularity, and
optimality metrics were finite; for this integration batch,
`H(Z) = 0.8805911541`. The current analysis CLI uses `pathlib` and therefore
requires local staging rather than accepting the `gs://` directory directly.

The checkpoint has 48 layers with hidden dimension 2,048. A full batch contains
at most 1,022 valid input/label activation rows because each 512-token source
document yields at most 511 shifted positions. At 5,000 batches, persisting all
48 block outputs would still be prohibitive; online accumulation avoids that
storage.

Hooks retain full `[B, S, D]` representations before masking/export and may need
more memory than generation alone. TP=8 is therefore a generation baseline,
not a guarantee that many simultaneous BLS hooks will fit.

All extraction uses the checkpoint evaluation-loader interface. C4 comparisons
replace only its data source with `--eval_data_path`; there is no custom JSONL
branch in Tax.

### Scientific choices still required

The fixed protocol is English C4 validation, 10,000 frozen samples, 512-token
maximum context, all transformer block outputs, input/output unigram labels,
100 bins, and seed 0. Higher-order backoff requires explicit opt-in. The paper
does not specify its exact
split, sampler, activation hook, special-token policy, or seeds, so the artifact
and activation manifests record our explicit choices.

Random-model and CI-batch results are integration tests only.

## Working-tree state

The checkpoint/resume baseline is committed locally as `7d73880` in `soft_h`
and `f0ab51983` in Tax. The paper-aligned unpacked-data changes described above
are newer working-tree changes and remain uncommitted. Generated result
artifacts are intentionally excluded from source commits.
