# Tax activation extraction to soft-entropy: project handoff

Last updated: 2026-08-26

## Goal

The long-term goal is to use Fax/Tax to load internal Cohere checkpoints, run
model inference with intermediate activation hooks, and use `soft_h` to estimate
soft entropy and mutual information without persisting prohibitively large
activation datasets.

The immediate objective is a final sanity check between two versions of the same
model: one public Hugging Face release and one internal Cohere Fax checkpoint.
Both backends must process exactly the same token IDs with the same estimator
configuration. If their per-layer entropy metrics agree within a prespecified
tolerance, this validates that entropy estimation in the Cohere Tax/Fax setup is
faithful to the previous Hugging Face `soft_h` code path.

This is an integration-equivalence test, not a new scientific result. The
current paired models are internal `c3_7B_12-2024_command_release` checkpoint
499 and public `CohereLabs/c4ai-command-r7b-12-2024` revision
`4f3d0aa6856e322f2f4480fe65420d5d53d297b8`.

## Native data path decision

Always use Fax's existing evaluation pipeline for dataset-backed activation
experiments. Do not add a parallel JSONL tokenizer or batcher to Tax.

The current C4 path is:

1. `soft_h` froze 10,000 selected public C4 validation documents once.
2. The model tokenizer was used to create an immutable Fax-native unpacked
   TFRecord artifact with one independently truncated row per source sample.
   The temporary preparation script has been removed; experiments consume the
   versioned artifact rather than regenerating it.
3. Patch the checkpoint run config to use one fixed source:
   `eval_data_dir_dict = {"unpacked": {"gs://.../unpacked": 1.0}}`.
4. Fax's existing `get_data_loaders()` and `GPTBalancedDataLoader` pad, label,
   batch, and shard those rows without packing documents or emitting
   continuation chunks.
5. Tax runs `forward_with_hooks_step` and either exports aligned arrays or
   immediately updates online entropy accumulators.
6. For paired-backend checks, `scripts/export_eval_token_sample.py` exports the
   exact rows emitted by the configured Fax loader. Hugging Face consumes this
   token artifact directly, so no second tokenizer or sampling path can alter
   the input.

This route requires no changes to `fax_mono`, no new Tax dataloader, no literal
prompt, and no custom model-side batcher. Dataset selection remains a run-config
override rather than model inference code.

The repositories remain independently versioned. Offline mode writes portable
numeric NPZ shards plus a JSON manifest for `soft_h` to read. Online mode stages
an exact hash-identified copy of the `soft_entropy` package into the Tax image;
it does not add a permanent Tax package dependency or modify `fax_mono`.

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

Those baseline integration changes are committed. Current paired-comparison
working-tree changes are listed at the end of this handoff.

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

## Current validated workflow and results

### C4 activation extraction

The controlled input begins with pinned English C4 validation source samples and
the immutable tokenizer-specific unpacked TFRecord artifact created from them.
Each variable-length row corresponds to one source sample and is capped at 512
tokens. Fax's existing unpacked evaluation path pads, labels, batches, and
shards the rows. It never combines samples or emits a second chunk for a long
sample. With evaluation batch size 2, 10,000 paper samples are exactly 5,000
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

### Five-checkpoint 30B trajectory

Five late-pretraining Baby Houselight checkpoints were evaluated at steps
`481500`, `483500`, `485500`, `487500`, and `489470`. Every run used the same
5,000-sample C4 prefix, 1,250 batches of size 4, all 48 block outputs, unigram
labels, 100 bins, seed 0, and four GB200s on the SMIFS queue. Each hook
accumulated 1,809,250 valid token positions.

The first submissions all failed before model initialization because each
foundation checkpoint's 13.66 GB `metadata.json` expanded beyond the Ray head's
67 GiB memory limit. Tax now loads only `run_config.json` and
`arch_config.json`, applies the inference overrides, and binds the checkpoint
weight path afterward. This avoids deserializing training-only metadata without
changing `fax_mono`. Unit tests and mandatory CPU preflights passed before all
five jobs were resubmitted; all five reruns succeeded.

Layer-mean trajectories were:

- `H(Z)`: `0.90782 → 0.90585 → 0.90559 → 0.90375 → 0.90319`;
- `I(X;Z)`: `0.27857 → 0.27522 → 0.27535 → 0.27405 → 0.27374`;
- `I(Y;Z)`: `0.23368 → 0.23180 → 0.23143 → 0.23056 → 0.23008`;
- `I(Y;Z)/I(X;Z)`: `0.83886 → 0.84226 → 0.84050 → 0.84131 → 0.84050`.

From first to last checkpoint, entropy fell 0.51%, input information fell 1.73%
in 47/48 layers, output information fell 1.54% in 47/48 layers, and
`I(X;Z)-I(Y;Z)` shrank about 2.7%. This is consistent with late-training
compression in the paper, but optimality increased only 0.20% and was
non-monotonic. These checkpoints span only the final approximately 1.6% of
optimizer steps, so they cannot reproduce the paper's full fitting-to-compression
trajectory. Earlier ancestral checkpoints are required for that experiment.

### Public-HF versus internal-Cohere sanity check

The current objective is to verify that the Tax/Fax entropy pipeline is faithful
to the previous Hugging Face `soft_h` pipeline, not merely that each pipeline
runs independently.

The paired model identity is:

- internal alias: `c3_7B_12-2024_command_release`;
- internal checkpoint:
  `gs://cohere-command/models_experimental/eugenecho/checkpoints/r7b/ckpt-499`;
- public model: `CohereLabs/c4ai-command-r7b-12-2024`;
- pinned public revision:
  `4f3d0aa6856e322f2f4480fe65420d5d53d297b8`;
- architecture: Cohere2, 32 layers, hidden size 4,096, vocabulary 256,000;
- inference precision: BF16 on both paths, with no FP8 or integer weight
  quantization.

#### Logit identity gate

Tax's `scripts/extract_logits.py` and
`soft_h/examples/extract_hf_logits.py` ran the two models on the exact same
Fax-exported token IDs. `soft_h/examples/compare_logits.py` compared full
256,000-way pre-softmax distributions. Across seven valid prediction positions:

- argmax agreement: 100%;
- mean cosine similarity: 0.999975;
- mean centered cosine similarity: 0.999966;
- mean absolute logit error: 0.00944;
- mean KL divergence: 0.000594;
- mean top-10 overlap: 97.1%.

An eighth selected position was excluded because its next-token target was
padding and the corresponding Fax prediction is not semantically valid. The
exporter and comparator now preserve the loader prediction mask and exclude such
positions. The logit gate establishes numerical model equivalence within
expected BF16 and backend differences.

#### Entropy agreement gate

The HF runner is a narrow adapter around the original `SoftEntropyAccumulator`
numerical core. It aligns raw block outputs and exact Fax token labels without
forking or reimplementing the entropy estimator.

The paired entropy protocol is:

- first 100 samples emitted by the frozen Fax unpacked C4 loader;
- sequence length 512 and batch size 2;
- all 32 raw transformer block outputs;
- 30,682 valid current/next-token positions per layer;
- 100 bins, seed 0, unigram input/output labels;
- NumPy `SoftEntropyAccumulator` backend on both sides, producing identical
  random reference directions;
- exact packaged `soft_h` source hash
  `57fad313eab09f5344b6182881286d18fdb352ee200e45d6b039e155bb364d78`;
- token-row hash
  `552c0eb49fcbeb38ee893df8837a926f753ac0d38fd97d4459008ffe2ff2bfdd`;
- Fax data-prefix hash
  `a11e6b09877f33bf332a76666e1f0a1b75feeda42714a674f1a2fa86ad38b0d0`.

`tax/scripts/export_eval_token_sample.py` exports the exact loader rows to:

```text
gs://cohere-dev/michael-rizvi/soft_h/entropy_comparison/command_r7b_n100_seq512_seed0/tokens/tokens.npz
```

This replaced the original plan to tokenize public C4 independently. The
internal and public tokenizers agreed on BPE content but appended different
special-token suffixes, so independently tokenizing the same text would have
introduced a real data mismatch. Both entropy jobs now consume the Fax token
artifact, and both results record the token, data-prefix, and source-package
hashes.

New paired-entropy components are:

- `tax/scripts/export_eval_token_sample.py`: exports exact Fax-loader sequences
  and hashes the complete loader batch prefix;
- `soft_h/examples/extract_hf_entropy.py`: loads the pinned public causal LM,
  captures raw block outputs, and updates one online accumulator per layer;
- `soft_h/examples/compare_entropy_results.py`: validates experiment identity
  and compares every layer and metric;
- `soft_h/tests/test_entropy_comparison.py`: proves that the HF and Tax
  accumulation adapters return identical metrics for identical activations;
- `soft_h/examples/kjobs-hf-entropy-comparison.yaml`: one-GPU HF job resources;
- `soft_h/examples/probe_hf_entropy_environment.py`: zero-GPU runtime dependency
  and artifact-access probe.

The comparison gates `H(Z)`, input/output mutual information, and
input/output regularity at mean absolute error at most 0.002, maximum per-layer
error at most 0.01, and layer-trajectory correlation at least 0.999.
Optimality is reported but not used as a hard gate because its ratio can become
unstable when input mutual information is near zero.

Verification before launch:

- Tax: 11 focused tests passed;
- `soft_h`: 31 focused tests passed;
- Fax CPU preflight passed with 32 hooks, batch size 2, sequence length 512,
  and TP=1;
- token artifact shape, source ordering, token hash, and Fax data-prefix hash
  were validated;
- synthetic identical-activation tests produced identical Tax and HF metrics.

Cluster status:

- internal Fax entropy job
  `vm-michael-rizvi-fax-202608262145-mhrx` succeeded and wrote
  `.../command_r7b_n100_seq512_seed0/fax/results.json`;
- public HF job `vm-soft-h-entropy-comparison-202608262145-ovel` failed before
  loading model weights because the vLLM image lacked SciPy;
- zero-GPU dependency probe
  `vm-soft-h-entropy-comparison-202608262207-duwf` subsequently succeeded.

The successful probe installed and pinned `gcsfs==2026.8.0` and
`scipy==1.18.1`, imported every runtime dependency, read and hash-checked the
real GCS token artifact, ran the SciPy-backed entropy accumulator, and loaded
the authenticated pinned Hugging Face model config. It also verified Torch,
Transformers, Accelerate, Safetensors, Hugging Face Hub, NumPy, fsspec, Google
authentication/storage, Requests, and aiohttp. The base image has unrelated
`pip check` warnings for PyGObject/PyCairo and vLLM's aiohttp constraint; neither
package path is used by this Transformers-based job, and real GCS I/O passed.

The image is immutable and pods are ephemeral, so every HF submission must
install the two pinned missing packages at startup.

#### Numerics and tokenizer provenance

A cross-backend entropy difference has two possible causes: the extraction and
accumulation harness, which is the subject of the test, or the numerics of the
forward pass that produced the activations. The first artifacts recorded neither
the attention kernel nor the quantization settings, so neither a pass nor a
failure was fully attributable. Both sides now record and enforce them.

Tax writes `attention_impl`, `quantize_params`, `quantize_activations`,
`quantize_residuals`, and `use_fp8_gemm` into `run_metadata`, and therefore into
`results.json`, and into `run_fingerprint`. Existing accumulator checkpoints
predating this change no longer resume, by design. Recording the value Fax
resolves at run time is necessary rather than cosmetic: this checkpoint's own
config migrates `ops_implementation_set='gpu_pretraining'` to
`attention_impl='fax_fa3'`, and only the recorded field establishes that the
`configs/sample.logit_compare.run.fragment` override to `jax_native` prevailed
on the accelerator.

`soft_h/examples/extract_hf_entropy.py` now takes `--token-artifact-dir` and
reads `token_ids_sha256`, `data_prefix_sha256`, and `tokenizer_path` from the
artifact's own `manifest.json` rather than from repeated command-line digests.
It loads the public tokenizer and asserts `pad=0`, `bos=5`, `eos=255001`,
asserts that the highest token identifier in the artifact lies inside the
256,000-token public vocabulary, and records the observed parameter dtype and
`model.config._attn_implementation` rather than the requested values. The
vocabulary bound is the substantive cross-tokenizer check, because both sides
report the same `tokenizer_path` by construction.

`soft_h/examples/compare_entropy_results.py` refuses to compare artifacts whose
numerics fall outside per-side allowlists, namely `jax_native` with all three
quantization flags enabled and FP8 disabled on the Fax side, and `eager` with
`torch.bfloat16` on the Hugging Face side, and requires both sides to report the
same tokenizer.

`soft_h/examples/submit_tax_activations.sh` exposes `--sample-config-path` and
derives the reported attention kernel from the selected fragment instead of
restating a hardcoded default. The launcher still accepts only
`s3://us-east-01a/` checkpoints, so the Command R7B runs described here are
submitted directly through `kjobs fax submit`.

#### Entropy agreement gate result

The gate passed. Jobs `vm-michael-rizvi-fax-202608262321-jmnz` and
`vm-soft-h-entropy-comparison-202608262324-1lvn` both succeeded on
`cw-us-east-04-prod`, writing
`.../command_r7b_n100_seq512_seed0/fax_numerics_v2/results.json` and
`.../command_r7b_n100_seq512_seed0/hf/results.json`. The comparison is durable
at `.../command_r7b_n100_seq512_seed0/comparison.json`.

Both sides selected 30,682 positions per layer across 32 layers. The Fax side
recorded `attention_impl='jax_native'`, all three quantization flags enabled,
and FP8 disabled. The Hugging Face side recorded `eager` attention and
`torch.bfloat16` parameters.

| Metric | Mean absolute error | Maximum absolute error | Layer correlation |
| --- | --- | --- | --- |
| `H(Z)` | 0.000238 | 0.000515 | 0.999990 |
| `I(X;Z)/input_unigram` | 0.000071 | 0.000277 | 0.999998 |
| `I(X;Z)/output_unigram` | 0.000032 | 0.000097 | 0.999998 |
| `regularity/input_unigram` | 0.000140 | 0.000412 | 0.999997 |
| `regularity/output_unigram` | 0.000038 | 0.000123 | 0.999996 |
| `optimality/unigram` (ungated) | 0.000185 | 0.000521 | 0.999995 |

Every gated metric clears its tolerance of 0.002 mean absolute error, 0.01
maximum absolute error, and 0.999 correlation by roughly an order of magnitude
or more. We therefore conclude that the Tax extraction and accumulation path
reproduces the previous Hugging Face `soft_h` pipeline on this model.

The re-run also reproduced the earlier `.../fax/results.json` exactly, with a
maximum per-layer difference of zero on every metric, which establishes that the
online path is deterministic and that the earlier artifact was produced under
the same configuration.

Two caveats remain. The Hugging Face job resolves Transformers from
`/app/cohere/transformers` inside the vLLM image rather than from the upstream
distribution, so the baseline is Fax against Cohere's Transformers fork; the
results do not yet record the Transformers version or distribution. Agreement is
established at 100 samples and unigram labels only, not at the full
10,000-sample protocol or for higher-order backoff.

### Scientific choices still required

The fixed protocol is English C4 validation, 10,000 frozen samples, 512-token
maximum context, all transformer block outputs, input/output unigram labels,
100 bins, and seed 0. Higher-order backoff requires explicit opt-in. The paper
does not specify its exact
split, sampler, activation hook, special-token policy, or seeds, so the artifact
and activation manifests record our explicit choices.

Random-model and CI-batch results are integration tests only.

## Working-tree state

Both repositories are on branch `tax-activation-integration`.

Committed baselines:

- `soft_h`: `48986c3` (`Add paper-aligned entropy and logit comparison`);
- Tax: `5779f1b58` (`feat: align entropy evaluation and add logit comparison`);
- checkpoint/resume: `7d73880` in `soft_h` and `f0ab51983` in Tax.

Current uncommitted source work:

- Tax modifies `scripts/extract_logits.py` and its tests, and adds
  `scripts/export_eval_token_sample.py` plus tests;
- Tax also modifies `scripts/extract_activations.py` to record the forward-pass
  numerics, and `tests/fax/extract_activations_test.py` to cover them and the
  logit-compare fragment;
- `soft_h` modifies the HF logit runner/comparator and tests, and adds the HF
  entropy runner, entropy comparator, job configuration, dependency probe, and
  entropy-comparison tests;
- `soft_h` also modifies `examples/analyze_tax_activations.py` for the extended
  manifest schema and `examples/submit_tax_activations.sh` for
  `--sample-config-path`;
- this handoff update is also uncommitted.

Test state after these changes: 28 Tax tests in
`tests/fax/extract_activations_test.py` and 51 `soft_h` tests in `tests/` pass.
`scripts/extract_logits.py` records four of the five numerics fields, omitting
`quantize_residuals`, so the logit gate retains a narrower form of the gap
closed above.

`soft_h/artifacts/` contains generated BLS analysis output and must remain
excluded from source commits. No commit or push has been performed for the
current paired-comparison work.
