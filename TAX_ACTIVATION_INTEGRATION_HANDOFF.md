# Tax activation extraction to soft-entropy: project handoff

Last updated: 2026-08-21

## Goal

Use Fax/Tax to load a model checkpoint, run model inference with intermediate
activation hooks, export aligned activation and token-label tensors, and consume
those tensors in `soft_h` to estimate soft entropy and mutual information.

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

### Real BLS activation extraction

After generation succeeds:

1. Run `scripts/extract_activations.py` for one evaluation batch and one hook,
   preferably `block_0_block_output`.
2. Keep scanned forward passes and stacked-at-init parameters disabled.
3. Write to a unique durable object-storage path rather than `/tmp`.
4. Validate the manifest and shard shapes.
5. Run the `soft_h` consumer with a fixed seed.
6. Only then add more layers/hooks or batches.

Hooks retain full `[B, S, D]` representations before masking/export and may need
more memory than generation alone. TP=8 is therefore a generation baseline,
not a guarantee that many simultaneous BLS hooks will fit.

The current extractor consumes the checkpoint's evaluation dataloader. It does
not yet accept literal prompt strings. Add prompt-based input only if needed;
the dataset path is the simplest route for entropy/MI experiments.

### Scientific choices still required

Before interpreting metrics, decide:

- which dataset and split to use;
- which token positions to include;
- whether labels should be input tokens, next tokens, sequence/task labels, or
  another experimental variable;
- which layers to compare;
- a fixed `soft_h` seed and bin count;
- sample counts sufficient for stable estimation.

Random-model and CI-batch results are integration tests only.

## Working-tree state

Nothing has been committed or pushed.

Expected Tax changes:

```text
 M tests/fax/hooks.py
?? scripts/extract_activations.py
?? tests/fax/extract_activations_test.py
```

Expected `soft_h` changes before this handoff file:

```text
 M soft_entropy/__init__.py
?? examples/analyze_tax_activations.py
?? tests/test_analyze_tax_activations.py
?? uv.lock
```

Review the untracked `uv.lock` before including it in a commit.
