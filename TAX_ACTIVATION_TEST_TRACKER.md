# Tax Activation Entropy Test Tracker

Last updated: 2026-08-25

## Goal

Estimate representation entropy for Cohere checkpoints on external C4 data
without persisting prohibitively large activation datasets.

The current path is:

1. Tax loads a Fax checkpoint.
2. A Tax preparation script tokenizes each frozen C4 document independently,
   truncates it once, and writes one native unpacked TFRecord row.
3. Tax captures selected model hooks during inference.
4. `soft_h` accumulates soft-bin and conditional n-gram counts online.
5. Raw activations are discarded after each batch.
6. Tax writes `results.json` and optional numeric accumulator checkpoints.

Model inference is JAX/Fax. Online entropy accumulation currently runs with
NumPy on the Ray head.

## Reference configuration

The validated 30B configuration uses:

- Checkpoint: BLS 30B checkpoint 536
- GPUs: 8 H100s
- Sharding: TP=4, FSDP=2
- Evaluation batch size: 2
- Sequence length: 512
- Hook: `block_0_block_output`
- Dataset: tokenizer-specific unpacked TFRecord with one row per C4 document
- Bins: 100
- Seed: 0
- Small tests: `fax-ci-queue`

C4 source documents:

`gs://cohere-dev/michael-rizvi/soft_h/c4/c4_en_validation_rev1588ec45_seed0_n10000_drydock/documents.jsonl`

## Completed implementation

- [x] Replaced the custom Tax JSONL batching path with Fax's native Drydock
  loader.
- [x] Resolved the paper's sample unit as one independently tokenized C4
  document with maximum context 512.
- [x] Added tokenizer-specific unpacked TFRecord preparation: one source
  document becomes one truncated Fax row, with no packing or continuation
  chunks.
- [x] Made the launcher derive 5,000 batches from 10,000 documents at evaluation
  batch size 2.
- [x] Added aligned token, sequence, row, and position metadata to Tax exports.
- [x] Added online entropy accumulation through `TaxActivationAccumulator`.
- [x] Made token-level unigram backoff the default and added explicit
  `label_types` selection for higher-order analyses.
- [x] Made evaluation batch size override the final Fax generation fragment and
  added a resolved-config assertion; 30B CPU preflight produced shape `(4, 512)`.
- [x] Added a Tax-local inference config loader that binds the model weights only
  after loading the small run and architecture configs. This avoids
  deserializing foundation-checkpoint training metadata, including the 13.66 GB
  serialized prefetched batch, without changing `fax_mono`.
- [x] Added reproducible packaging of the local `soft_h` source into the Tax
  image.
- [x] Added periodic numeric accumulator checkpoints.
- [x] Added strict checkpoint schema and run-fingerprint validation.
- [x] Added deterministic Drydock-prefix verification during resume.
- [x] Added explicit `--resume-from` launcher support.
- [x] Retained only the newest complete accumulator checkpoint.

## Completed local tests

### Unit tests

- [x] Sequential online updates agree with one global update.
- [x] Multiple hooks accumulate independently.
- [x] Online and offline entropy metrics agree numerically.
- [x] Checkpoint state round-trips without pickle or object arrays.
- [x] Incompatible seeds and configurations are rejected.
- [x] Corrupted or negative counts are rejected.
- [x] Incomplete checkpoints are rejected.
- [x] Changed evaluation rows alter the data-prefix hash.
- [x] Resume skips completed model forwards.
- [x] One-token tuple labels remain tuples across checkpoint restoration.
- [x] Unpacked preparation preserves one row per document and truncates each row
  once.
- [x] Config loading passes `state_ckpt=None`, binds the resolved checkpoint
  afterward, and rejects legacy pre-GDA checkpoints explicitly.

### Tiny random-model integration tests

- [x] Native C4 data passes through a two-layer random Fax model.
- [x] Online accumulation agrees with NPZ-based offline analysis.
- [x] Twenty batches complete with finite metrics.
- [x] A split checkpoint/restore run agrees with uninterrupted accumulation.
- [x] A fresh Drydock loader reproduces the consumed data prefix.

The latest paper-aligned rerun used 80 C4 samples over 20 unpacked batches. The
random two-layer model processed 2,491 valid unigram positions per hook. Online
accumulation, checkpoint/restore, deterministic loader replay, and offline NPZ
analysis agreed.

This stress test exposed and fixed the scalar-versus-tuple unigram-label
restoration bug.

## Completed 30B tests

### Raw activation gate

Job: `vm-michael-rizvi-fax-202608241920-ub6y`

- [x] Loaded checkpoint 536.
- [x] Completed one C4 batch.
- [x] Extracted `block_0_block_output`.
- [x] Wrote and analyzed an NPZ activation shard.

### One-batch online equivalence gate

Job: `vm-michael-rizvi-fax-202608242053-jk6j`

- [x] Completed one C4 batch online.
- [x] Discarded raw activations.
- [x] Wrote `results.json`.
- [x] Reproduced the raw/NPZ entropy result within numerical tolerance.

### 500-row online run

Job: `vm-michael-rizvi-fax-202608242111-phdn`

- [x] Completed 250 batches, corresponding to 500 packed rows.
- [x] Produced online entropy results without activation shards.
- [x] Completed without restarts.

### 500-row checkpoint-writing run

Job: `vm-michael-rizvi-fax-202608242143-p6ek`

- [x] Completed all 250 batches without restarts.
- [x] Successfully wrote periodic checkpoints.
- [x] Produced `results.json`.
- [x] Retained a complete final `batch_00250` checkpoint.
- [ ] Did not test on-cluster restoration because the run finished before it
  was deliberately interrupted.

Output:

`gs://cohere-dev/michael-rizvi/soft_h_tax/bls30b_ckpt536_c4_block0_rows500_b250_seq512_resume_test_v1`

### GB200 compatibility gate

Job: `vm-michael-rizvi-fax-202608251414-ggkr`

- [x] Used one node with four GB200s, TP=4, FSDP=1, and FA4.
- [x] Avoided the prior single-GPU-worker NCCL device-ordinal failure.
- [x] Completed one online C4 batch and accumulated 1,012 entropy positions.
- [x] Wrote `results.json`.

This gate used the earlier Drydock artifact. The next 30B gate must use the new
paper-aligned unpacked artifact.

## Remaining validation

### Highest priority

- [ ] Compare the two completed 500-row `results.json` files and confirm
  checkpointing did not alter metrics.
- [ ] Run a genuine on-cluster interruption/resume test:
  1. Start a fresh run with frequent checkpointing.
  2. Wait for a complete checkpoint.
  3. Cancel the job.
  4. Resume from that exact checkpoint.
  5. Confirm prefix replay succeeds.
  6. Confirm completed batches do not repeat model inference.
  7. Compare final metrics against an uninterrupted run.

### Before full scientific runs

- [x] Define "10,000 samples": 10,000 C4 source documents, each independently
  tokenized and truncated to at most 512 tokens.
- [x] Build and upload the 10,000-sample unpacked artifact with the BLS
  checkpoint tokenizer.
- [x] Run CPU preflight on the foundation checkpoint with all 48 hooks, batch
  size 4, sequence length 512, and unigram labels. It completed in about 16
  seconds instead of exhausting Ray-head memory.
- [ ] Run a one-batch 30B gate against the unpacked artifact before scaling.
- [ ] Select the model checkpoints from `mid-train-experiments`.
- [ ] Select the complete set of layers/hooks.
- [ ] Estimate runtime and Ray-head memory growth for the requested hooks.
- [ ] Move larger runs from `fax-ci-queue` to the appropriate team queue.
- [ ] Run one intermediate multi-hook or multi-model gate before the full
  sweep.
- [ ] Establish run naming and output paths that encode checkpoint, hook set,
  bin count, seed, and dataset version.

## Known limitations

- Activations are discarded after accumulation, but each hooked batch is still
  transferred from Fax workers to the Ray head.
- Resume reloads and recompiles the model.
- Resume is explicit rather than automatic.
- Only the newest complete accumulator checkpoint is retained.
- The metadata-free config loader supports modern post-GDA checkpoints. Legacy
  pre-GDA checkpoints still require Fax's metadata-dependent loading path and
  are rejected.
- The 30B tests currently cover one hook.
- Historical 500-row and GB200 gates used Drydock and are integration evidence,
  not paper-aligned scientific runs.

## Updating this tracker

Every test entry should record:

- Job name or local command
- Exact configuration
- Output location
- Pass/fail result
- Any discrepancy or bug found
- Whether the result is an integration check or a scientific measurement
