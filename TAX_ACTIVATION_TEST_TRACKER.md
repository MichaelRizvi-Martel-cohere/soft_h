# Tax Activation Entropy Test Tracker

Last updated: 2026-08-24

## Goal

Estimate representation entropy for Cohere checkpoints on external C4 data
without persisting prohibitively large activation datasets.

The current path is:

1. Tax loads a Fax checkpoint.
2. Fax/Drydock tokenizes and packs the external C4 Parquet dataset.
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
- Dataset: external C4 validation subset stored as Drydock-compatible Parquet
- Bins: 100
- Seed: 0
- Small tests: `fax-ci-queue`

C4 source:

`gs://cohere-dev/michael-rizvi/soft_h/c4/c4_en_validation_rev1588ec45_seed0_n10000_drydock/documents.parquet`

## Completed implementation

- [x] Replaced the custom Tax JSONL batching path with Fax's native Drydock
  loader.
- [x] Added aligned token, sequence, row, and position metadata to Tax exports.
- [x] Added online entropy accumulation through `TaxActivationAccumulator`.
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

### Tiny random-model integration tests

- [x] Native C4 data passes through a two-layer random Fax model.
- [x] Online accumulation agrees with NPZ-based offline analysis.
- [x] Twenty batches complete with finite metrics.
- [x] A split checkpoint/restore run agrees with uninterrupted accumulation.
- [x] A fresh Drydock loader reproduces the consumed data prefix.

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

- [ ] Define "10,000 samples" precisely: C4 documents, packed rows, or selected
  token positions.
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
- The 30B tests currently cover one hook.
- The current 500-row count refers to packed evaluation rows, not necessarily
  500 original C4 documents.

## Updating this tracker

Every test entry should record:

- Job name or local command
- Exact configuration
- Output location
- Pass/fail result
- Any discrepancy or bug found
- Whether the result is an integration check or a scientific measurement
