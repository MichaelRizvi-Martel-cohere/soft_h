# Tax activation SweepConfig plan

Status: deferred. The current Bash/kjobs launcher remains the supported path
until this plan is implemented. The missing adapter described below was
rechecked against the latest shared Sweep code on 2026-08-24.

This plan does not authorize edits to Tax or Sweep. Treat both repositories as
shared dependencies. Before implementation, obtain Michael's explicit approval
for the proposed diffs in each repository. In particular, do not modify
`tax/fax_mono/`; prefer existing Tax APIs and configuration. See
`TAX_ACTIVATION_EXPERIMENTS.md` for the full Tax ownership and edit policy.

## Decision

Add a reusable, argv-based `TaxCommandConfig` to the shared
`cohere-ai/sweep` package, then consume it from a thin Agent Expert activation
extraction SweepConfig in Tax.

Do not extend `FaxRunCommand`. `FaxConfig` is training-specific and assumes
run/architecture config generation. Sweep currently has no supported
arbitrary-command task.

## Latest Sweep validation

Checked `cohere-ai/sweep` `main` at commit
`81363aa5e84787f892098d0ddfccaea7158e7261` and the latest release,
`v1.17.2`, on 2026-08-24. There is still no concrete generic command adapter:

- the exported task types are the existing Fax, Bee, merge, deployment,
  export, TIF, and data-pipeline adapters;
- `ConfigStagesEnum` has no generic command stage;
- `TaskConfig` is an extension interface, not a usable command task. A
  subclass must still implement compute overrides, per-trial and array script
  generation, and image handling;
- `CommandDataConfig` configures local data preparation for Fax and does not
  launch an arbitrary Kubernetes command;
- no open or merged pull request or issue matching a generic/arbitrary command
  adapter was found.

Therefore the matrix, trial metadata, fanout, dependencies, and submission
should remain native Sweep behavior, while `TaxCommandConfig` supplies only
the missing task-to-command adapter. Before implementing it, check Sweep
`main` again and ask the Sweep maintainers whether they prefer a generic
repository-command adapter over a Tax-specific name.

## Architecture

```text
Tax activation SweepConfig
  -> sweep.TaxCommandConfig
  -> per-trial scripts and compute specs
  -> kjobs or TrainJob
  -> Kueue
  -> GCS activation shards
```

## 1. Add `TaxCommandConfig` to Sweep

In a local `cohere-ai/sweep` checkout:

- add `sweep/training/config_tax_command.py`;
- export the type through `sweep/__init__.py`;
- add `tax_command` to `ConfigStagesEnum`;
- add the required generic script writer under `sweep/utils/create_jobs.py`;
- add `sweep/training/config_tax_command_test.py`;
- add a minimal example at
  `examples/sweep_configs/single_tasks/tax_command.py`.

The task should:

- accept a non-empty `command: tuple[str, ...]`, never a raw shell string;
- render command arguments with `shlex.join`;
- reuse `TaskConfig` fields for repository, partition, queue, priority, time
  limit, retries, fanout, dependencies, environment, and compute patches;
- load Tax's existing `ops/kjobs-compute.yaml`;
- derive worker and GPU counts from `partition`;
- build one sweep-tagged Tax image shared by all trials;
- support both kjobs and TrainJob using the Tax Ray head/worker layout;
- optionally emit a heartbeat during quiet compilation while preserving the
  child process exit code.

Tests should cover validation, shell-safe argv rendering, kjobs and TrainJob
resource rendering, image tags, heartbeat exit propagation, dependencies, and
fanout.

## 2. Release and consume the Sweep feature

Tax currently pins `sweep==1.16.8` in `workspace/pyproject.toml`, while Sweep
main was at 1.17.2 when this plan was written.

After the shared Sweep change is reviewed and released:

1. bump Tax to the released Sweep version;
2. refresh the workspace lockfile with `uv`;
3. use `RepoConfig(directory="${HOME}/repos/tax", kind=RepoKinds.tax)`.

Do not set `RepoConfig.version` for Tax. Sweep currently rejects versioned Tax
repositories. The local checkout should produce a unique sweep image tag.

## 3. Add the activation extraction sweep in Tax

Add:

```text
workspace/agent_expert/standalone/extract_activations_sweep.py
```

Match existing Agent Expert standalone conventions:

- explicit tracked run specs containing model name, checkpoint, hooks, and
  batch count;
- no runtime dependency on the external checkpoint CSV;
- command:
  `uv run --no-sync python scripts/extract_activations.py`;
- sample fragment: `configs/sample.run.fragment`;
- cluster: `cw_us_east_04_prod`;
- queue: `post_training_smifs_queue`;
- priority: `dev_low`;
- partition: `gpu_8`;
- time limit: one hour;
- `jobs_max_fanout=1`;
- `retries=0`, because partial outputs are intentionally non-overwritable;
- `enable_wandb=False`.

Use `hyper.zip` so model name, checkpoint, and output stay aligned. Generate
unique outputs under:

```text
gs://cohere-dev/michael-rizvi/soft_h_tax/${SWEEP_UNAME}/
```

Keep soft_h analysis outside the sweep initially. The consumer is CPU-local
and currently accepts only local paths.

## 4. Workflow and verification

Document and follow:

```bash
python path/to/extract_activations_sweep.py diagnose
python path/to/extract_activations_sweep.py start
# Inspect generated commands, YAML, trial mapping, image behavior, queue,
# gpu_8 allocation, fanout, and output paths.
python path/to/extract_activations_sweep.py start --submit
```

`start --submit` requires fresh explicit approval.

Before a real matrix:

1. generate a two-trial synthetic preview;
2. inspect generated artifacts and `metadata.json`;
3. run one real checkpoint, one hook, and one batch;
4. require both successful cluster state and real output artifacts:
   `manifest.json` plus at least one `batch_XXXXX.npz`.

## Sources read

- Cookbook:
  `src/cookbook/site/recipes/00_components/sweep/`
- Sweep examples:
  `cohere-ai/sweep/examples/sweep_configs/`
- Sweep extension interface:
  `sweep/config_task.py`
- Closest concrete task:
  `sweep/deployment/config_hf_export.py`
- Current local launcher and documentation:
  `soft_h/examples/submit_tax_activations.sh`
  and `soft_h/examples/TAX_ACTIVATION_EXPERIMENTS.md`
