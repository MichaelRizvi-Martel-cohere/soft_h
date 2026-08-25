"""Run Tax online entropy accumulation on native C4 with a tiny random Fax model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np

DEFAULT_C4_PARQUET = (
    "gs://cohere-dev/michael-rizvi/soft_h/c4/"
    "c4_en_validation_rev1588ec45_seed0_n10000_drydock/documents.parquet"
)


def _assert_results_are_finite(results: dict, hooks: tuple[str, ...]) -> None:
    for hook in hooks:
        hook_results = results["hooks"][hook]
        if hook_results["n_samples"] <= 0:
            raise AssertionError(f"Hook {hook} accumulated no samples.")
        for name, value in hook_results.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise AssertionError(f"Hook {hook} produced non-finite {name}={value}.")


def _assert_results_match(actual: dict, expected: dict, hooks: tuple[str, ...]) -> None:
    for hook in hooks:
        if actual["hooks"][hook]["n_samples"] != expected["hooks"][hook]["n_samples"]:
            raise AssertionError(f"Sample count differs for hook {hook}.")
        for name, value in actual["hooks"][hook].items():
            if isinstance(value, float) and not np.isclose(
                value,
                expected["hooks"][hook][name],
                rtol=1e-6,
                atol=1e-7,
            ):
                raise AssertionError(
                    f"Online and offline {name} differ for {hook}: "
                    f"{value} != {expected['hooks'][hook][name]}."
                )


def run(
    tax_repo: Path,
    dataset_parquet: str,
    output_dir: Path | None,
    n_batches: int = 1,
) -> dict:
    """Execute native C4 batches and compare online, resumed, and offline results."""
    if n_batches < 1:
        raise ValueError(f"n_batches must be positive, got {n_batches}.")
    soft_h_repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(soft_h_repo))
    sys.path.insert(0, str(tax_repo))
    sys.path.insert(0, str(soft_h_repo / "examples"))
    os.chdir(tax_repo)

    import fax.register.all  # noqa: F401
    from analyze_tax_activations import analyze
    from fax.data.data_registry import get_data_loaders
    from fax.minizord.ray.ray_manager import RayManager
    from fax.profiling.global_timer import GlobalTimer
    from fax.zord.inferencezord import InferenceZord
    from scripts.extract_activations import (
        _update_data_prefix_hash,
        load_online_checkpoint,
        prepare_export_arrays,
        write_manifest,
        write_npz_shard,
        write_online_checkpoint,
        write_results,
    )
    from soft_entropy.tax_online import TaxActivationAccumulator
    from tests.fax.hooks import load_config

    source = {"drydock": {dataset_parquet: 1.0}}
    hooks = ("block_0_block_output", "block_1_block_output")
    config = load_config(
        arch_overrides={"layer_switch": 2, "vocab_size": 50261},
        run_overrides={
            "output_dir": "/tmp/tax_online_entropy_tiny_random",
            "data_loader.name": "gpt_balanced_dataloader",
            "data_loader.kwargs": {},
            "data_dir_dict": source,
            "eval_data_dir_dict": source,
            "max_sequence_length": 32,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "n_validation_steps": n_batches,
            "quantize.params": False,
            "quantize.activations": False,
            "sharding.n_fsdp_parallel": 1,
        },
    )
    _, eval_loader, _ = get_data_loaders(config)
    batches = [next(eval_loader) for _ in range(n_batches)]

    with (
        RayManager(config=config, ray_init_timer=GlobalTimer()) as ray_manager,
        InferenceZord(config, ray_manager) as inference_zord,
    ):
        outputs = [
            inference_zord.forward_with_hooks_step(batch=batch, hooks=hooks).outputs
            for batch in batches
        ]
    activation_batches = [
        prepare_export_arrays(batch_outputs, batch, hooks)
        for batch_outputs, batch in zip(outputs, batches)
    ]

    accumulator = TaxActivationAccumulator(hooks, n_bins=16, seed=0)
    selected_samples = sum(accumulator.update(arrays) for arrays in activation_batches)
    accumulated_results = accumulator.results()
    _assert_results_are_finite(accumulated_results, hooks)

    output_context = (
        nullcontext(output_dir)
        if output_dir is not None
        else tempfile.TemporaryDirectory(prefix="tax_online_entropy_tiny_")
    )
    with output_context as managed_output:
        online_root = Path(managed_output)
        if output_dir is not None:
            online_root.mkdir(parents=True, exist_ok=False)
        final_results = accumulated_results
        final_checkpoint = None
        if n_batches > 1:
            split_index = n_batches // 2
            partial_accumulator = TaxActivationAccumulator(hooks, n_bins=16, seed=0)
            partial_selected = sum(
                partial_accumulator.update(arrays)
                for arrays in activation_batches[:split_index]
            )
            data_prefix_digest = hashlib.sha256()
            for batch in batches[:split_index]:
                _update_data_prefix_hash(data_prefix_digest, batch)
            _, replay_loader, _ = get_data_loaders(config)
            replay_digest = hashlib.sha256()
            for _ in range(split_index):
                _update_data_prefix_hash(replay_digest, next(replay_loader))
            if replay_digest.hexdigest() != data_prefix_digest.hexdigest():
                raise AssertionError(
                    "A fresh Drydock loader did not reproduce the checkpoint prefix."
                )
            run_fingerprint = {
                "checkpoint": "random_init:tiny_mup",
                "hooks": list(hooks),
                "eval_data_path": dataset_parquet,
                "sequence_length": 32,
                "tokenizer_path": config.run.tokenizer_path,
                "n_bins": 16,
                "seed": 0,
                "soft_h_package_sha256": "integration-test",
            }
            final_checkpoint = write_online_checkpoint(
                output_dir=str(online_root),
                completed_batches=split_index,
                total_samples=sum(
                    int(arrays["input_token"].shape[0])
                    for arrays in activation_batches[:split_index]
                ),
                total_selected_samples=partial_selected,
                data_prefix_sha256=data_prefix_digest.hexdigest(),
                run_fingerprint=run_fingerprint,
                accumulator_state=partial_accumulator.state_dict(),
            )
            restored_state, completed, _, restored_selected, _ = load_online_checkpoint(
                str(online_root),
                final_checkpoint,
                run_fingerprint,
            )
            if completed != split_index or restored_selected != partial_selected:
                raise AssertionError("Checkpoint counters did not round-trip.")
            resumed_accumulator = TaxActivationAccumulator(hooks, n_bins=16, seed=0)
            resumed_accumulator.load_state_dict(restored_state)
            for arrays in activation_batches[split_index:]:
                resumed_accumulator.update(arrays)
            final_results = resumed_accumulator.results()
            _assert_results_match(final_results, accumulated_results, hooks)

        total_samples = sum(
            int(arrays["input_token"].shape[0]) for arrays in activation_batches
        )
        results = {
            "schema_version": 2,
            "checkpoint": "random_init:tiny_mup",
            "mode": "online_entropy",
            "eval_data_path": dataset_parquet,
            "sequence_length": 32,
            "n_bins": 16,
            "seed": 0,
            "n_batches": n_batches,
            "total_samples": total_samples,
            "total_selected_samples": selected_samples,
            "final_checkpoint": final_checkpoint,
            **final_results,
        }
        write_results(str(online_root), results)
        expected_files = (
            {"results.json", "checkpoints"} if n_batches > 1 else {"results.json"}
        )
        if {path.name for path in online_root.iterdir()} != expected_files:
            raise AssertionError("Online mode persisted unexpected files.")

        with tempfile.TemporaryDirectory(
            prefix="tax_entropy_offline_reference_"
        ) as tmp:
            reference_root = Path(tmp)
            shard_specs = []
            total_sequences = 0
            for batch_index, arrays in enumerate(activation_batches):
                shard_name = write_npz_shard(
                    str(reference_root),
                    batch_index,
                    arrays,
                )
                sequence_keys = np.stack(
                    [arrays["batch_row"], arrays["sequence_id"]], axis=-1
                )
                total_sequences += int(np.unique(sequence_keys, axis=0).shape[0])
                shard_specs.append(
                    {
                        "file": shard_name,
                        "n_samples": int(arrays["input_token"].shape[0]),
                        "hook_dimensions": {
                            hook: int(arrays[f"activation__{hook}"].shape[-1])
                            for hook in hooks
                        },
                    }
                )
            write_manifest(
                str(reference_root),
                {
                    "schema_version": 2,
                    "checkpoint": "random_init:tiny_mup",
                    "hooks": list(hooks),
                    "activation_keys": {hook: f"activation__{hook}" for hook in hooks},
                    "activation_dtype": "float32",
                    "label_keys": [
                        "input_token",
                        "output_token",
                        "batch_row",
                        "sequence_id",
                        "position",
                    ],
                    "input_source": "fixed_drydock_eval",
                    "eval_data_path": dataset_parquet,
                    "sequence_length": 32,
                    "tokenizer_path": config.run.tokenizer_path,
                    "shards": shard_specs,
                    "total_samples": total_samples,
                    "total_sequences": total_sequences,
                },
            )
            offline_results = analyze(str(reference_root), n_bins=16, seed=0)
        _assert_results_match(results, offline_results, hooks)

        rendered = json.dumps(results, indent=2, sort_keys=True)
        if output_dir is not None:
            print(f"Wrote {online_root / 'results.json'}")
        print(rendered)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tax-repo",
        type=Path,
        default=Path.home() / "repos" / "tax",
    )
    parser.add_argument("--dataset-parquet", default=DEFAULT_C4_PARQUET)
    parser.add_argument(
        "--n-batches",
        type=int,
        default=1,
        help="Number of C4 batches to run; values above one exercise checkpoint/resume.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional new directory in which to retain results.json.",
    )
    args = parser.parse_args()

    tax_repo = args.tax_repo.resolve()
    if not (tax_repo / "scripts" / "extract_activations.py").is_file():
        raise FileNotFoundError(f"Tax checkout not found at {tax_repo}.")
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}.")
    run(tax_repo, args.dataset_parquet, args.output_dir, args.n_batches)


if __name__ == "__main__":
    main()
