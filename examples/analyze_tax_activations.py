"""Compute soft entropy and token mutual information from Tax activation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from soft_entropy.accumulator import SoftEntropyAccumulator

_SCHEMA_VERSION = 1
_MANIFEST_FIELDS = {
    "schema_version",
    "checkpoint",
    "hooks",
    "activation_keys",
    "activation_dtype",
    "label_keys",
    "shards",
    "total_samples",
}
_SHARD_FIELDS = {"file", "n_samples", "hook_dimensions"}


def _load_manifest(activation_dir: Path) -> dict[str, Any]:
    manifest_path = activation_dir / "manifest.json"
    with manifest_path.open() as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("Activation manifest must be a JSON object.")

    missing = _MANIFEST_FIELDS - manifest.keys()
    unexpected = manifest.keys() - _MANIFEST_FIELDS
    if missing or unexpected:
        raise ValueError(
            f"Invalid manifest fields: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    if manifest["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema {manifest['schema_version']!r}; expected {_SCHEMA_VERSION}."
        )
    if not isinstance(manifest["hooks"], list) or not manifest["hooks"]:
        raise ValueError("Manifest hooks must be a non-empty list.")
    if set(manifest["activation_keys"]) != set(manifest["hooks"]):
        raise ValueError(
            "Manifest activation_keys must contain exactly one entry per hook."
        )
    if not isinstance(manifest["shards"], list) or not manifest["shards"]:
        raise ValueError("Manifest shards must be a non-empty list.")
    for shard in manifest["shards"]:
        if not isinstance(shard, dict) or set(shard) != _SHARD_FIELDS:
            raise ValueError(f"Invalid shard entry: {shard!r}.")
    return manifest


def _resolve_shard(activation_dir: Path, shard_name: str) -> Path:
    if not isinstance(shard_name, str) or not shard_name:
        raise ValueError(f"Invalid shard filename {shard_name!r}.")
    activation_dir = activation_dir.resolve()
    shard_path = (activation_dir / shard_name).resolve()
    if not shard_path.is_relative_to(activation_dir):
        raise ValueError(f"Shard path escapes activation directory: {shard_name!r}.")
    return shard_path


def analyze(
    activation_dir: str,
    n_bins: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Stream an exported Tax activation dataset into one accumulator per hook."""
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}.")

    root = Path(activation_dir)
    manifest = _load_manifest(root)
    hooks = manifest["hooks"]
    accumulators: dict[str, SoftEntropyAccumulator] = {}
    sample_counts = {hook: 0 for hook in hooks}

    for shard_spec in manifest["shards"]:
        shard_path = _resolve_shard(root, shard_spec["file"])
        with np.load(shard_path, allow_pickle=False) as shard:
            required_keys = {
                "input_token",
                "output_token",
                *(manifest["activation_keys"][hook] for hook in hooks),
            }
            missing_keys = required_keys - set(shard.files)
            if missing_keys:
                raise ValueError(
                    f"Shard {shard_path} is missing keys {sorted(missing_keys)}."
                )

            input_token = shard["input_token"]
            output_token = shard["output_token"]
            if input_token.ndim != 1 or output_token.shape != input_token.shape:
                raise ValueError(
                    f"Shard labels must be aligned 1-D arrays, got {input_token.shape} and {output_token.shape}."
                )

            for hook in hooks:
                activation = shard[manifest["activation_keys"][hook]]
                if activation.ndim != 2 or activation.shape[0] != input_token.shape[0]:
                    raise ValueError(
                        f"Activation {hook!r} in {shard_path} must be [N, D] and align with labels; "
                        f"got {activation.shape} and {input_token.shape}."
                    )
                if not np.isfinite(activation).all():
                    raise ValueError(
                        f"Activation {hook!r} in {shard_path} contains non-finite values."
                    )
                if np.any(np.linalg.norm(activation, axis=-1) == 0):
                    raise ValueError(
                        f"Activation {hook!r} in {shard_path} contains zero-norm rows."
                    )

                if hook not in accumulators:
                    accumulators[hook] = SoftEntropyAccumulator(
                        d=activation.shape[-1],
                        n_bins=n_bins,
                        seed=seed,
                        backend="numpy",
                    )
                elif accumulators[hook].w.shape[-1] != activation.shape[-1]:
                    raise ValueError(
                        f"Activation dimension changed across shards for hook {hook!r}."
                    )

                accumulators[hook].update(
                    activation,
                    labels={"input_token": input_token, "output_token": output_token},
                )
                sample_counts[hook] += activation.shape[0]

    expected_samples = manifest["total_samples"]
    if any(count != expected_samples for count in sample_counts.values()):
        raise ValueError(
            f"Sample counts {sample_counts} do not match manifest total {expected_samples}."
        )

    return {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": manifest["checkpoint"],
        "n_bins": n_bins,
        "seed": seed,
        "hooks": {
            hook: {
                **accumulators[hook].results(),
                "dimension": int(accumulators[hook].w.shape[-1]),
                "n_samples": sample_counts[hook],
            }
            for hook in hooks
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "activation_dir", help="Directory containing manifest.json and NPZ shards."
    )
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; prints to stdout otherwise.",
    )
    args = parser.parse_args()

    results = analyze(args.activation_dir, n_bins=args.n_bins, seed=args.seed)
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
