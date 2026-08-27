"""Compute paper-aligned soft entropy and n-gram MI from Tax activation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from soft_entropy.tax_online import (
    DEFAULT_LABEL_TYPES,
    TaxActivationAccumulator,
    parse_label_types,
)

_SCHEMA_VERSION = 2
_REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "checkpoint",
    "hooks",
    "activation_keys",
    "activation_dtype",
    "label_keys",
    "shards",
    "total_samples",
    "input_source",
    "eval_data_path",
    "sequence_length",
    "tokenizer_path",
    "total_sequences",
}
_OPTIONAL_MANIFEST_FIELDS = {
    "eval_data_type",
    "label_types",
    "attention_impl",
    "quantize_params",
    "quantize_activations",
    "quantize_residuals",
    "use_fp8_gemm",
}
_SHARD_FIELDS = {"file", "n_samples", "hook_dimensions"}


def _load_manifest(activation_dir: Path) -> dict[str, Any]:
    manifest_path = activation_dir / "manifest.json"
    with manifest_path.open() as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise TypeError("Activation manifest must be a JSON object.")

    allowed_fields = _REQUIRED_MANIFEST_FIELDS | _OPTIONAL_MANIFEST_FIELDS
    missing = _REQUIRED_MANIFEST_FIELDS - manifest.keys()
    unexpected = manifest.keys() - allowed_fields
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
    label_types: str | tuple[str, ...] | list[str] = DEFAULT_LABEL_TYPES,
) -> dict[str, Any]:
    """Stream an exported Tax activation dataset into one accumulator per hook."""
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}.")

    root = Path(activation_dir)
    manifest = _load_manifest(root)
    hooks = manifest["hooks"]
    parsed_label_types = parse_label_types(label_types)
    accumulator = TaxActivationAccumulator(
        hooks,
        n_bins=n_bins,
        seed=seed,
        label_types=parsed_label_types,
    )

    for shard_spec in manifest["shards"]:
        shard_path = _resolve_shard(root, shard_spec["file"])
        with np.load(shard_path, allow_pickle=False) as shard:
            required_keys = {
                "input_token",
                "output_token",
                "batch_row",
                "sequence_id",
                "position",
                *(manifest["activation_keys"][hook] for hook in hooks),
            }
            missing_keys = required_keys - set(shard.files)
            if missing_keys:
                raise ValueError(
                    f"Shard {shard_path} is missing keys {sorted(missing_keys)}."
                )

            arrays = {
                "input_token": shard["input_token"],
                "output_token": shard["output_token"],
                "batch_row": shard["batch_row"],
                "sequence_id": shard["sequence_id"],
                "position": shard["position"],
                **{
                    f"activation__{hook}": shard[manifest["activation_keys"][hook]]
                    for hook in hooks
                },
            }
            if accumulator.update(arrays) == 0:
                raise ValueError(
                    f"Shard {shard_path} has no positions with the requested n-gram context."
                )
    accumulated_results = accumulator.results()
    return {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": manifest["checkpoint"],
        "eval_data_path": manifest["eval_data_path"],
        "eval_data_type": manifest.get("eval_data_type"),
        "tokenizer_path": manifest["tokenizer_path"],
        "sequence_length": manifest["sequence_length"],
        "n_bins": n_bins,
        "seed": seed,
        "label_types": list(parsed_label_types),
        **accumulated_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "activation_dir", help="Directory containing manifest.json and NPZ shards."
    )
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label-types",
        default="unigram",
        help="Comma-separated n-gram backoff orders (default: unigram).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; prints to stdout otherwise.",
    )
    args = parser.parse_args()

    results = analyze(
        args.activation_dir,
        n_bins=args.n_bins,
        seed=args.seed,
        label_types=args.label_types,
    )
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
