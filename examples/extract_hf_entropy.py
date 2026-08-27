"""Estimate Command R7B entropy on the frozen C4 comparison sample."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

from soft_entropy.accumulator import SoftEntropyAccumulator

_MODEL_ID = "CohereLabs/c4ai-command-r7b-12-2024"
_MODEL_REVISION = "4f3d0aa6856e322f2f4480fe65420d5d53d297b8"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_ALLOWED_GCS_PREFIX = "gs://cohere-dev/michael-rizvi/"
_MANIFEST_FIELDS = (
    "token_ids_sha256",
    "data_prefix_sha256",
    "tokenizer_path",
    "n_samples",
    "max_sequence_length",
)


def _validate_gcs_path(name: str, path: str) -> None:
    if not path.startswith(_ALLOWED_GCS_PREFIX) or any(
        part == ".." for part in path.split("/")
    ):
        raise ValueError(f"{name} must stay under {_ALLOWED_GCS_PREFIX!r}.")


def load_artifact_manifest(
    token_artifact_dir: str,
    n_samples: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Read the digests and tokenizer recorded by the Tax token exporter.

    Sourcing these from the artifact rather than the command line is what binds
    this run to the exact evaluation batches Fax consumed.
    """
    import fsspec

    manifest_path = f"{token_artifact_dir.rstrip('/')}/manifest.json"
    with fsspec.open(manifest_path, "rt").open() as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected a JSON object at {manifest_path!r}.")

    missing = [name for name in _MANIFEST_FIELDS if name not in manifest]
    if missing:
        raise ValueError(f"Token manifest {manifest_path!r} is missing {missing}.")
    for name in ("token_ids_sha256", "data_prefix_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(manifest[name])) is None:
            raise ValueError(f"Token manifest {name} is not a lowercase SHA-256 digest.")
    if not isinstance(manifest["tokenizer_path"], str) or not manifest["tokenizer_path"]:
        raise ValueError("Token manifest tokenizer_path must be a non-empty string.")
    if manifest["n_samples"] != n_samples:
        raise ValueError(
            f"Token manifest holds {manifest['n_samples']} samples, requested {n_samples}."
        )
    if manifest["max_sequence_length"] != max_sequence_length:
        raise ValueError(
            f"Token manifest was written at sequence length "
            f"{manifest['max_sequence_length']}, requested {max_sequence_length}."
        )
    return manifest


def _load_token_rows(
    path: str,
    n_samples: int,
    max_sequence_length: int,
    expected_token_sha256: str,
) -> tuple[list[np.ndarray], str]:
    import fsspec

    with fsspec.open(path, "rb").open() as source:
        payload = source.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as artifact:
        if set(artifact.files) != {"input_ids", "lengths", "source_indices"}:
            raise ValueError(
                f"Token artifact contains unexpected arrays {sorted(artifact.files)}."
            )
        input_ids = artifact["input_ids"]
        lengths = artifact["lengths"]
        source_indices = artifact["source_indices"]
    if input_ids.shape != (n_samples, max_sequence_length):
        raise ValueError(
            f"input_ids must have shape {(n_samples, max_sequence_length)}, "
            f"got {input_ids.shape}."
        )
    if (
        lengths.shape != (n_samples,)
        or not np.issubdtype(lengths.dtype, np.integer)
        or np.any(lengths < 2)
        or np.any(lengths > max_sequence_length)
    ):
        raise ValueError(f"Invalid token lengths with shape {lengths.shape}.")
    if (
        source_indices.shape != (n_samples,)
        or not np.issubdtype(source_indices.dtype, np.integer)
        or np.any(source_indices < 0)
        or np.any(np.diff(source_indices) <= 0)
    ):
        raise ValueError("source_indices must be non-negative and strictly increasing.")

    token_rows = []
    for sample_index, length in enumerate(lengths):
        length = int(length)
        row = np.asarray(input_ids[sample_index, :length], dtype=np.int64)
        padding = input_ids[sample_index, length:]
        if np.any(row <= 0) or np.any(padding != 0):
            raise ValueError(
                f"Sample {sample_index} has invalid token IDs or right padding."
            )
        token_rows.append(row)
    observed_digest = token_rows_sha256(token_rows)
    if observed_digest != expected_token_sha256:
        raise ValueError(
            f"Shared token artifact hashes to {observed_digest}, "
            f"expected {expected_token_sha256}."
        )
    return token_rows, observed_digest


def token_rows_sha256(token_rows: Iterable[np.ndarray]) -> str:
    """Hash ordered, variable-length token rows in a canonical representation."""
    digest = hashlib.sha256()
    for row in token_rows:
        canonical = np.ascontiguousarray(row, dtype="<i8")
        digest.update(np.asarray([canonical.size], dtype="<u8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def build_unigram_labels(
    token_rows: Iterable[np.ndarray],
) -> tuple[dict[str, np.ndarray], list[slice]]:
    """Build current/next-token labels and aligned per-sample activation slices."""
    input_labels = []
    output_labels = []
    slices = []
    offset = 0
    for row in token_rows:
        row = np.asarray(row, dtype=np.int64)
        n_positions = row.size - 1
        input_labels.append(row[:-1, None])
        output_labels.append(row[1:, None])
        slices.append(slice(offset, offset + n_positions))
        offset += n_positions
    return {
        "input_unigram": np.concatenate(input_labels),
        "output_unigram": np.concatenate(output_labels),
    }, slices


def _decoder_layers(model: Any, expected_layers: int) -> list[Any]:
    candidates = (
        getattr(model, "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    )
    for layers in candidates:
        if layers is not None and len(layers) == expected_layers:
            return list(layers)
    observed = [len(layers) for layers in candidates if layers is not None]
    raise ValueError(
        f"Could not locate {expected_layers} decoder layers; observed lengths {observed}."
    )


def _layer_output(output: Any) -> Any:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if getattr(hidden, "ndim", None) != 3:
        raise ValueError(
            f"Decoder layer output must have shape [B, S, D], got "
            f"{getattr(hidden, 'shape', None)}."
        )
    return hidden


def collect_results(
    accumulators: list[SoftEntropyAccumulator],
) -> dict[str, dict[str, float | int]]:
    """Return Tax-compatible metric dictionaries for raw block outputs."""
    hook_results: dict[str, dict[str, float | int]] = {}
    for layer_index, accumulator in enumerate(accumulators):
        metrics = accumulator.results()
        input_mi = metrics["I(X;Z)/input_unigram"]
        output_mi = metrics["I(X;Z)/output_unigram"]
        metrics["optimality/unigram"] = (
            output_mi / input_mi if input_mi > 0 else float("nan")
        )
        hook_results[f"block_{layer_index}_block_output"] = {
            **metrics,
            "dimension": accumulator.d,
            "n_samples": accumulator.n_samples,
        }
    return hook_results


def _write_json(path: str, payload: dict[str, Any]) -> None:
    import fsspec
    from fsspec.core import url_to_fs

    filesystem, filesystem_path = url_to_fs(path)
    if filesystem.exists(filesystem_path):
        raise FileExistsError(f"Refusing to overwrite existing artifact {path!r}.")
    with fsspec.open(path, "wt").open() as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def main(
    token_artifact_dir: str,
    output_path: str,
    soft_h_package_sha256: str,
    n_samples: int = 100,
    batch_size: int = 2,
    max_sequence_length: int = 512,
    n_bins: int = 100,
    seed: int = 0,
    model_id: str = _MODEL_ID,
    revision: str = _MODEL_REVISION,
    attention_implementation: str = "eager",
) -> None:
    """Run public Command R7B and accumulate entropy for every raw block output."""
    _validate_gcs_path("token_artifact_dir", token_artifact_dir)
    _validate_gcs_path("output_path", output_path)
    if model_id != _MODEL_ID:
        raise ValueError(f"model_id must be pinned to {_MODEL_ID!r}.")
    if _REVISION_PATTERN.fullmatch(revision) is None or revision != _MODEL_REVISION:
        raise ValueError(f"revision must be pinned to {_MODEL_REVISION!r}.")
    if attention_implementation != "eager":
        raise ValueError("The agreement gate requires eager attention.")
    if n_samples < 1 or batch_size < 1 or n_samples % batch_size:
        raise ValueError("n_samples must be positive and divisible by batch_size.")
    if max_sequence_length < 2 or n_bins < 2 or seed < 0:
        raise ValueError("Invalid sequence length, bin count, or seed.")
    if re.fullmatch(r"[0-9a-f]{64}", soft_h_package_sha256) is None:
        raise ValueError("soft_h_package_sha256 must be a lowercase SHA-256 digest.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be injected at runtime for the gated model.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = load_artifact_manifest(
        token_artifact_dir,
        n_samples,
        max_sequence_length,
    )
    token_rows, observed_token_sha256 = _load_token_rows(
        f"{token_artifact_dir.rstrip('/')}/tokens.npz",
        n_samples,
        max_sequence_length,
        manifest["token_ids_sha256"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    expected_special_ids = {"pad_token_id": 0, "bos_token_id": 5, "eos_token_id": 255001}
    observed_special_ids = {
        name: getattr(tokenizer, name) for name in expected_special_ids
    }
    if observed_special_ids != expected_special_ids:
        raise ValueError(
            f"Unexpected public tokenizer special IDs: {observed_special_ids}, "
            f"expected {expected_special_ids}."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attention_implementation,
    )
    expected_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 256000,
    }
    observed_config = {name: getattr(model.config, name) for name in expected_config}
    if observed_config != expected_config:
        raise ValueError(
            f"Unexpected public model config: {observed_config}, "
            f"expected {expected_config}."
        )
    # A token ID beyond the public vocabulary means the Fax checkpoint that wrote
    # the artifact does not share this model's tokenizer.
    highest_token_id = max(int(row.max()) for row in token_rows)
    if highest_token_id >= expected_config["vocab_size"]:
        raise ValueError(
            f"Token artifact contains ID {highest_token_id}, outside the "
            f"{expected_config['vocab_size']}-token public vocabulary; "
            f"it was written with tokenizer {manifest['tokenizer_path']!r}."
        )
    # `dtype` and `attn_implementation` are silently ignored by older
    # transformers releases, which would void the comparison.
    observed_dtype = next(model.parameters()).dtype
    if observed_dtype is not torch.bfloat16:
        raise ValueError(f"Model loaded in {observed_dtype}, expected torch.bfloat16.")
    observed_attention = str(getattr(model.config, "_attn_implementation", ""))
    if observed_attention != attention_implementation:
        raise ValueError(
            f"Model is using {observed_attention!r} attention, "
            f"requested {attention_implementation!r}."
        )
    model.eval()

    layers = _decoder_layers(model, expected_config["num_hidden_layers"])
    accumulators = [
        SoftEntropyAccumulator(
            d=expected_config["hidden_size"],
            n_bins=n_bins,
            seed=seed,
            backend="numpy",
        )
        for _ in layers
    ]
    captured: list[np.ndarray | None] = [None] * len(layers)
    handles = []
    for layer_index, layer in enumerate(layers):

        def capture(_module: Any, _inputs: Any, output: Any, index: int = layer_index):
            captured[index] = _layer_output(output).detach().float().cpu().numpy()

        handles.append(layer.register_forward_hook(capture))

    try:
        device = next(model.parameters()).device
        total_positions = 0
        for start in range(0, n_samples, batch_size):
            rows = token_rows[start : start + batch_size]
            padded_length = max(row.size for row in rows)
            input_ids = torch.zeros(
                (batch_size, padded_length),
                device=device,
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(input_ids)
            for batch_row, row in enumerate(rows):
                row_tensor = torch.as_tensor(row, device=device, dtype=torch.long)
                input_ids[batch_row, : row.size] = row_tensor
                attention_mask[batch_row, : row.size] = 1

            captured[:] = [None] * len(layers)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            if any(hidden is None for hidden in captured):
                missing = [
                    index for index, hidden in enumerate(captured) if hidden is None
                ]
                raise RuntimeError(f"Forward hooks did not capture layers {missing}.")

            labels, _ = build_unigram_labels(rows)
            total_positions += labels["input_unigram"].shape[0]
            for accumulator, hidden in zip(accumulators, captured):
                assert hidden is not None
                activations = np.concatenate(
                    [
                        hidden[row_index, : row.size - 1]
                        for row_index, row in enumerate(rows)
                    ]
                )
                if activations.shape[0] != labels["input_unigram"].shape[0]:
                    raise ValueError("Captured activations do not align with labels.")
                accumulator.update(activations, labels=labels)
    finally:
        for handle in handles:
            handle.remove()

    hook_results = collect_results(accumulators)
    if any(result["n_samples"] != total_positions for result in hook_results.values()):
        raise ValueError("Layer accumulators disagree on the selected sample count.")

    _write_json(
        output_path,
        {
            "schema_version": 1,
            "backend": "huggingface",
            "model_id": model_id,
            "revision": revision,
            "dtype": str(observed_dtype),
            "attention_implementation": observed_attention,
            "token_artifact_dir": token_artifact_dir,
            "tokenizer_path": manifest["tokenizer_path"],
            "n_samples": n_samples,
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "n_bins": n_bins,
            "seed": seed,
            "label_types": ["unigram"],
            "soft_h_package_sha256": soft_h_package_sha256,
            "token_ids_sha256": observed_token_sha256,
            "data_prefix_sha256": manifest["data_prefix_sha256"],
            "total_selected_samples": total_positions,
            "model_config": observed_config,
            "hooks": hook_results,
        },
    )
    print(f"Wrote Hugging Face entropy results to {output_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--token-artifact-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--soft-h-package-sha256", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default=_MODEL_ID)
    parser.add_argument("--revision", default=_MODEL_REVISION)
    parser.add_argument("--attention-implementation", default="eager")
    args = parser.parse_args()
    main(
        token_artifact_dir=args.token_artifact_dir,
        output_path=args.output_path,
        soft_h_package_sha256=args.soft_h_package_sha256,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        n_bins=args.n_bins,
        seed=args.seed,
        model_id=args.model_id,
        revision=args.revision,
        attention_implementation=args.attention_implementation,
    )
