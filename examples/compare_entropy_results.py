"""Compare paired Tax and Hugging Face soft-entropy results."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

_ALLOWED_GCS_PREFIX = "gs://cohere-dev/michael-rizvi/"
_HOOK_PATTERN = re.compile(r"block_([0-9]+)_block_output")
_METRICS = (
    "H(Z)",
    "I(X;Z)/input_unigram",
    "I(X;Z)/output_unigram",
    "regularity/input_unigram",
    "regularity/output_unigram",
    "optimality/unigram",
)
_GATED_METRICS = _METRICS[:-1]
# A cross-backend agreement gate is only interpretable when both sides ran with
# exact-math attention and a bf16 residual stream. Field names differ per backend,
# so each side carries its own allowlist rather than a normalized abstraction.
_EXPECTED_FAX_NUMERICS = {
    "attention_impl": "jax_native",
    "quantize_params": True,
    "quantize_activations": True,
    "quantize_residuals": True,
    "use_fp8_gemm": False,
}
_EXPECTED_HF_NUMERICS = {
    "attention_implementation": "eager",
    "dtype": "torch.bfloat16",
}


def _validate_gcs_path(name: str, path: str) -> None:
    if not path.startswith(_ALLOWED_GCS_PREFIX) or any(
        part == ".." for part in path.split("/")
    ):
        raise ValueError(f"{name} must stay under {_ALLOWED_GCS_PREFIX!r}.")


def _load_json(path: str) -> dict[str, Any]:
    import fsspec

    with fsspec.open(path, "rt").open() as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path!r}.")
    return payload


def _ordered_hooks(results: dict[str, Any]) -> list[str]:
    hooks = results.get("hooks")
    if not isinstance(hooks, dict):
        raise TypeError("Entropy results are missing a hooks dictionary.")
    indexed_hooks = []
    for hook in hooks:
        match = _HOOK_PATTERN.fullmatch(hook)
        if match is None:
            raise ValueError(f"Unexpected entropy hook name {hook!r}.")
        indexed_hooks.append((int(match.group(1)), hook))
    indexed_hooks.sort()
    expected_indices = list(range(len(indexed_hooks)))
    observed_indices = [index for index, _ in indexed_hooks]
    if observed_indices != expected_indices:
        raise ValueError(
            f"Entropy hooks must be contiguous from block 0, got {observed_indices}."
        )
    return [hook for _, hook in indexed_hooks]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.array_equal(left, right):
        return 1.0
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def compare_payloads(
    fax: dict[str, Any],
    hf: dict[str, Any],
    *,
    mean_abs_tolerance: float = 0.002,
    max_abs_tolerance: float = 0.01,
    correlation_tolerance: float = 0.999,
) -> dict[str, Any]:
    """Validate experiment identity and summarize per-layer metric agreement."""
    if min(mean_abs_tolerance, max_abs_tolerance) < 0:
        raise ValueError("Absolute-error tolerances must be non-negative.")
    if not 0 <= correlation_tolerance <= 1:
        raise ValueError("correlation_tolerance must lie in [0, 1].")

    expected_identity = {
        "n_bins": 100,
        "seed": 0,
        "label_types": ["unigram"],
    }
    for name, expected in expected_identity.items():
        if fax.get(name) != expected or hf.get(name) != expected:
            raise ValueError(
                f"Paired result {name} must both equal {expected!r}; "
                f"got Fax={fax.get(name)!r}, HF={hf.get(name)!r}."
            )
    for side, payload, expected_numerics in (
        ("Fax", fax, _EXPECTED_FAX_NUMERICS),
        ("Hugging Face", hf, _EXPECTED_HF_NUMERICS),
    ):
        observed = {name: payload.get(name) for name in expected_numerics}
        if observed != expected_numerics:
            raise ValueError(
                f"{side} result was not produced with comparable numerics: "
                f"got {observed}, expected {expected_numerics}."
            )
    tokenizer_path = fax.get("tokenizer_path")
    if not isinstance(tokenizer_path, str) or hf.get("tokenizer_path") != tokenizer_path:
        raise ValueError(
            "Paired results must report the same tokenizer; got "
            f"Fax={tokenizer_path!r}, HF={hf.get('tokenizer_path')!r}."
        )
    package_digest = fax.get("soft_h_package_sha256")
    if (
        not isinstance(package_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", package_digest) is None
        or hf.get("soft_h_package_sha256") != package_digest
    ):
        raise ValueError("Paired results must use the same valid soft_h package hash.")
    data_prefix_digest = fax.get("data_prefix_sha256")
    if (
        not isinstance(data_prefix_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", data_prefix_digest) is None
        or hf.get("data_prefix_sha256") != data_prefix_digest
    ):
        raise ValueError("Paired results must use the same Fax data-prefix hash.")

    fax_hooks = _ordered_hooks(fax)
    hf_hooks = _ordered_hooks(hf)
    if fax_hooks != hf_hooks or len(fax_hooks) != 32:
        raise ValueError(
            f"Expected the same 32 block hooks, got Fax={fax_hooks}, HF={hf_hooks}."
        )
    if fax.get("total_selected_samples") != hf.get("total_selected_samples"):
        raise ValueError(
            "Fax and Hugging Face selected-position counts differ: "
            f"{fax.get('total_selected_samples')} versus "
            f"{hf.get('total_selected_samples')}."
        )

    fax_hook_results = fax["hooks"]
    hf_hook_results = hf["hooks"]
    for hook in fax_hooks:
        if fax_hook_results[hook].get("n_samples") != hf_hook_results[hook].get(
            "n_samples"
        ):
            raise ValueError(f"Selected-position counts differ for {hook}.")

    metric_summaries: dict[str, dict[str, Any]] = {}
    per_layer: dict[str, dict[str, dict[str, float]]] = {}
    passed = True
    for metric in _METRICS:
        fax_values = np.asarray(
            [fax_hook_results[hook][metric] for hook in fax_hooks],
            dtype=np.float64,
        )
        hf_values = np.asarray(
            [hf_hook_results[hook][metric] for hook in hf_hooks],
            dtype=np.float64,
        )
        if not np.isfinite(fax_values).all() or not np.isfinite(hf_values).all():
            raise ValueError(f"Metric {metric!r} contains non-finite values.")
        absolute_error = np.abs(fax_values - hf_values)
        denominator = np.maximum(
            np.maximum(np.abs(fax_values), np.abs(hf_values)),
            np.finfo(np.float64).eps,
        )
        relative_error = absolute_error / denominator
        correlation = _correlation(fax_values, hf_values)
        mean_abs_error = float(absolute_error.mean())
        max_abs_error = float(absolute_error.max())
        metric_passed = (
            mean_abs_error <= mean_abs_tolerance
            and max_abs_error <= max_abs_tolerance
            and math.isfinite(correlation)
            and correlation >= correlation_tolerance
        )
        is_gated = metric in _GATED_METRICS
        if is_gated:
            passed = passed and metric_passed
        metric_summaries[metric] = {
            "mean_abs_error": mean_abs_error,
            "max_abs_error": max_abs_error,
            "mean_relative_error": float(relative_error.mean()),
            "max_relative_error": float(relative_error.max()),
            "layer_trajectory_correlation": correlation,
            "gated": is_gated,
            "passed": metric_passed,
        }
        per_layer[metric] = {
            hook: {
                "fax": float(fax_value),
                "huggingface": float(hf_value),
                "abs_error": float(error),
            }
            for hook, fax_value, hf_value, error in zip(
                fax_hooks,
                fax_values,
                hf_values,
                absolute_error,
            )
        }

    return {
        "schema_version": 1,
        "passed": passed,
        "n_layers": len(fax_hooks),
        "n_selected_positions": fax["total_selected_samples"],
        "tolerances": {
            "mean_abs_error": mean_abs_tolerance,
            "max_abs_error": max_abs_tolerance,
            "layer_trajectory_correlation": correlation_tolerance,
        },
        "metric_summaries": metric_summaries,
        "per_layer": per_layer,
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    import fsspec
    from fsspec.core import url_to_fs

    filesystem, filesystem_path = url_to_fs(path)
    if filesystem.exists(filesystem_path):
        raise FileExistsError(f"Refusing to overwrite existing artifact {path!r}.")
    with fsspec.open(path, "wt").open() as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def compare(
    fax_results_path: str,
    hf_results_path: str,
    output_path: str,
    mean_abs_tolerance: float = 0.002,
    max_abs_tolerance: float = 0.01,
    correlation_tolerance: float = 0.999,
) -> dict[str, Any]:
    """Load paired artifacts, compare them, and write a durable report."""
    for name, path in (
        ("fax_results_path", fax_results_path),
        ("hf_results_path", hf_results_path),
        ("output_path", output_path),
    ):
        _validate_gcs_path(name, path)
    comparison = compare_payloads(
        _load_json(fax_results_path),
        _load_json(hf_results_path),
        mean_abs_tolerance=mean_abs_tolerance,
        max_abs_tolerance=max_abs_tolerance,
        correlation_tolerance=correlation_tolerance,
    )
    _write_json(output_path, comparison)
    print(json.dumps(comparison["metric_summaries"], indent=2, sort_keys=True))
    print(f"Agreement gate passed: {comparison['passed']}.")
    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fax-results-path", required=True)
    parser.add_argument("--hf-results-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mean-abs-tolerance", type=float, default=0.002)
    parser.add_argument("--max-abs-tolerance", type=float, default=0.01)
    parser.add_argument("--correlation-tolerance", type=float, default=0.999)
    args = parser.parse_args()
    compare(
        fax_results_path=args.fax_results_path,
        hf_results_path=args.hf_results_path,
        output_path=args.output_path,
        mean_abs_tolerance=args.mean_abs_tolerance,
        max_abs_tolerance=args.max_abs_tolerance,
        correlation_tolerance=args.correlation_tolerance,
    )
