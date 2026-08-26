"""Compare selected Fax and Hugging Face pre-softmax logits."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np

_ALLOWED_GCS_PREFIX = "gs://cohere-dev/michael-rizvi/"


def _load_npz(path: str) -> dict[str, np.ndarray]:
    from co import fs

    with fs.open(path, "rb") as source:
        payload = source.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / denominator


def compute_metrics(
    fax_logits: np.ndarray, hf_logits: np.ndarray, top_k: int = 10
) -> dict[str, Any]:
    """Compute deterministic distribution-level agreement metrics."""
    fax_logits = np.asarray(fax_logits, dtype=np.float64)
    hf_logits = np.asarray(hf_logits, dtype=np.float64)
    if fax_logits.shape != hf_logits.shape or fax_logits.ndim != 2:
        raise ValueError(
            f"Logits must have the same [N, V] shape, got {fax_logits.shape} and {hf_logits.shape}."
        )
    if top_k < 1 or top_k > fax_logits.shape[-1]:
        raise ValueError(f"top_k must be in [1, {fax_logits.shape[-1]}], got {top_k}.")
    if not np.isfinite(fax_logits).all() or not np.isfinite(hf_logits).all():
        raise ValueError("Logits contain non-finite values.")

    difference = fax_logits - hf_logits
    fax_centered = fax_logits - fax_logits.mean(axis=-1, keepdims=True)
    hf_centered = hf_logits - hf_logits.mean(axis=-1, keepdims=True)
    centered_difference = fax_centered - hf_centered

    fax_log_probs = _log_softmax(fax_logits)
    hf_log_probs = _log_softmax(hf_logits)
    fax_probs = np.exp(fax_log_probs)
    hf_probs = np.exp(hf_log_probs)
    kl_fax_hf = np.sum(fax_probs * (fax_log_probs - hf_log_probs), axis=-1)
    kl_hf_fax = np.sum(hf_probs * (hf_log_probs - fax_log_probs), axis=-1)

    fax_argmax = np.argmax(fax_logits, axis=-1)
    hf_argmax = np.argmax(hf_logits, axis=-1)
    fax_top = np.argpartition(fax_logits, -top_k, axis=-1)[:, -top_k:]
    hf_top = np.argpartition(hf_logits, -top_k, axis=-1)[:, -top_k:]
    top_overlap = np.asarray(
        [
            len(set(left.tolist()) & set(right.tolist())) / top_k
            for left, right in zip(fax_top, hf_top)
        ]
    )

    return {
        "n_distributions": int(fax_logits.shape[0]),
        "vocab_size": int(fax_logits.shape[1]),
        "argmax_agreement": float(np.mean(fax_argmax == hf_argmax)),
        "top_k": top_k,
        "top_k_overlap_mean": float(np.mean(top_overlap)),
        "top_k_overlap_min": float(np.min(top_overlap)),
        "raw_max_abs_error": float(np.max(np.abs(difference))),
        "raw_mean_abs_error": float(np.mean(np.abs(difference))),
        "raw_rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "centered_max_abs_error": float(np.max(np.abs(centered_difference))),
        "centered_mean_abs_error": float(np.mean(np.abs(centered_difference))),
        "centered_rms_error": float(np.sqrt(np.mean(np.square(centered_difference)))),
        "cosine_similarity_mean": float(np.mean(_cosine_rows(fax_logits, hf_logits))),
        "cosine_similarity_min": float(np.min(_cosine_rows(fax_logits, hf_logits))),
        "centered_cosine_similarity_mean": float(
            np.mean(_cosine_rows(fax_centered, hf_centered))
        ),
        "centered_cosine_similarity_min": float(
            np.min(_cosine_rows(fax_centered, hf_centered))
        ),
        "kl_fax_to_hf_mean": float(np.mean(kl_fax_hf)),
        "kl_fax_to_hf_max": float(np.max(kl_fax_hf)),
        "kl_hf_to_fax_mean": float(np.mean(kl_hf_fax)),
        "kl_hf_to_fax_max": float(np.max(kl_hf_fax)),
    }


def compare(
    fax_logits_path: str,
    hf_logits_path: str,
    output_path: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Load paired artifacts, verify alignment, and report agreement."""
    for name, path in (
        ("fax_logits_path", fax_logits_path),
        ("hf_logits_path", hf_logits_path),
    ):
        if not path.startswith(_ALLOWED_GCS_PREFIX):
            raise ValueError(f"{name} must stay under {_ALLOWED_GCS_PREFIX!r}.")
    if output_path is not None and not output_path.startswith(_ALLOWED_GCS_PREFIX):
        raise ValueError(f"output_path must stay under {_ALLOWED_GCS_PREFIX!r}.")

    fax = _load_npz(fax_logits_path)
    hf = _load_npz(hf_logits_path)
    for key in ("selected_rows", "selected_positions"):
        if key not in fax or key not in hf or not np.array_equal(fax[key], hf[key]):
            raise ValueError(f"Fax and Hugging Face artifacts disagree on {key}.")
    metrics = compute_metrics(fax["logits"], hf["logits"], top_k=top_k)

    if output_path is not None:
        from co import fs

        if fs.exists(output_path):
            raise FileExistsError(
                f"Refusing to overwrite existing result {output_path!r}."
            )
        with fs.open(output_path, "w") as output:
            json.dump(metrics, output, indent=2, sort_keys=True)
            output.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fax-logits-path", required=True)
    parser.add_argument("--hf-logits-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    compare(
        fax_logits_path=args.fax_logits_path,
        hf_logits_path=args.hf_logits_path,
        output_path=args.output_path,
        top_k=args.top_k,
    )
