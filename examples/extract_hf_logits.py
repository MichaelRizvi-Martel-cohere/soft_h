"""Export Hugging Face logits for the exact token batch produced by Fax."""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any

import numpy as np

_MODEL_ID = "CohereLabs/c4ai-command-r7b-12-2024"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_ALLOWED_GCS_PREFIX = "gs://cohere-dev/michael-rizvi/"


def _validate_args(
    model_id: str, revision: str, fax_logits_path: str, output_dir: str
) -> None:
    if model_id != _MODEL_ID:
        raise ValueError(
            f"This comparison is pinned to {_MODEL_ID!r}, got {model_id!r}."
        )
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(
            "revision must be a full lowercase 40-character Git commit SHA."
        )
    for name, path in (
        ("fax_logits_path", fax_logits_path),
        ("output_dir", output_dir),
    ):
        if not path.startswith(_ALLOWED_GCS_PREFIX) or any(
            part == ".." for part in path.split("/")
        ):
            raise ValueError(f"{name} must stay under {_ALLOWED_GCS_PREFIX!r}.")


def _load_npz(path: str) -> dict[str, np.ndarray]:
    import fsspec

    with fsspec.open(path, "rb").open() as source:
        payload = source.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _write_npz(path: str, arrays: dict[str, np.ndarray]) -> None:
    import fsspec
    from fsspec.core import url_to_fs

    filesystem, filesystem_path = url_to_fs(path)
    if filesystem.exists(filesystem_path):
        raise FileExistsError(f"Refusing to overwrite existing artifact {path!r}.")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with fsspec.open(path, "wb").open() as output:
        output.write(buffer.getvalue())


def _write_json(path: str, payload: dict[str, Any]) -> None:
    import fsspec
    from fsspec.core import url_to_fs

    filesystem, filesystem_path = url_to_fs(path)
    if filesystem.exists(filesystem_path):
        raise FileExistsError(f"Refusing to overwrite existing artifact {path!r}.")
    with fsspec.open(path, "w").open() as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def select_hf_logits(
    logits: Any,
    selected_rows: np.ndarray,
    selected_positions: np.ndarray,
) -> np.ndarray:
    """Select the same rows and token positions exported by Fax."""
    import torch

    rows = torch.as_tensor(selected_rows, device=logits.device, dtype=torch.long)
    positions = torch.as_tensor(
        selected_positions, device=logits.device, dtype=torch.long
    )
    selected = logits[rows, positions].float().cpu().numpy()
    if not np.isfinite(selected).all():
        raise ValueError("Selected Hugging Face logits contain non-finite values.")
    return selected


def main(
    fax_logits_path: str,
    output_dir: str,
    model_id: str = _MODEL_ID,
    revision: str = "4f3d0aa6856e322f2f4480fe65420d5d53d297b8",
    attention_implementation: str = "eager",
) -> None:
    """Load pinned public Command R7B and evaluate the Fax-exported token batch."""
    _validate_args(model_id, revision, fax_logits_path, output_dir)
    if attention_implementation != "eager":
        raise ValueError("The identity gate requires attention_implementation='eager'.")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be injected at runtime for the gated model.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fax_arrays = _load_npz(fax_logits_path)
    required = {"input_ids", "attention_mask", "selected_rows", "selected_positions"}
    missing = required - fax_arrays.keys()
    if missing:
        raise KeyError(f"Fax artifact is missing arrays: {sorted(missing)}.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    expected_special_ids = {
        "pad_token_id": 0,
        "bos_token_id": 5,
        "eos_token_id": 255001,
    }
    observed_special_ids = {
        name: getattr(tokenizer, name) for name in expected_special_ids
    }
    if observed_special_ids != expected_special_ids:
        raise ValueError(
            f"Unexpected public tokenizer special IDs: {observed_special_ids}, expected {expected_special_ids}."
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
            f"Unexpected public model config: {observed_config}, expected {expected_config}."
        )
    model.eval()

    device = next(model.parameters()).device
    input_ids = torch.as_tensor(
        fax_arrays["input_ids"], device=device, dtype=torch.long
    )
    attention_mask = torch.as_tensor(
        fax_arrays["attention_mask"], device=device, dtype=torch.long
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    selected_logits = select_hf_logits(
        outputs.logits,
        fax_arrays["selected_rows"],
        fax_arrays["selected_positions"],
    )

    output_dir = output_dir.rstrip("/")
    logits_path = f"{output_dir}/hf_logits.npz"
    manifest_path = f"{output_dir}/hf_manifest.json"
    _write_npz(
        logits_path,
        {
            "selected_rows": fax_arrays["selected_rows"],
            "selected_positions": fax_arrays["selected_positions"],
            "logits": selected_logits,
        },
    )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "backend": "huggingface",
            "model_id": model_id,
            "revision": revision,
            "dtype": str(next(model.parameters()).dtype),
            "attention_implementation": attention_implementation,
            "special_token_ids": observed_special_ids,
            "model_config": observed_config,
            "n_selected_positions": int(selected_logits.shape[0]),
            "vocab_size": int(selected_logits.shape[-1]),
        },
    )
    print(f"Wrote Hugging Face logits to {logits_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fax-logits-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=_MODEL_ID)
    parser.add_argument(
        "--revision", default="4f3d0aa6856e322f2f4480fe65420d5d53d297b8"
    )
    parser.add_argument("--attention-implementation", default="eager")
    args = parser.parse_args()
    main(
        fax_logits_path=args.fax_logits_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        revision=args.revision,
        attention_implementation=args.attention_implementation,
    )
