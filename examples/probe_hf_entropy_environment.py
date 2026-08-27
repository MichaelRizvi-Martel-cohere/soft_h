"""Validate every runtime dependency needed by the HF entropy job."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os

import numpy as np
from extract_hf_entropy import _load_token_rows

from soft_entropy.accumulator import SoftEntropyAccumulator

_RUNTIME_MODULES = {
    "accelerate": "accelerate",
    "aiohttp": "aiohttp",
    "fsspec": "fsspec",
    "gcsfs": "gcsfs",
    "google.auth": "google-auth",
    "google.cloud.storage": "google-cloud-storage",
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "requests": "requests",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "torch": "torch",
    "transformers": "transformers",
}
_TOKEN_ARTIFACT = (
    "gs://cohere-dev/michael-rizvi/soft_h/entropy_comparison/"
    "command_r7b_n100_seq512_seed0/tokens/tokens.npz"
)
_TOKEN_SHA256 = "552c0eb49fcbeb38ee893df8837a926f753ac0d38fd97d4459008ffe2ff2bfdd"
_MODEL_ID = "CohereLabs/c4ai-command-r7b-12-2024"
_REVISION = "4f3d0aa6856e322f2f4480fe65420d5d53d297b8"


def main() -> None:
    """Exercise imports, GCS I/O, entropy math, and gated HF config access."""
    versions = {}
    for module, package in _RUNTIME_MODULES.items():
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)

    token_rows, token_digest = _load_token_rows(
        _TOKEN_ARTIFACT,
        n_samples=100,
        max_sequence_length=512,
        expected_token_sha256=_TOKEN_SHA256,
    )
    if len(token_rows) != 100 or token_digest != _TOKEN_SHA256:
        raise ValueError("Shared token artifact validation failed.")

    accumulator = SoftEntropyAccumulator(d=4, n_bins=8, seed=0, backend="numpy")
    activations = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    labels = {
        "input_unigram": np.arange(4, dtype=np.int64)[:, None],
        "output_unigram": np.arange(1, 5, dtype=np.int64)[:, None],
    }
    accumulator.update(activations, labels=labels)
    results = accumulator.results()
    if not all(np.isfinite(value) for value in results.values()):
        raise ValueError("Entropy accumulator produced non-finite results.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated config probe.")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        token=token,
    )
    observed_config = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
    }
    expected_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 256000,
    }
    if observed_config != expected_config:
        raise ValueError(
            f"Unexpected model config {observed_config}, expected {expected_config}."
        )

    print(
        json.dumps(
            {
                "status": "HF_ENTROPY_ENVIRONMENT_OK",
                "packages": versions,
                "token_ids_sha256": token_digest,
                "model_config": observed_config,
                "entropy_keys": sorted(results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
