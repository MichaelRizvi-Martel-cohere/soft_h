"""Freeze a deterministic sample of English C4 for model comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET_ID = "allenai/c4"
DATASET_CONFIG = "en"
DATASET_SPLIT = "validation"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def reservoir_sample(
    examples: Iterable[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select non-empty documents uniformly without loading the corpus."""
    if sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    n_valid = 0
    for source_index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"Example {source_index} must be a mapping.")
        text = example.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        record = {
            "source_index": source_index,
            "text": text,
            "url": example.get("url"),
            "timestamp": example.get("timestamp"),
        }
        if len(sample) < sample_size:
            sample.append(record)
        else:
            replacement_index = rng.randrange(n_valid + 1)
            if replacement_index < sample_size:
                sample[replacement_index] = record
        n_valid += 1

    if n_valid < sample_size:
        raise ValueError(
            f"Requested {sample_size} documents, but found only {n_valid} valid documents."
        )
    return sorted(sample, key=lambda record: record["source_index"])


def write_sample(
    records: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Write JSONL and a reproducibility manifest without overwriting."""
    output_dir = output_dir.resolve()
    data_path = output_dir / "documents.jsonl"
    manifest_path = output_dir / "manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing C4 sample under {output_dir}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    with data_path.open("wb") as output:
        for record in records:
            line = (
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            output.write(line)
            digest.update(line)

    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "sampling": "reservoir",
        "seed": seed,
        "n_documents": len(records),
        "source_indices": [record["source_index"] for record in records],
        "data_file": data_path.name,
        "data_sha256": digest.hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_drydock_parquet(
    records: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Write the content-column Parquet format consumed by Fax Drydock."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = output_dir.resolve() / "documents.parquet"
    if parquet_path.exists():
        raise FileExistsError(f"Refusing to overwrite {parquet_path}.")
    table = pa.table(
        {
            "content": pa.array(
                [record["text"].encode("utf-8") for record in records],
                type=pa.binary(),
            )
        }
    )
    pq.write_table(table, parquet_path)

    digest = hashlib.sha256()
    with parquet_path.open("rb") as parquet_file:
        for chunk in iter(lambda: parquet_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "parquet_file": parquet_path.name,
        "parquet_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-documents", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        streaming=True,
    )
    records = reservoir_sample(dataset, args.n_documents, args.seed)
    manifest = write_sample(records, args.output_dir, args.seed)
    manifest.update(write_drydock_parquet(records, args.output_dir))
    (args.output_dir.resolve() / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
