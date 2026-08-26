import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.analyze_tax_activations import analyze


class AnalyzeTaxActivationsTest(unittest.TestCase):
    def _write_dataset(self, root: Path) -> None:
        hook = "block_0_block_output"
        activation_key = f"activation__{hook}"
        activation = np.arange(1, 49, dtype=np.float32).reshape(12, 4)
        np.savez(
            root / "batch_00000.npz",
            input_token=np.arange(1, 13, dtype=np.int64),
            output_token=np.arange(2, 14, dtype=np.int64),
            batch_row=np.zeros(12, dtype=np.int64),
            sequence_id=np.full(12, 7, dtype=np.int64),
            position=np.arange(12, dtype=np.int64),
            **{activation_key: activation},
        )
        manifest = {
            "schema_version": 2,
            "checkpoint": "test-checkpoint",
            "hooks": [hook],
            "activation_keys": {hook: activation_key},
            "activation_dtype": "float32",
            "label_keys": [
                "input_token",
                "output_token",
                "batch_row",
                "sequence_id",
                "position",
            ],
            "input_source": "fixed_drydock_eval",
            "eval_data_path": "c4.parquet",
            "sequence_length": 512,
            "tokenizer_path": "tokenizer.json",
            "shards": [
                {
                    "file": "batch_00000.npz",
                    "n_samples": 12,
                    "hook_dimensions": {hook: 4},
                }
            ],
            "total_samples": 12,
            "total_sequences": 1,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_analyze_is_deterministic_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_dataset(root)

            first = analyze(str(root), n_bins=8, seed=7)
            second = analyze(str(root), n_bins=8, seed=7)
            all_orders = analyze(
                str(root),
                n_bins=8,
                seed=7,
                label_types="unigram,bigram,trigram,quadgram",
            )

        self.assertEqual(first, second)
        hook_result = first["hooks"]["block_0_block_output"]
        self.assertEqual(hook_result["dimension"], 4)
        self.assertEqual(hook_result["n_samples"], 12)
        self.assertEqual(first["label_types"], ["unigram"])
        self.assertTrue(0 <= hook_result["H(Z)"] <= 1)
        self.assertTrue(np.isfinite(hook_result["I(X;Z)/input_unigram"]))
        self.assertNotIn("I(X;Z)/input_bigram", hook_result)
        self.assertEqual(
            all_orders["hooks"]["block_0_block_output"]["n_samples"],
            6,
        )
        self.assertTrue(
            np.isfinite(
                all_orders["hooks"]["block_0_block_output"]["I(X;Z)/input_quadgram"]
            )
        )

    def test_rejects_shard_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["shards"][0]["file"] = "../batch_00000.npz"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "escapes activation directory"):
                analyze(str(root), n_bins=8)


if __name__ == "__main__":
    unittest.main()
