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
        activation = np.arange(1, 25, dtype=np.float32).reshape(6, 4)
        np.savez(
            root / "batch_00000.npz",
            input_token=np.array([1, 1, 2, 2, 3, 3], dtype=np.int64),
            output_token=np.array([2, 2, 3, 3, 4, 4], dtype=np.int64),
            **{activation_key: activation},
        )
        manifest = {
            "schema_version": 1,
            "checkpoint": "test-checkpoint",
            "hooks": [hook],
            "activation_keys": {hook: activation_key},
            "activation_dtype": "float32",
            "label_keys": ["input_token", "output_token"],
            "shards": [
                {
                    "file": "batch_00000.npz",
                    "n_samples": 6,
                    "hook_dimensions": {hook: 4},
                }
            ],
            "total_samples": 6,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_analyze_is_deterministic_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_dataset(root)

            first = analyze(str(root), n_bins=8, seed=7)
            second = analyze(str(root), n_bins=8, seed=7)

        self.assertEqual(first, second)
        hook_result = first["hooks"]["block_0_block_output"]
        self.assertEqual(hook_result["dimension"], 4)
        self.assertEqual(hook_result["n_samples"], 6)
        self.assertTrue(0 <= hook_result["H(Z)"] <= 1)
        self.assertTrue(np.isfinite(hook_result["I(X;Z)/input_token"]))
        self.assertTrue(np.isfinite(hook_result["I(X;Z)/output_token"]))

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
