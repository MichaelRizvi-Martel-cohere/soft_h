import json
import tempfile
import unittest
from pathlib import Path

from examples.package_soft_h_for_tax import package_digest, stage_package


class PackageSoftHForTaxTest(unittest.TestCase):
    def test_stage_package_is_deterministic_and_excludes_caches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_repo = root / "soft_h"
            package_dir = source_repo / "soft_entropy"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("")
            (package_dir / "accumulator.py").write_text("VALUE = 1\n")
            cache_dir = package_dir / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "accumulator.py").write_text("not source\n")

            expected_digest, expected_hashes = package_digest(package_dir)
            output_dir = root / "tax" / "scripts" / "_soft_h_runtime"
            actual_digest = stage_package(source_repo, output_dir)

            self.assertEqual(actual_digest, expected_digest)
            self.assertFalse((output_dir / "soft_entropy" / "__pycache__").exists())
            manifest = json.loads((output_dir / "PACKAGE_MANIFEST.json").read_text())
            self.assertEqual(manifest["sha256"], expected_digest)
            self.assertEqual(manifest["files"], expected_hashes)

    def test_stage_package_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            package_dir = root / "soft_h" / "soft_entropy"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("")
            output_dir = root / "existing"
            output_dir.mkdir()

            with self.assertRaises(FileExistsError):
                stage_package(root / "soft_h", output_dir)


if __name__ == "__main__":
    unittest.main()
