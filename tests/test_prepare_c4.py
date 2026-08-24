import json
import tempfile
import unittest
from pathlib import Path

from examples.prepare_c4 import reservoir_sample, write_sample


class PrepareC4Test(unittest.TestCase):
    def test_reservoir_sample_is_deterministic_and_skips_empty_text(self):
        examples = [
            {"text": f"document {index}", "url": f"https://example.com/{index}"}
            for index in range(20)
        ]
        examples.insert(3, {"text": "  "})

        first = reservoir_sample(examples, sample_size=5, seed=7)
        second = reservoir_sample(examples, sample_size=5, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(
            [record["source_index"] for record in first],
            sorted(record["source_index"] for record in first),
        )

    def test_write_sample_is_byte_deterministic(self):
        records = reservoir_sample(
            [{"text": f"document {index}"} for index in range(10)],
            sample_size=4,
            seed=3,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_dir = Path(tmp_dir) / "first"
            second_dir = Path(tmp_dir) / "second"
            first_manifest = write_sample(records, first_dir, seed=3)
            second_manifest = write_sample(records, second_dir, seed=3)

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                (first_dir / "documents.jsonl").read_bytes(),
                (second_dir / "documents.jsonl").read_bytes(),
            )
            self.assertEqual(
                json.loads((first_dir / "manifest.json").read_text()),
                json.loads((second_dir / "manifest.json").read_text()),
            )

    def test_write_sample_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "sample"
            write_sample(
                [
                    {
                        "source_index": 0,
                        "text": "document",
                        "url": None,
                        "timestamp": None,
                    }
                ],
                output_dir,
                0,
            )
            with self.assertRaises(FileExistsError):
                write_sample([], output_dir, 0)


if __name__ == "__main__":
    unittest.main()
