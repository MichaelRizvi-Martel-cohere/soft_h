import unittest

import numpy as np
import torch

from soft_entropy.accumulator import SoftEntropyAccumulator
from soft_entropy.llm import _encode_ngram


class NgramLabelsTest(unittest.TestCase):
    def test_quadgrams_preserve_token_tuples(self):
        token_ids = torch.arange(10)

        input_labels = _encode_ngram(token_ids, start=3, end=6, order=4, forward=False)
        output_labels = _encode_ngram(token_ids, start=3, end=6, order=4, forward=True)

        torch.testing.assert_close(
            input_labels,
            torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]),
        )
        torch.testing.assert_close(
            output_labels,
            torch.tensor([[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]),
        )

    def test_large_vocabulary_quadgrams_do_not_overflow(self):
        token_ids = torch.tensor([255_000, 254_999, 254_998, 254_997, 254_996])

        labels = _encode_ngram(token_ids, start=3, end=4, order=4, forward=False)

        torch.testing.assert_close(labels, token_ids[:4].reshape(1, 4))

    def test_accumulator_groups_row_labels(self):
        activations = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        row_labels = np.array([[1, 2], [1, 2], [2, 3], [2, 3]])
        scalar_labels = np.array([0, 0, 1, 1])
        row_accumulator = SoftEntropyAccumulator(d=4, n_bins=8, seed=3)
        scalar_accumulator = SoftEntropyAccumulator(d=4, n_bins=8, seed=3)

        row_accumulator.update(activations, row_labels)
        scalar_accumulator.update(activations, scalar_labels)

        self.assertAlmostEqual(
            row_accumulator.results()["I(X;Z)/labels"],
            scalar_accumulator.results()["I(X;Z)/labels"],
        )


if __name__ == "__main__":
    unittest.main()
