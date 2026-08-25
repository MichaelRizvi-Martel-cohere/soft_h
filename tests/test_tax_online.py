import copy
import unittest

import numpy as np
from soft_entropy.tax_online import (
    NGRAM_ORDERS,
    TaxActivationAccumulator,
    build_ngram_labels,
)


def _make_batch(offset: int, scale: float = 1.0) -> dict[str, np.ndarray]:
    hook = "block_0_block_output"
    input_token = np.arange(offset, offset + 12, dtype=np.int64)
    activation = np.arange(1, 49, dtype=np.float32).reshape(12, 4) * scale + np.array(
        [0.0, 0.25, 0.5, 0.75], dtype=np.float32
    )
    return {
        "input_token": input_token,
        "output_token": input_token + 1,
        "batch_row": np.zeros(12, dtype=np.int64),
        "sequence_id": np.full(12, offset, dtype=np.int64),
        "position": np.arange(12, dtype=np.int64),
        f"activation__{hook}": activation,
    }


class TaxOnlineAccumulatorTest(unittest.TestCase):
    def test_build_ngram_labels_selects_paper_aligned_context(self):
        batch = _make_batch(10)
        selected, labels = build_ngram_labels(
            batch["input_token"],
            batch["output_token"],
            batch["batch_row"],
            batch["sequence_id"],
            batch["position"],
        )

        np.testing.assert_array_equal(selected, np.arange(3, 9))
        self.assertEqual(
            set(labels),
            {
                f"{direction}_{name}"
                for name in NGRAM_ORDERS
                for direction in ("input", "output")
            },
        )
        np.testing.assert_array_equal(labels["input_quadgram"][0], [10, 11, 12, 13])
        np.testing.assert_array_equal(labels["output_quadgram"][0], [14, 15, 16, 17])

    def test_batchwise_updates_equal_one_global_update(self):
        hook = "block_0_block_output"
        first = _make_batch(10, scale=1.0)
        second = _make_batch(100, scale=1.5)

        online = TaxActivationAccumulator((hook,), n_bins=8, seed=7)
        self.assertEqual(online.update(first), 6)
        self.assertEqual(online.update(second), 6)

        combined = {
            key: np.concatenate([first[key], second[key]], axis=0) for key in first
        }
        combined["batch_row"][12:] = 1
        reference = TaxActivationAccumulator((hook,), n_bins=8, seed=7)
        self.assertEqual(reference.update(combined), 12)

        online_results = online.results()
        reference_results = reference.results()
        self.assertEqual(
            online_results["hooks"][hook]["n_samples"],
            reference_results["hooks"][hook]["n_samples"],
        )
        for name, value in online_results["hooks"][hook].items():
            if isinstance(value, float):
                self.assertTrue(
                    np.isclose(
                        value,
                        reference_results["hooks"][hook][name],
                        rtol=1e-5,
                        atol=1e-7,
                    ),
                    msg=f"{name} differs between online and global accumulation.",
                )

    def test_multiple_hooks_are_accumulated_independently(self):
        first_hook = "block_0_block_output"
        second_hook = "block_1_block_output"
        batch = _make_batch(10)
        batch[f"activation__{second_hook}"] = (
            batch[f"activation__{first_hook}"][:, ::-1] + 3
        )
        accumulator = TaxActivationAccumulator(
            (first_hook, second_hook), n_bins=8, seed=3
        )

        accumulator.update(batch)
        results = accumulator.results()

        self.assertEqual(set(results["hooks"]), {first_hook, second_hook})
        self.assertEqual(results["hooks"][first_hook]["n_samples"], 6)
        self.assertEqual(results["hooks"][second_hook]["n_samples"], 6)
        self.assertTrue(np.isfinite(list(results["mean"].values())).all())
        for name in NGRAM_ORDERS:
            for direction in ("input", "output"):
                label_name = f"{direction}_{name}"
                mean_mi = np.mean(
                    [
                        results["hooks"][hook][f"I(X;Z)/{label_name}"]
                        for hook in (first_hook, second_hook)
                    ]
                )
                self.assertAlmostEqual(
                    results["mean"][f"I(X;Z)/{label_name}"], mean_mi
                )
                self.assertAlmostEqual(
                    results["mean"][f"regularity/{label_name}"],
                    mean_mi / results["mean"]["H(Z)"],
                )
            self.assertAlmostEqual(
                results["mean"][f"optimality/{name}"],
                results["mean"][f"I(X;Z)/output_{name}"]
                / results["mean"][f"I(X;Z)/input_{name}"],
            )

    def test_model_ratios_are_computed_after_averaging_layer_metrics(self):
        hooks = ("block_0_block_output", "block_1_block_output")
        accumulator = TaxActivationAccumulator(hooks, n_bins=8, seed=3)

        class FakeLayerAccumulator:
            def __init__(self, entropy, input_mi, output_mi):
                self.w = np.empty((8, 4))
                self._metrics = {"H(Z)": entropy}
                for name in NGRAM_ORDERS:
                    self._metrics[f"I(X;Z)/input_{name}"] = input_mi
                    self._metrics[f"I(X;Z)/output_{name}"] = output_mi
                    self._metrics[f"regularity/input_{name}"] = input_mi / entropy
                    self._metrics[f"regularity/output_{name}"] = (
                        output_mi / entropy
                    )

            def results(self):
                return dict(self._metrics)

        accumulator._accumulators = {
            hooks[0]: FakeLayerAccumulator(0.2, 0.1, 0.1),
            hooks[1]: FakeLayerAccumulator(0.8, 0.9, 0.45),
        }
        accumulator._sample_counts = {hook: 1 for hook in hooks}

        mean = accumulator.results()["mean"]

        self.assertAlmostEqual(mean["H(Z)"], 0.5)
        self.assertAlmostEqual(mean["I(X;Z)/input_unigram"], 0.5)
        self.assertAlmostEqual(mean["I(X;Z)/output_unigram"], 0.275)
        self.assertAlmostEqual(mean["regularity/input_unigram"], 1.0)
        self.assertAlmostEqual(mean["optimality/unigram"], 0.55)
        self.assertNotAlmostEqual(mean["optimality/unigram"], (1.0 + 0.5) / 2)

    def test_checkpoint_resume_matches_uninterrupted_updates(self):
        hooks = ("block_0_block_output", "block_1_block_output")
        batches = []
        for index in range(6):
            batch = _make_batch(10 + 20 * index, scale=1 + index / 10)
            batch[f"activation__{hooks[1]}"] = (
                batch[f"activation__{hooks[0]}"][:, ::-1] + index
            )
            batches.append(batch)

        uninterrupted = TaxActivationAccumulator(hooks, n_bins=8, seed=3)
        for batch in batches:
            uninterrupted.update(batch)

        before_restart = TaxActivationAccumulator(hooks, n_bins=8, seed=3)
        for batch in batches[:3]:
            before_restart.update(batch)
        state = before_restart.state_dict()
        resumed = TaxActivationAccumulator(hooks, n_bins=8, seed=3)
        resumed.load_state_dict(state)
        for batch in batches[3:]:
            resumed.update(batch)

        expected = uninterrupted.results()
        actual = resumed.results()
        for hook in hooks:
            self.assertEqual(
                actual["hooks"][hook]["n_samples"],
                expected["hooks"][hook]["n_samples"],
            )
            for name, value in expected["hooks"][hook].items():
                if isinstance(value, float):
                    self.assertTrue(
                        np.isclose(
                            actual["hooks"][hook][name],
                            value,
                            rtol=1e-6,
                            atol=1e-7,
                        ),
                        msg=f"{name} differs after restoring hook {hook}.",
                    )

    def test_checkpoint_preserves_repeated_unigram_tuple_labels(self):
        hook = "block_0_block_output"
        first = _make_batch(10, scale=1)
        second = _make_batch(10, scale=2)

        uninterrupted = TaxActivationAccumulator((hook,), n_bins=8, seed=3)
        uninterrupted.update(first)
        uninterrupted.update(second)

        before_restart = TaxActivationAccumulator((hook,), n_bins=8, seed=3)
        before_restart.update(first)
        resumed = TaxActivationAccumulator((hook,), n_bins=8, seed=3)
        resumed.load_state_dict(before_restart.state_dict())
        resumed.update(second)

        expected = uninterrupted.results()["hooks"][hook]
        actual = resumed.results()["hooks"][hook]
        for name, value in expected.items():
            if isinstance(value, float):
                self.assertTrue(
                    np.isclose(actual[name], value, rtol=1e-6, atol=1e-7),
                    msg=f"{name} differs after restoring repeated labels.",
                )

    def test_checkpoint_rejects_incompatible_seed(self):
        hook = "block_0_block_output"
        accumulator = TaxActivationAccumulator((hook,), n_bins=8, seed=3)
        accumulator.update(_make_batch(10))

        restored = TaxActivationAccumulator((hook,), n_bins=8, seed=4)
        with self.assertRaisesRegex(ValueError, "seed"):
            restored.load_state_dict(accumulator.state_dict())

    def test_checkpoint_rejects_tampered_counts(self):
        hook = "block_0_block_output"
        accumulator = TaxActivationAccumulator((hook,), n_bins=8, seed=3)
        accumulator.update(_make_batch(10))
        state = copy.deepcopy(accumulator.state_dict())
        state["accumulators"][hook]["counts"][0] = -1

        restored = TaxActivationAccumulator((hook,), n_bins=8, seed=3)
        with self.assertRaisesRegex(ValueError, "global counts"):
            restored.load_state_dict(state)

    def test_empty_context_is_skipped_without_creating_partial_state(self):
        hook = "block_0_block_output"
        batch = _make_batch(10)
        short_batch = {key: value[:6] for key, value in batch.items()}
        accumulator = TaxActivationAccumulator((hook,), n_bins=8)

        self.assertEqual(accumulator.update(short_batch), 0)
        with self.assertRaisesRegex(ValueError, "No positions"):
            accumulator.results()

    def test_nonfinite_selected_activation_is_rejected(self):
        hook = "block_0_block_output"
        batch = _make_batch(10)
        batch[f"activation__{hook}"][4, 0] = np.nan
        accumulator = TaxActivationAccumulator((hook,), n_bins=8)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            accumulator.update(batch)


if __name__ == "__main__":
    unittest.main()
