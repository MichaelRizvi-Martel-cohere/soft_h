"""Online soft-entropy accumulation for aligned Tax activation batches."""

from __future__ import annotations

from typing import Any

import numpy as np

from soft_entropy.accumulator import SoftEntropyAccumulator

NGRAM_ORDERS = {"unigram": 1, "bigram": 2, "trigram": 3, "quadgram": 4}
_ACTIVATION_PREFIX = "activation__"
_STATE_SCHEMA_VERSION = 1
_METADATA_KEYS = {
    "input_token",
    "output_token",
    "batch_row",
    "sequence_id",
    "position",
}


def build_ngram_labels(
    input_token: np.ndarray,
    output_token: np.ndarray,
    batch_row: np.ndarray,
    sequence_id: np.ndarray,
    position: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select positions with complete quadgram context and build tuple labels."""
    selected_indices: list[int] = []
    label_rows: dict[str, list[list[int]]] = {
        f"{direction}_{name}": []
        for name in NGRAM_ORDERS
        for direction in ("input", "output")
    }
    sequence_keys = np.stack([batch_row, sequence_id], axis=-1)
    for current_sequence_key in np.unique(sequence_keys, axis=0):
        sequence_indices = np.flatnonzero(
            np.all(sequence_keys == current_sequence_key, axis=-1)
        )
        sequence_indices = sequence_indices[np.argsort(position[sequence_indices])]
        sequence_positions = position[sequence_indices]
        if not np.array_equal(sequence_positions, np.arange(sequence_indices.size)):
            raise ValueError(
                f"Sequence {current_sequence_key.tolist()} positions must be contiguous from zero, "
                f"got {sequence_positions}."
            )

        sequence_inputs = input_token[sequence_indices]
        sequence_outputs = output_token[sequence_indices]
        if sequence_inputs.size > 1 and not np.array_equal(
            sequence_inputs[1:], sequence_outputs[:-1]
        ):
            raise ValueError(
                f"Sequence {current_sequence_key.tolist()} has inconsistent shifted token labels."
            )
        tokens = np.concatenate([sequence_inputs, sequence_outputs[-1:]])
        max_order = max(NGRAM_ORDERS.values())
        for token_position in range(max_order - 1, tokens.size - max_order):
            selected_indices.append(int(sequence_indices[token_position]))
            for name, order in NGRAM_ORDERS.items():
                label_rows[f"input_{name}"].append(
                    tokens[token_position - order + 1 : token_position + 1].tolist()
                )
                label_rows[f"output_{name}"].append(
                    tokens[token_position + 1 : token_position + order + 1].tolist()
                )

    return np.asarray(selected_indices, dtype=np.int64), {
        name: np.asarray(rows, dtype=np.int64) for name, rows in label_rows.items()
    }


class TaxActivationAccumulator:
    """Accumulate entropy statistics from one aligned Tax batch at a time."""

    def __init__(
        self,
        hooks: tuple[str, ...] | list[str],
        n_bins: int = 100,
        seed: int = 0,
        backend: str = "numpy",
    ) -> None:
        if not hooks:
            raise ValueError("At least one hook must be provided.")
        if len(set(hooks)) != len(hooks):
            raise ValueError(f"Hooks must be unique, got {hooks}.")
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}.")

        self.hooks = tuple(hooks)
        self.n_bins = n_bins
        self.seed = seed
        self.backend = backend
        self._accumulators: dict[str, SoftEntropyAccumulator] = {}
        self._sample_counts = {hook: 0 for hook in self.hooks}

    def update(self, arrays: dict[str, Any]) -> int:
        """Update all hook accumulators from one flattened, aligned batch."""
        required_keys = {
            *_METADATA_KEYS,
            *(_ACTIVATION_PREFIX + hook for hook in self.hooks),
        }
        missing_keys = required_keys - arrays.keys()
        if missing_keys:
            raise ValueError(
                f"Activation batch is missing keys {sorted(missing_keys)}."
            )

        input_token = np.asarray(arrays["input_token"])
        output_token = np.asarray(arrays["output_token"])
        if input_token.ndim != 1 or output_token.shape != input_token.shape:
            raise ValueError(
                "Batch labels must be aligned 1-D arrays, got "
                f"{input_token.shape} and {output_token.shape}."
            )

        batch_row = np.asarray(arrays["batch_row"])
        sequence_id = np.asarray(arrays["sequence_id"])
        position = np.asarray(arrays["position"])
        if (
            batch_row.shape != input_token.shape
            or sequence_id.shape != input_token.shape
            or position.shape != input_token.shape
        ):
            raise ValueError(
                "Batch sequence metadata must align with tokens, got "
                f"{batch_row.shape}, {sequence_id.shape}, {position.shape}, "
                f"and {input_token.shape}."
            )

        selected_indices, labels = build_ngram_labels(
            input_token,
            output_token,
            batch_row,
            sequence_id,
            position,
        )
        if selected_indices.size == 0:
            return 0

        for hook in self.hooks:
            activation = np.asarray(arrays[_ACTIVATION_PREFIX + hook])
            if activation.ndim != 2 or activation.shape[0] != input_token.shape[0]:
                raise ValueError(
                    f"Activation {hook!r} must be [N, D] and align with labels; "
                    f"got {activation.shape} and {input_token.shape}."
                )
            if not np.isfinite(activation).all():
                raise ValueError(f"Activation {hook!r} contains non-finite values.")

            selected_activation = activation[selected_indices]
            if np.any(np.linalg.norm(selected_activation, axis=-1) == 0):
                raise ValueError(
                    f"Activation {hook!r} contains zero-norm selected rows."
                )

            if hook not in self._accumulators:
                self._accumulators[hook] = SoftEntropyAccumulator(
                    d=activation.shape[-1],
                    n_bins=self.n_bins,
                    seed=self.seed,
                    backend=self.backend,
                )
            elif self._accumulators[hook].w.shape[-1] != activation.shape[-1]:
                raise ValueError(
                    f"Activation dimension changed across batches for hook {hook!r}."
                )

            self._accumulators[hook].update(selected_activation, labels=labels)
            self._sample_counts[hook] += selected_activation.shape[0]

        return int(selected_indices.size)

    def state_dict(self) -> dict[str, Any]:
        """Return a safe numeric checkpoint of every hook accumulator."""
        missing_hooks = set(self.hooks) - self._accumulators.keys()
        if missing_hooks:
            raise ValueError(
                f"Cannot checkpoint hooks without accumulated state: {sorted(missing_hooks)}."
            )
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "hooks": list(self.hooks),
            "n_bins": self.n_bins,
            "seed": self.seed,
            "backend": self.backend,
            "sample_counts": dict(self._sample_counts),
            "accumulators": {
                hook: self._accumulators[hook].state_dict() for hook in self.hooks
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint after validating its full experiment identity."""
        expected_keys = {
            "schema_version",
            "hooks",
            "n_bins",
            "seed",
            "backend",
            "sample_counts",
            "accumulators",
        }
        if set(state) != expected_keys:
            raise ValueError(
                "Tax accumulator checkpoint fields differ from the expected schema."
            )
        expected_metadata = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "hooks": list(self.hooks),
            "n_bins": self.n_bins,
            "seed": self.seed,
            "backend": self.backend,
        }
        for name, expected in expected_metadata.items():
            if state[name] != expected:
                raise ValueError(
                    f"Tax accumulator checkpoint {name}={state[name]!r} does not "
                    f"match the configured value {expected!r}."
                )

        sample_counts = state["sample_counts"]
        accumulator_states = state["accumulators"]
        if (
            not isinstance(sample_counts, dict)
            or set(sample_counts) != set(self.hooks)
            or not isinstance(accumulator_states, dict)
            or set(accumulator_states) != set(self.hooks)
        ):
            raise ValueError("Tax accumulator checkpoint hook state is incomplete.")

        restored_accumulators: dict[str, SoftEntropyAccumulator] = {}
        restored_sample_counts: dict[str, int] = {}
        for hook in self.hooks:
            hook_state = accumulator_states[hook]
            if not isinstance(hook_state, dict):
                raise TypeError(
                    f"Tax accumulator checkpoint for hook {hook!r} must be a dictionary."
                )
            sample_count = sample_counts[hook]
            if not isinstance(sample_count, int) or sample_count < 0:
                raise ValueError(
                    f"Tax accumulator sample count for hook {hook!r} is invalid."
                )
            dimension = hook_state.get("dimension")
            if not isinstance(dimension, int) or dimension < 1:
                raise ValueError(
                    f"Tax accumulator dimension for hook {hook!r} is invalid."
                )
            accumulator = SoftEntropyAccumulator(
                d=dimension,
                n_bins=self.n_bins,
                seed=self.seed,
                backend=self.backend,
            )
            accumulator.load_state_dict(hook_state)
            if accumulator.n_samples != sample_count:
                raise ValueError(
                    f"Tax accumulator sample count for hook {hook!r} is inconsistent."
                )
            restored_accumulators[hook] = accumulator
            restored_sample_counts[hook] = sample_count

        self._accumulators = restored_accumulators
        self._sample_counts = restored_sample_counts

    def results(self) -> dict[str, Any]:
        """Return per-hook and uniformly averaged paper-aligned metrics."""
        missing_hooks = set(self.hooks) - self._accumulators.keys()
        if missing_hooks:
            raise ValueError(
                "No positions with complete quadgram context were accumulated for "
                f"hooks {sorted(missing_hooks)}."
            )

        hook_results: dict[str, dict[str, Any]] = {}
        for hook in self.hooks:
            accumulator = self._accumulators[hook]
            metrics = accumulator.results()
            for name in NGRAM_ORDERS:
                input_mi = metrics[f"I(X;Z)/input_{name}"]
                output_mi = metrics[f"I(X;Z)/output_{name}"]
                metrics[f"optimality/{name}"] = (
                    output_mi / input_mi if input_mi > 0 else float("nan")
                )
            hook_results[hook] = {
                **metrics,
                "dimension": int(accumulator.w.shape[-1]),
                "n_samples": self._sample_counts[hook],
            }

        base_metric_keys = [
            key
            for key, value in hook_results[self.hooks[0]].items()
            if isinstance(value, float)
            and not key.startswith(("regularity/", "optimality/"))
        ]
        mean_metrics = {
            key: sum(hook_results[hook][key] for hook in self.hooks)
            / len(self.hooks)
            for key in base_metric_keys
        }
        for name in NGRAM_ORDERS:
            for direction in ("input", "output"):
                label_name = f"{direction}_{name}"
                mutual_information = mean_metrics[f"I(X;Z)/{label_name}"]
                mean_metrics[f"regularity/{label_name}"] = (
                    mutual_information / mean_metrics["H(Z)"]
                    if mean_metrics["H(Z)"] > 0
                    else float("nan")
                )
            input_mi = mean_metrics[f"I(X;Z)/input_{name}"]
            output_mi = mean_metrics[f"I(X;Z)/output_{name}"]
            mean_metrics[f"optimality/{name}"] = (
                output_mi / input_mi if input_mi > 0 else float("nan")
            )
        return {
            "hooks": hook_results,
            "mean": mean_metrics,
        }
