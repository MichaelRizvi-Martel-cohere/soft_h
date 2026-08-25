"""
Backend-agnostic soft entropy accumulator.

Accumulates soft bin assignment counts across batches so that entropy and
mutual information can be estimated over large datasets without storing all
embeddings in memory. Reference points are sampled once at construction and
reused across batches.

Multiple named label sets can be passed to update() in a single forward pass,
producing a separate conditional entropy and mutual information estimate for
each one.

Supports 'numpy', 'torch', and 'jax' backends.

Usage::

    acc = SoftEntropyAccumulator(d=768, backend='torch')
    for z, token_ids, bigram_ids, pref_labels in dataloader:
        acc.update(z, labels={
            "token":    token_ids,
            "bigram":   bigram_ids,
            "preference": pref_labels,
        })
    print(acc.entropy())
    print(acc.mutual_information())          # dict keyed by label-set name
    print(acc.conditional_entropy())         # dict of dicts
    print(acc.results())                     # flat summary dict (H(Z), I(X;Z), regularity per label set)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from soft_entropy.temp_calibration import find_eps

_STATE_SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


def _get_ops(backend: str, dtype=None):
    """Return a namespace of array operations for the requested backend."""
    if backend == "numpy":
        import numpy as np

        class NumpyOps:
            def zeros(self, shape):
                return np.zeros(shape, dtype=np.float32)

            def normalize(self, x):
                return x / np.linalg.norm(x, axis=-1, keepdims=True)

            def randn(self, shape, seed):
                return (
                    np.random.default_rng(seed)
                    .standard_normal(shape)
                    .astype(np.float32)
                )

            def softmax(self, x, temp):
                x = x / temp
                x -= x.max(axis=-1, keepdims=True)
                e = np.exp(x)
                return e / e.sum(axis=-1, keepdims=True)

            def matmul(self, a, b):
                return a @ b

            def sum_axis0(self, x):
                return x.sum(axis=0)

            def to_device(self, x):
                return x

            def to_numpy(self, x):
                return x

        return NumpyOps()

    elif backend == "torch":
        import torch
        import torch.nn.functional as F

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        dtype = dtype or torch.float32

        class TorchOps:
            def zeros(self, shape):
                return torch.zeros(shape, device=device, dtype=dtype)

            def normalize(self, x):
                return F.normalize(x, dim=-1)

            def randn(self, shape, seed):
                g = torch.Generator(device=device).manual_seed(seed)
                return torch.randn(*shape, generator=g, device=device, dtype=dtype)

            def softmax(self, x, temp):
                return F.softmax(x / temp, dim=-1)

            def matmul(self, a, b):
                return a @ b

            def sum_axis0(self, x):
                return x.sum(0)

            def to_device(self, x):
                return x.to(device=device, dtype=dtype)

            def to_numpy(self, x):
                return x.detach().cpu().float().numpy()

        return TorchOps()

    elif backend == "jax":
        import jax
        import jax.numpy as jnp

        jax_dtype = dtype or jnp.float32

        class JaxOps:
            def zeros(self, shape):
                return jnp.zeros(shape, dtype=jax_dtype)

            def normalize(self, x):
                return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

            def randn(self, shape, seed):
                return jax.random.normal(
                    jax.random.PRNGKey(seed), shape=shape, dtype=jax_dtype
                )

            def softmax(self, x, temp):
                return jax.nn.softmax(x / temp, axis=-1)

            def matmul(self, a, b):
                return a @ b

            def sum_axis0(self, x):
                return x.sum(axis=0)

            def to_device(self, x):
                return x.astype(
                    jax_dtype
                )  # JAX places arrays on GPU automatically via XLA

            def to_numpy(self, x):
                import numpy as np

                return np.array(x, dtype=np.float32)

        return JaxOps()

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose 'numpy', 'torch', or 'jax'."
        )


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class SoftEntropyAccumulator:
    """
    Accumulates soft bin assignment counts across batches.

    Multiple label sets can be passed as a dict to update(), e.g.::

        acc.update(z, labels={
            "token":      token_labels,
            "bigram":     bigram_labels,
            "preference": pref_labels,
        })

    A plain 1-D array is also accepted and stored under the key "labels".

    Args:
        d:       embedding dimensionality
        n_bins:  number of random reference points on the unit sphere
        seed:    random seed for reference point sampling (fixed across batches)
        backend: one of 'numpy', 'torch', 'jax'
        dtype:   torch dtype for reference points and counts (torch backend only);
                 defaults to torch.float32. Pass e.g. torch.bfloat16 to match a
                 model that outputs bfloat16 and avoid casting overhead.
    """

    def __init__(
        self,
        d: int,
        n_bins: int = 100,
        seed: int = 0,
        backend: str = "numpy",
        dtype=None,
    ):
        self.d = d
        self.n_bins = n_bins
        self.seed = seed
        self.backend = backend
        self.ops = _get_ops(backend, dtype=dtype)

        # temperature calibrated per dimensionality (eq. 7 in the paper)
        self.temp = find_eps(n_bins, d)

        # reference points — sampled once, reused every batch
        w = self.ops.randn((n_bins, d), seed)
        self.w = self.ops.normalize(w)  # [n_bins, d]

        self._reset_counts()

    def _reset_counts(self):
        self._counts = self.ops.zeros((self.n_bins,))
        self._n_samples = 0

        # nested: label_set_name -> label_value -> bin counts
        self._label_counts: dict[str, dict[Any, Any]] = defaultdict(
            lambda: defaultdict(lambda: self.ops.zeros((self.n_bins,)))
        )
        self._label_n_samples: dict[str, dict[Any, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def reset(self):
        """Clear all accumulated counts."""
        self._reset_counts()

    @property
    def n_samples(self) -> int:
        """Number of embeddings accumulated so far."""
        return self._n_samples

    def state_dict(self) -> dict[str, Any]:
        """Return a backend-independent, numeric checkpoint state."""
        import numpy as np

        label_counts: dict[str, dict[str, Any]] = {}
        for set_name, counts_by_label in self._label_counts.items():
            ordered_items = sorted(
                counts_by_label.items(), key=lambda item: repr(item[0])
            )
            labels_are_tuples = {isinstance(label, tuple) for label, _ in ordered_items}
            if len(labels_are_tuples) != 1:
                raise TypeError(
                    f"Label set {set_name!r} mixes scalar and tuple labels."
                )
            label_rows = [
                list(label) if isinstance(label, tuple) else [label]
                for label, _ in ordered_items
            ]
            labels = np.asarray(label_rows)
            if labels.dtype == np.dtype("O"):
                raise TypeError(
                    f"Label set {set_name!r} contains values that cannot be checkpointed safely."
                )
            label_counts[set_name] = {
                "labels": labels,
                "labels_are_tuples": labels_are_tuples.pop(),
                "counts": np.stack(
                    [self.ops.to_numpy(counts) for _, counts in ordered_items]
                ),
                "n_samples": np.asarray(
                    [
                        self._label_n_samples[set_name][label]
                        for label, _ in ordered_items
                    ],
                    dtype=np.int64,
                ),
            }

        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "dimension": self.d,
            "n_bins": self.n_bins,
            "seed": self.seed,
            "backend": self.backend,
            "temperature": self.temp,
            "reference_points": self.ops.to_numpy(self.w),
            "counts": self.ops.to_numpy(self._counts),
            "n_samples": self._n_samples,
            "label_counts": label_counts,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore and validate a state produced by :meth:`state_dict`."""
        import numpy as np

        expected_keys = {
            "schema_version",
            "dimension",
            "n_bins",
            "seed",
            "backend",
            "temperature",
            "reference_points",
            "counts",
            "n_samples",
            "label_counts",
        }
        if set(state) != expected_keys:
            raise ValueError(
                "Accumulator checkpoint fields differ from the expected schema: "
                f"expected {sorted(expected_keys)}, got {sorted(state)}."
            )
        expected_metadata = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "dimension": self.d,
            "n_bins": self.n_bins,
            "seed": self.seed,
            "backend": self.backend,
        }
        for name, expected in expected_metadata.items():
            if state[name] != expected:
                raise ValueError(
                    f"Accumulator checkpoint {name}={state[name]!r} does not match "
                    f"the configured value {expected!r}."
                )
        if not np.isclose(state["temperature"], self.temp, rtol=0, atol=1e-12):
            raise ValueError("Accumulator checkpoint temperature does not match.")

        reference_points = np.asarray(state["reference_points"], dtype=np.float32)
        expected_reference_points = self.ops.to_numpy(self.w)
        if (
            reference_points.shape != (self.n_bins, self.d)
            or not np.isfinite(reference_points).all()
            or not np.array_equal(reference_points, expected_reference_points)
        ):
            raise ValueError(
                "Accumulator checkpoint reference points do not match the configured seed."
            )

        counts = np.asarray(state["counts"], dtype=np.float32)
        n_samples = state["n_samples"]
        if not isinstance(n_samples, int) or n_samples < 0:
            raise ValueError(
                f"Accumulator checkpoint n_samples must be non-negative, got {n_samples!r}."
            )
        if (
            counts.shape != (self.n_bins,)
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
            or not np.isclose(counts.sum(), n_samples, rtol=1e-4, atol=1e-3)
        ):
            raise ValueError("Accumulator checkpoint global counts are invalid.")
        label_counts = state["label_counts"]
        if not isinstance(label_counts, dict):
            raise TypeError("Accumulator checkpoint label_counts must be a dictionary.")

        self._reset_counts()
        self.w = self.ops.to_device(reference_points)
        self._counts = self.ops.to_device(counts)
        self._n_samples = n_samples
        for set_name, label_state in label_counts.items():
            if not isinstance(set_name, str) or not isinstance(label_state, dict):
                raise TypeError("Accumulator checkpoint label state is invalid.")
            if set(label_state) != {
                "labels",
                "labels_are_tuples",
                "counts",
                "n_samples",
            }:
                raise ValueError(
                    f"Accumulator checkpoint label set {set_name!r} has invalid fields."
                )
            labels = np.asarray(label_state["labels"])
            labels_are_tuples = label_state["labels_are_tuples"]
            conditional_counts = np.asarray(label_state["counts"], dtype=np.float32)
            label_n_samples = np.asarray(label_state["n_samples"])
            if (
                not isinstance(labels_are_tuples, bool)
                or labels.ndim != 2
                or labels.shape[1] < 1
                or labels.dtype == np.dtype("O")
                or conditional_counts.shape != (labels.shape[0], self.n_bins)
                or label_n_samples.shape != (labels.shape[0],)
                or not np.issubdtype(label_n_samples.dtype, np.integer)
                or not np.isfinite(conditional_counts).all()
                or np.any(conditional_counts < 0)
                or np.any(label_n_samples < 0)
                or not np.allclose(
                    conditional_counts.sum(axis=1),
                    label_n_samples,
                    rtol=1e-4,
                    atol=1e-3,
                )
            ):
                raise ValueError(
                    f"Accumulator checkpoint label set {set_name!r} is invalid."
                )

            restored_labels: list[Any] = []
            for row in labels:
                values = tuple(value.item() for value in row)
                restored_labels.append(values if labels_are_tuples else values[0])
            if len(set(restored_labels)) != len(restored_labels):
                raise ValueError(
                    f"Accumulator checkpoint label set {set_name!r} contains duplicates."
                )
            for index, label in enumerate(restored_labels):
                self._label_counts[set_name][label] = self.ops.to_device(
                    conditional_counts[index]
                )
                self._label_n_samples[set_name][label] = int(label_n_samples[index])

    def update(self, z: Any, labels: Any = None):
        """
        Accumulate soft assignments for a batch of embeddings.

        Args:
            z:      embeddings [batch, d]
            labels: one of:
                    - None: accumulate H(Z) only
                    - 1-D array/tensor: single label set, stored as "labels"
                    - dict[str, array]: multiple named label sets, each a 1-D
                      array/tensor of the same length as z
        """
        ops = self.ops

        z = ops.to_device(z)
        z_norm = ops.normalize(z)  # [batch, d]
        scores = ops.matmul(z_norm, self.w.T)  # [batch, n_bins]
        p = ops.softmax(scores, self.temp)  # [batch, n_bins]

        self._counts = self._counts + ops.sum_axis0(p)
        self._n_samples += z.shape[0]

        if labels is None:
            return

        # normalise to dict
        label_sets = labels if isinstance(labels, dict) else {"labels": labels}

        for set_name, set_labels in label_sets.items():
            for label_val, idx in self._group_by_label(set_labels):
                self._label_counts[set_name][label_val] = self._label_counts[set_name][
                    label_val
                ] + ops.sum_axis0(p[idx])
                self._label_n_samples[set_name][label_val] += len(idx)

    # ------------------------------------------------------------------
    # Entropy / MI
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """H(Z): efficiency-normalized entropy of the full embedding distribution."""
        return self._h_from_counts(self._counts)

    def conditional_entropy(self) -> dict[str, dict[Any, float]]:
        """
        H(Z | X=x) for each label value x, for each label set.

        Returns a nested dict: label_set_name -> label_value -> entropy.
        """
        return {
            set_name: {
                label_val: self._h_from_counts(counts)
                for label_val, counts in label_vals.items()
            }
            for set_name, label_vals in self._label_counts.items()
        }

    def mutual_information(self) -> dict[str, float]:
        """
        I(X; Z) for each label set.

        Returns a dict: label_set_name -> mutual information.
        """
        h_z = self.entropy()
        cond_h = self.conditional_entropy()
        result = {}
        for set_name, label_hs in cond_h.items():
            expected_cond_h = sum(
                (self._label_n_samples[set_name][lv] / self._n_samples) * h
                for lv, h in label_hs.items()
            )
            result[set_name] = h_z - expected_cond_h
        return result

    def results(self) -> dict[str, float]:
        """
        Flat summary dict with H(Z), and per-label-set I(X;Z) and regularity.
        """
        h = self.entropy()
        out: dict[str, float] = {"H(Z)": h}
        for set_name, mi in self.mutual_information().items():
            out[f"I(X;Z)/{set_name}"] = mi
            out[f"regularity/{set_name}"] = (mi / h) if h > 0 else float("nan")
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _h_from_counts(self, counts) -> float:
        import numpy as np

        p = self.ops.to_numpy(counts)
        p = p / p.sum()
        h = -(p * np.log(np.clip(p, 1e-9, None))).sum()
        return float(h / math.log(self.n_bins))

    def _group_by_label(self, labels):
        """Yield (label_value, indices) pairs, backend-agnostically."""
        import numpy as np

        if hasattr(labels, "detach"):  # torch
            labels_np = labels.detach().cpu().numpy()
        elif hasattr(labels, "devices"):  # jax
            labels_np = np.array(labels)
        else:
            labels_np = np.asarray(labels)
        if labels_np.ndim == 1:
            for label in np.unique(labels_np):
                yield label, np.where(labels_np == label)[0]
            return
        if labels_np.ndim == 2:
            unique_labels, inverse = np.unique(labels_np, axis=0, return_inverse=True)
            for label_index, label in enumerate(unique_labels):
                yield tuple(label.tolist()), np.where(inverse == label_index)[0]
            return
        raise ValueError(
            f"Labels must be one- or two-dimensional, got shape {labels_np.shape}."
        )
