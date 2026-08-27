import numpy as np
import pytest
import torch

from examples.compare_logits import _valid_prediction_mask, compute_metrics
from examples.extract_hf_logits import _load_npz, _write_npz, select_hf_logits


def test_compute_metrics_identical_logits():
    logits = np.array(
        [
            [1.0, 2.0, -1.0, 0.5],
            [-2.0, 0.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )

    metrics = compute_metrics(logits, logits.copy(), top_k=2)

    assert metrics["argmax_agreement"] == 1.0
    assert metrics["top_k_overlap_mean"] == 1.0
    assert metrics["raw_max_abs_error"] == 0.0
    assert metrics["centered_max_abs_error"] == 0.0
    assert metrics["cosine_similarity_min"] == pytest.approx(1.0)
    assert metrics["centered_cosine_similarity_min"] == pytest.approx(1.0)
    assert metrics["kl_fax_to_hf_max"] == pytest.approx(0.0)
    assert metrics["kl_hf_to_fax_max"] == pytest.approx(0.0)


def test_compute_metrics_distinguishes_additive_offsets_from_distributions():
    fax = np.array([[1.0, 3.0, -2.0]], dtype=np.float32)
    hf = fax + 7.0

    metrics = compute_metrics(fax, hf, top_k=2)

    assert metrics["argmax_agreement"] == 1.0
    assert metrics["raw_max_abs_error"] == 7.0
    assert metrics["centered_max_abs_error"] == pytest.approx(0.0)
    assert metrics["kl_fax_to_hf_max"] == pytest.approx(0.0)
    assert metrics["kl_hf_to_fax_max"] == pytest.approx(0.0)


def test_compute_metrics_detects_distribution_mismatch():
    fax = np.array([[5.0, 0.0, -1.0]], dtype=np.float32)
    hf = np.array([[-1.0, 0.0, 5.0]], dtype=np.float32)

    metrics = compute_metrics(fax, hf, top_k=1)

    assert metrics["argmax_agreement"] == 0.0
    assert metrics["top_k_overlap_mean"] == 0.0
    assert metrics["centered_max_abs_error"] > 0
    assert metrics["kl_fax_to_hf_mean"] > 0


def test_compute_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same"):
        compute_metrics(np.zeros((1, 3)), np.zeros((2, 3)))


def test_valid_prediction_mask_excludes_padding_targets():
    mask = _valid_prediction_mask(
        {
            "logits": np.zeros((3, 5), dtype=np.float32),
            "selected_target_tokens": np.array([10, 20, 0]),
        }
    )

    np.testing.assert_array_equal(mask, [True, True, False])


def test_select_hf_logits_matches_numpy_indexing():
    logits = torch.arange(2 * 4 * 5, dtype=torch.bfloat16).reshape(2, 4, 5)
    rows = np.array([0, 0, 1], dtype=np.int64)
    positions = np.array([0, 3, 2], dtype=np.int64)

    selected = select_hf_logits(logits, rows, positions)

    np.testing.assert_array_equal(selected, logits[rows, positions].float().numpy())


def test_hf_artifact_io_round_trip(tmp_path):
    path = str(tmp_path / "logits.npz")
    arrays = {"logits": np.arange(12, dtype=np.float32).reshape(3, 4)}

    _write_npz(path, arrays)

    loaded = _load_npz(path)
    np.testing.assert_array_equal(loaded["logits"], arrays["logits"])
    with pytest.raises(FileExistsError):
        _write_npz(path, arrays)
