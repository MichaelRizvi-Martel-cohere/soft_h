import copy
import json

import numpy as np
import pytest
import torch

from examples.compare_entropy_results import compare_payloads
from examples.extract_hf_entropy import (
    _decoder_layers,
    _layer_output,
    _load_token_rows,
    build_unigram_labels,
    collect_results,
    load_artifact_manifest,
    token_rows_sha256,
)
from soft_entropy.accumulator import SoftEntropyAccumulator
from soft_entropy.tax_online import TaxActivationAccumulator

METRICS = (
    "H(Z)",
    "I(X;Z)/input_unigram",
    "I(X;Z)/output_unigram",
    "regularity/input_unigram",
    "regularity/output_unigram",
    "optimality/unigram",
)


_TOKENIZER_PATH = "gs://cohere-prod/encoders/releases/0.10.1/r2l255k.json"
_FAX_NUMERICS = {
    "attention_impl": "jax_native",
    "quantize_params": True,
    "quantize_activations": True,
    "quantize_residuals": True,
    "use_fp8_gemm": False,
}
_HF_NUMERICS = {
    "attention_implementation": "eager",
    "dtype": "torch.bfloat16",
}


def _payload(offsets=None):
    offsets = offsets or {}
    hooks = {}
    for layer in range(32):
        hooks[f"block_{layer}_block_output"] = {
            metric: 0.1 + layer / 100 + offsets.get((layer, metric), 0)
            for metric in METRICS
        }
        hooks[f"block_{layer}_block_output"].update(
            {"dimension": 4096, "n_samples": 12}
        )
    return {
        "n_bins": 100,
        "seed": 0,
        "label_types": ["unigram"],
        "soft_h_package_sha256": "a" * 64,
        "data_prefix_sha256": "b" * 64,
        "tokenizer_path": _TOKENIZER_PATH,
        "total_selected_samples": 12,
        "hooks": hooks,
        **_FAX_NUMERICS,
    }


def _hf_payload(offsets=None):
    payload = _payload(offsets)
    for name in _FAX_NUMERICS:
        del payload[name]
    payload.update(_HF_NUMERICS)
    return payload


def _write_manifest(directory, **overrides):
    manifest = {
        "token_ids_sha256": "c" * 64,
        "data_prefix_sha256": "b" * 64,
        "tokenizer_path": _TOKENIZER_PATH,
        "n_samples": 100,
        "max_sequence_length": 512,
    }
    manifest.update(overrides)
    for name, value in list(manifest.items()):
        if value is None:
            del manifest[name]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_load_artifact_manifest_returns_recorded_identity(tmp_path):
    expected = _write_manifest(tmp_path)

    manifest = load_artifact_manifest(
        str(tmp_path), n_samples=100, max_sequence_length=512
    )

    assert manifest == expected


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tokenizer_path": None}, "missing"),
        ({"token_ids_sha256": "NOTADIGEST"}, "SHA-256"),
        ({"tokenizer_path": ""}, "non-empty string"),
        ({"n_samples": 50}, "requested 100"),
        ({"max_sequence_length": 256}, "requested 512"),
    ],
)
def test_load_artifact_manifest_rejects_unusable_manifest(tmp_path, overrides, message):
    _write_manifest(tmp_path, **overrides)

    with pytest.raises((ValueError, TypeError), match=message):
        load_artifact_manifest(str(tmp_path), n_samples=100, max_sequence_length=512)


def test_load_token_rows_validates_exact_artifact_and_digest(tmp_path):
    path = tmp_path / "tokens.npz"
    expected_rows = [
        np.array([5, 4, 255001]),
        np.array([5, 2, 255001]),
    ]
    np.savez(
        path,
        input_ids=np.array([[5, 4, 255001, 0], [5, 2, 255001, 0]]),
        lengths=np.array([3, 3]),
        source_indices=np.array([3, 9]),
    )
    expected_digest = token_rows_sha256(expected_rows)

    rows, observed_digest = _load_token_rows(
        str(path),
        n_samples=2,
        max_sequence_length=4,
        expected_token_sha256=expected_digest,
    )

    np.testing.assert_array_equal(rows[0], expected_rows[0])
    np.testing.assert_array_equal(rows[1], expected_rows[1])
    assert observed_digest == expected_digest
    assert token_rows_sha256(rows) == token_rows_sha256(copy.deepcopy(rows))
    assert token_rows_sha256(rows) != token_rows_sha256(reversed(rows))


def test_build_unigram_labels_aligns_current_and_next_tokens():
    labels, slices = build_unigram_labels([np.array([5, 10, 11]), np.array([5, 20])])

    np.testing.assert_array_equal(labels["input_unigram"], [[5], [10], [5]])
    np.testing.assert_array_equal(labels["output_unigram"], [[10], [11], [20]])
    assert slices == [slice(0, 2), slice(2, 3)]


def test_decoder_layer_discovery_and_output_selection():
    class _BaseModel:
        def __init__(self):
            self.layers = [object(), object()]

    class _CausalModel:
        def __init__(self):
            self.model = _BaseModel()

    model = _CausalModel()
    assert _decoder_layers(model, expected_layers=2) == model.model.layers
    hidden = torch.zeros((2, 3, 4))
    assert _layer_output((hidden, "cache")) is hidden
    with pytest.raises(ValueError, match=r"shape \[B, S, D\]"):
        _layer_output(torch.zeros((2, 3)))


def test_hf_and_tax_accumulation_paths_agree_on_identical_activations():
    hook = "block_0_block_output"
    token_row = np.array([5, 10, 11, 12])
    activations = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    labels, _ = build_unigram_labels([token_row])

    hf_accumulator = SoftEntropyAccumulator(d=4, n_bins=8, seed=3)
    hf_accumulator.update(activations, labels=labels)
    hf_results = collect_results([hf_accumulator])[hook]

    tax_accumulator = TaxActivationAccumulator(
        (hook,),
        n_bins=8,
        seed=3,
        label_types=("unigram",),
    )
    tax_accumulator.update(
        {
            "input_token": token_row[:-1],
            "output_token": token_row[1:],
            "batch_row": np.zeros(3, dtype=np.int64),
            "sequence_id": np.zeros(3, dtype=np.int64),
            "position": np.arange(3),
            f"activation__{hook}": activations,
        }
    )
    tax_results = tax_accumulator.results()["hooks"][hook]

    for metric in METRICS:
        assert hf_results[metric] == pytest.approx(tax_results[metric], abs=1e-12)
    assert hf_results["n_samples"] == tax_results["n_samples"] == 3


def test_compare_payloads_passes_identical_results():
    comparison = compare_payloads(_payload(), _hf_payload())

    assert comparison["passed"] is True
    assert comparison["n_layers"] == 32
    assert comparison["n_selected_positions"] == 12
    for summary in comparison["metric_summaries"].values():
        assert summary["max_abs_error"] == 0
        assert summary["layer_trajectory_correlation"] == 1


def test_compare_payloads_fails_large_gated_layer_error():
    hf = _hf_payload({(17, "H(Z)"): 0.02})

    comparison = compare_payloads(_payload(), hf)

    assert comparison["passed"] is False
    assert comparison["metric_summaries"]["H(Z)"]["max_abs_error"] == pytest.approx(
        0.02
    )


def test_compare_payloads_reports_but_does_not_gate_optimality():
    hf = _hf_payload({(17, "optimality/unigram"): 0.02})

    comparison = compare_payloads(_payload(), hf)

    assert comparison["passed"] is True
    assert comparison["metric_summaries"]["optimality/unigram"]["passed"] is False
    assert comparison["metric_summaries"]["optimality/unigram"]["gated"] is False


def test_compare_payloads_rejects_selected_position_mismatch():
    hf = _hf_payload()
    hf["total_selected_samples"] = 13

    with pytest.raises(ValueError, match="selected-position counts differ"):
        compare_payloads(_payload(), hf)


@pytest.mark.parametrize(
    "field, value",
    [
        ("attention_impl", "fax_fa3"),
        ("quantize_residuals", False),
        ("use_fp8_gemm", True),
    ],
)
def test_compare_payloads_rejects_incomparable_fax_numerics(field, value):
    fax = _payload()
    fax[field] = value

    with pytest.raises(ValueError, match="Fax result was not produced"):
        compare_payloads(fax, _hf_payload())


@pytest.mark.parametrize(
    "field, value",
    [
        ("attention_implementation", "sdpa"),
        ("dtype", "torch.float32"),
    ],
)
def test_compare_payloads_rejects_incomparable_hf_numerics(field, value):
    hf = _hf_payload()
    hf[field] = value

    with pytest.raises(ValueError, match="Hugging Face result was not produced"):
        compare_payloads(_payload(), hf)


def test_compare_payloads_rejects_missing_fax_numerics():
    fax = _payload()
    del fax["attention_impl"]

    with pytest.raises(ValueError, match="Fax result was not produced"):
        compare_payloads(fax, _hf_payload())


def test_compare_payloads_rejects_tokenizer_mismatch():
    hf = _hf_payload()
    hf["tokenizer_path"] = "gs://cohere-prod/encoders/releases/0.9.0/other.json"

    with pytest.raises(ValueError, match="same tokenizer"):
        compare_payloads(_payload(), hf)
