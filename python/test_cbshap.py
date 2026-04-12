"""Round-trip tests for the CBSHAP v1 sidecar writer + reader.

Trains a small CatBoost model, precomputes its SHAP lookup table, writes
the CBSHAP binary, reads it back, and asserts the result matches the
reference TreeSHAP implementation within the float16 quantization limit.

Run with: ``cd python && uv run pytest test_cbshap.py``
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cbshap import CBSHAPReader, write_cbshap
from tree_shap import TreeSHAPExplainer


def _train_tiny_model(
    n_features: int = 5,
    depth: int = 3,
    n_trees: int = 20,
    n_classes: int = 3,
    n_samples: int = 200,
    seed: int = 0,
) -> tuple:
    """Train a tiny CatBoost model and return its JSON export."""
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, n_features)).astype(np.float32)
    w = rng.normal(size=(n_features, n_classes))
    y = (X @ w).argmax(axis=1).astype(np.int64)

    model = CatBoostClassifier(
        iterations=n_trees,
        depth=depth,
        learning_rate=0.3,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        model.save_model(tmp.name, format="json")
        tmp_path = Path(tmp.name)
    try:
        json_model = json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink()
    return model, json_model, X, y


@pytest.fixture(scope="module")
def tiny_setup():
    """Expensive-ish fixture; reused across tests."""
    _, jm, X, _ = _train_tiny_model()
    explainer = TreeSHAPExplainer(jm)
    shap_table, tree_expected = explainer.precompute_shap_table()
    return explainer, shap_table, tree_expected, X


def _write(path, explainer, shap_table, tree_expected):
    write_cbshap(
        path,
        n_features=explainer.n_features,
        n_classes=explainer.n_classes,
        depth=explainer.depth,
        tree_features=explainer.tree_features,
        tree_borders=explainer.tree_borders,
        shap_table=shap_table,
        tree_expected=tree_expected,
        scale=explainer.scale,
        bias=explainer.bias,
    )


def test_write_then_read_roundtrip(tmp_path, tiny_setup):
    """Write CBSHAP, read it back, verify header + dimensions survive."""
    explainer, shap_table, tree_expected, _ = tiny_setup
    out = tmp_path / "tiny.cbshap"
    _write(out, explainer, shap_table, tree_expected)
    reader = CBSHAPReader(out)
    assert reader.version == 1
    assert reader.n_trees == explainer.n_trees
    assert reader.depth == explainer.depth
    assert reader.n_features == explainer.n_features
    assert reader.n_classes == explainer.n_classes


def test_shap_matches_reference_within_float16_tolerance(tmp_path, tiny_setup):
    """CBSHAP lookup should match Algorithm 2 within float16 precision."""
    explainer, shap_table, tree_expected, X = tiny_setup
    out = tmp_path / "tiny.cbshap"
    _write(out, explainer, shap_table, tree_expected)
    reader = CBSHAPReader(out)

    for i in range(5):
        ref = explainer.shap_values(X[i])
        got = reader.shap_values(X[i])
        max_diff = float(np.abs(ref - got).max())
        # float16 quantization limit is ~1e-3 at worst on any realistic
        # tree model; 1e-2 leaves plenty of headroom for small models
        # where per-tree contributions can be large.
        assert max_diff < 1e-2, f"row {i}: max diff {max_diff:.4e}"


def test_completeness_axiom_holds(tmp_path, tiny_setup):
    """sum(phi) must equal predict_raw(x) — completeness of SHAP."""
    explainer, shap_table, tree_expected, X = tiny_setup
    out = tmp_path / "tiny.cbshap"
    _write(out, explainer, shap_table, tree_expected)
    reader = CBSHAPReader(out)

    for i in range(5):
        phi = reader.shap_values(X[i])
        phi_sum = phi.sum(axis=0)  # (n_classes,)
        raw = explainer.predict_raw(X[i])
        # Completeness should hold within float16 precision.
        assert np.allclose(phi_sum, raw, atol=1e-2), f"row {i}: {phi_sum} vs {raw}"


def test_rejects_non_cbshap_file(tmp_path):
    """Reader should refuse to load a file that isn't a CBSHAP sidecar."""
    bad = tmp_path / "not-cbshap.bin"
    bad.write_bytes(b"RANDOM__" + b"\x00" * 100)
    with pytest.raises(ValueError, match="not a CBSHAP file"):
        CBSHAPReader(bad)


def test_iris_sidecar_matches_committed_fixture():
    """Sanity-check the committed iris.cbshap + reference JSON round-trip."""
    fixtures = Path(__file__).resolve().parent.parent / "test" / "fixtures"
    reader = CBSHAPReader(fixtures / "iris.cbshap")
    ref = json.loads((fixtures / "iris_cbshap_reference.json").read_text())
    assert reader.n_features == ref["n_features"]
    assert reader.n_classes == ref["n_classes"]

    for row in ref["rows"]:
        phi = reader.shap_values(np.asarray(row["features"], dtype=np.float64))
        expected = np.asarray(row["phi"], dtype=np.float64)
        assert np.allclose(phi, expected, atol=1e-6)
