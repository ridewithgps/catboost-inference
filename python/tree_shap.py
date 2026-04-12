"""Path-dependent TreeSHAP for CatBoost oblivious trees.

Implements Algorithm 2 from Lundberg et al. 2020 (Nature Machine Intelligence,
"From local explanations to global understanding with explainable AI for
trees"), adapted to CatBoost's oblivious-tree structure.

This is a faithful port of the pure-Python reference in
``shap/explainers/pytree.py`` (from the shap library), which is itself a
port of the canonical C implementation in ``shap/cext/tree_shap.h``.

Two CatBoost-specific adaptations vs vanilla TreeSHAP:

1. **Oblivious tree walking**: general-tree left/right child pointers are
   replaced by bit arithmetic. The child at depth d+1 with direction bit
   ``h`` is ``node_idx | (h << d)``. This is what makes the recursion
   deterministic for symmetric trees.

2. **Leaf centering by ``tree_expected``**: CatBoost's internal SHAP
   subtracts each tree's expected value from the leaf value at the
   accumulation step, so per-feature contributions carry the
   "Shapley-attributable" mass only. That matches what CatBoost's own
   ``get_feature_importance(type='ShapValues')`` produces inside each
   tree, and keeps additivity clean: ``sum(phi) == raw_prediction``.

Two entry points:

- :meth:`TreeSHAPExplainer.shap_values` — slow, correct, on-the-fly.
  Used as a reference/oracle for tests and as the fallback if you don't
  want to ship a precomputed sidecar.
- :meth:`TreeSHAPExplainer.precompute_shap_table` — builds a per-leaf
  contribution table in O(n_trees · 2^depth · tree_shap) total work. This
  is what ``cbshap.write_cbshap`` calls once at training time.

The class is constructed from a CatBoost model's JSON export
(``cb_model.save_model(path, format='json')``). That's where all the tree
structure (splits, borders, leaf values, leaf weights) lives in a
portable form.
"""

from __future__ import annotations

import numpy as np

# ── Path-state primitives (Lundberg 2018 Algorithm 2) ─────────────────
#
# The "unique path" is four parallel 1-D arrays:
#   feature_indexes[i]  — which feature occupies slot i on the path
#   zero_fractions[i]   — cumulative "cold-branch" cover fraction
#   one_fractions[i]    — cumulative "hot-branch" cover fraction
#   pweights[i]         — the Shapley-kernel path weight built up by EXTEND
#
# We allocate a single scratch buffer per tree and slice into
# ``parent_array[unique_depth + 1:]`` per recursion level so sibling
# subtrees don't clobber each other's state. See pytree.py for the
# original idea; this is a direct port.


def _extend_path(
    feature_indexes: np.ndarray,
    zero_fractions: np.ndarray,
    one_fractions: np.ndarray,
    pweights: np.ndarray,
    unique_depth: int,
    zero_fraction: float,
    one_fraction: float,
    feature_index: int,
) -> None:
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    pweights[unique_depth] = 1.0 if unique_depth == 0 else 0.0

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1.0) / (unique_depth + 1.0)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1.0)


def _unwind_path(
    feature_indexes: np.ndarray,
    zero_fractions: np.ndarray,
    one_fractions: np.ndarray,
    pweights: np.ndarray,
    unique_depth: int,
    path_index: int,
) -> None:
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1.0) / ((i + 1.0) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1.0)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]


def _unwound_path_sum(
    feature_indexes: np.ndarray,
    zero_fractions: np.ndarray,
    one_fractions: np.ndarray,
    pweights: np.ndarray,
    unique_depth: int,
    path_index: int,
) -> float:
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0.0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = next_one_portion * (unique_depth + 1.0) / ((i + 1.0) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * (unique_depth - i) / (unique_depth + 1.0)
        else:
            total += (pweights[i] / zero_fraction) / ((unique_depth - i) / (unique_depth + 1.0))

    return total


def _aggregate_subtree_weights(leaf_weights: np.ndarray, depth: int) -> list[np.ndarray]:
    """Roll up leaf weights into a list of per-depth node weight arrays.

    Returns a list of length ``depth+1`` where entry d has shape ``(2^d,)``.
    Entry ``d=depth`` is the leaf weights themselves; entry ``d=0`` is
    the root total. Node index at depth d is a d-bit integer where bit k
    (for 0 ≤ k < d) is the direction taken at split k (LSB = first split).
    Children of node p at depth d live at p (bit d = 0) and ``p | (1 << d)``
    (bit d = 1).
    """
    out: list[np.ndarray] = [None] * (depth + 1)  # type: ignore[list-item]
    out[depth] = leaf_weights.astype(np.float64)
    for d in range(depth - 1, -1, -1):
        child = out[d + 1]
        parent = np.zeros(1 << d, dtype=np.float64)
        for p in range(1 << d):
            parent[p] = child[p] + child[p | (1 << d)]
        out[d] = parent
    return out


# ── Main explainer class ──────────────────────────────────────────────


class TreeSHAPExplainer:
    """Path-dependent TreeSHAP over a CatBoost oblivious-tree model."""

    def __init__(self, json_model: dict):
        self.feature_names: list[str] = [
            ff["feature_id"] for ff in json_model["features_info"]["float_features"]
        ]
        self.n_features: int = len(self.feature_names)

        trees = json_model["oblivious_trees"]
        self.n_trees: int = len(trees)
        self.depth: int = len(trees[0]["splits"])
        self.n_leaves: int = 1 << self.depth

        leaf_values_len = len(trees[0]["leaf_values"])
        assert leaf_values_len % self.n_leaves == 0, (
            f"leaf_values length {leaf_values_len} is not a multiple of n_leaves={self.n_leaves}"
        )
        self.n_classes: int = leaf_values_len // self.n_leaves

        T, D, L, K = self.n_trees, self.depth, self.n_leaves, self.n_classes
        self.tree_features = np.zeros((T, D), dtype=np.int32)
        self.tree_borders = np.zeros((T, D), dtype=np.float64)
        self.tree_leaves = np.zeros((T, L, K), dtype=np.float64)
        self.tree_subtree_weights: list[list[np.ndarray]] = []

        for t, tree in enumerate(trees):
            for d, split in enumerate(tree["splits"]):
                self.tree_features[t, d] = split["float_feature_index"]
                self.tree_borders[t, d] = split["border"]
            # leaf_values in the JSON export are flattened leaf-major:
            # [leaf0_c0..c(K-1), leaf1_c0..c(K-1), ...]
            self.tree_leaves[t] = np.asarray(tree["leaf_values"]).reshape(L, K)
            leaf_w = np.asarray(tree["leaf_weights"], dtype=np.float64)
            # Replace zero leaf weights with an epsilon. Zero-weight leaves
            # (branches a split never saw in training) produce NaN in the
            # cold-child _unwound_path_sum because it divides by the cold
            # zero_fraction. The shap library applies the same workaround —
            # see shap/explainers/_tree.py::SingleTree (~line 2126).
            leaf_w = np.where(leaf_w == 0.0, 1e-6, leaf_w)
            self.tree_subtree_weights.append(_aggregate_subtree_weights(leaf_w, D))

        scale, bias = json_model["scale_and_bias"]
        self.scale = float(scale)
        self.bias = np.asarray(bias, dtype=np.float64)

    # ── Prediction (for sanity checks; same convention as CatBoost) ──

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """Raw logits matching ``prediction_type='RawFormulaVal'``."""
        x = np.asarray(x, dtype=np.float64)
        total = np.zeros(self.n_classes, dtype=np.float64)
        for t in range(self.n_trees):
            leaf_idx = self._leaf_index_for_row(t, x)
            total += self.tree_leaves[t, leaf_idx]
        return self.scale * total + self.bias

    def _leaf_index_for_row(self, t: int, x: np.ndarray) -> int:
        """Leaf index for tree ``t`` and row ``x``.

        LSB = first split (depth 0). This matches the order used by the
        JSON export's ``leaf_values`` layout, and is the convention both
        the precompute step and the reader use.
        """
        idx = 0
        for d in range(self.depth):
            f = int(self.tree_features[t, d])
            if x[f] > self.tree_borders[t, d]:
                idx |= 1 << d
        return idx

    # ── SHAP (Algorithm 2) on-the-fly ─────────────────────────────

    def shap_values(self, x: np.ndarray) -> np.ndarray:
        """Per-feature SHAP contributions for a single row.

        Returns shape ``(n_features + 1, n_classes)``. Column ``-1`` is
        the "expected value" row: ``sum(tree_expected) * scale + bias``.
        Completeness: ``sum(phi) == predict_raw(x)``.
        """
        x = np.asarray(x, dtype=np.float64)
        phi = np.zeros((self.n_features + 1, self.n_classes), dtype=np.float64)

        tree_expected = self._tree_expected_values()

        buf_len = (self.depth + 1) * (self.depth + 2)
        for t in range(self.n_trees):
            feat_idx = np.full(buf_len, -1, dtype=np.int64)
            z = np.zeros(buf_len, dtype=np.float64)
            o = np.zeros(buf_len, dtype=np.float64)
            w = np.zeros(buf_len, dtype=np.float64)
            self._tree_shap(t, x, phi, feat_idx, z, o, w, tree_expected[t])

        phi[-1] += tree_expected.sum(axis=0) * self.scale
        phi[-1] += self.bias
        return phi

    # ── Precomputation (offline, for the CBSHAP sidecar) ──────────

    def precompute_shap_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a per-(tree, leaf) SHAP contribution table.

        For an oblivious tree, a single tree's contribution to phi
        depends *only* on which leaf ``x`` reaches — not on the raw
        feature values. So we can enumerate every possible leaf in every
        tree once and cache the result. At inference time, looking up
        ``shap_table[t, leaf]`` and summing across trees is orders of
        magnitude faster than the Algorithm 2 recursion per row.

        Returns ``(shap_table, tree_expected)``:

          - ``shap_table``:   ``(n_trees, 2^depth, n_features + 1, n_classes)``
            float64. Only columns corresponding to features this tree
            actually splits on are non-zero (~6 out of 31 typically);
            the sparse packer in cbshap.py exploits that.
          - ``tree_expected``: ``(n_trees, n_classes)`` float64, the
            weighted average of each tree's leaves under its leaf
            weights. Stored separately because the bias row of the
            CBSHAP sidecar is precomputed from their sum.
        """
        D = self.depth
        L = self.n_leaves
        F = self.n_features
        K = self.n_classes

        tree_expected = self._tree_expected_values()
        shap_table = np.zeros((self.n_trees, L, F + 1, K), dtype=np.float64)

        buf_len = (D + 1) * (D + 2)
        for t in range(self.n_trees):
            for leaf in range(L):
                hot_bits = np.array([(leaf >> d) & 1 for d in range(D)], dtype=np.int64)

                feat_idx = np.full(buf_len, -1, dtype=np.int64)
                z = np.zeros(buf_len, dtype=np.float64)
                o = np.zeros(buf_len, dtype=np.float64)
                w = np.zeros(buf_len, dtype=np.float64)
                phi_one = np.zeros((F + 1, K), dtype=np.float64)

                self._tree_shap(
                    t,
                    np.zeros(F, dtype=np.float64),  # dummy — overridden by hot_bits
                    phi_one,
                    feat_idx,
                    z,
                    o,
                    w,
                    tree_expected[t],
                    hot_bits_override=hot_bits,
                )
                shap_table[t, leaf] = phi_one
        return shap_table, tree_expected

    def _tree_expected_values(self) -> np.ndarray:
        """(n_trees, n_classes) weighted average of each tree's leaves."""
        tree_expected = np.zeros((self.n_trees, self.n_classes), dtype=np.float64)
        for t in range(self.n_trees):
            leaf_w = self.tree_subtree_weights[t][self.depth]
            total = leaf_w.sum()
            if total > 0:
                tree_expected[t] = (self.tree_leaves[t] * leaf_w[:, None]).sum(axis=0) / total
        return tree_expected

    # ── Per-tree Algorithm 2 ───────────────────────────────────────

    def _tree_shap(
        self,
        t: int,
        x: np.ndarray,
        phi: np.ndarray,
        feature_indexes: np.ndarray,
        zero_fractions: np.ndarray,
        one_fractions: np.ndarray,
        pweights: np.ndarray,
        tree_expected: np.ndarray,
        hot_bits_override: np.ndarray | None = None,
    ) -> None:
        """Algorithm 2 TreeSHAP for one oblivious tree.

        If ``hot_bits_override`` is provided, each split uses that array
        to decide hot direction rather than comparing ``x`` to the
        border. This is the entry point for :meth:`precompute_shap_table`,
        which needs to evaluate every leaf in turn without constructing
        a synthetic ``x``.
        """
        splits_f = self.tree_features[t]
        splits_b = self.tree_borders[t]
        leaves = self.tree_leaves[t]
        covers = self.tree_subtree_weights[t]
        D = self.depth

        def recurse(
            depth: int,
            node_idx: int,
            parent_feat: np.ndarray,
            parent_z: np.ndarray,
            parent_o: np.ndarray,
            parent_pw: np.ndarray,
            unique_depth: int,
            parent_zero_fraction: float,
            parent_one_fraction: float,
            parent_feature_index: int,
            condition_fraction: float,
        ) -> None:
            if condition_fraction == 0.0:
                return

            feat = parent_feat[unique_depth + 1 :]
            feat[: unique_depth + 1] = parent_feat[: unique_depth + 1]
            zf = parent_z[unique_depth + 1 :]
            zf[: unique_depth + 1] = parent_z[: unique_depth + 1]
            of_ = parent_o[unique_depth + 1 :]
            of_[: unique_depth + 1] = parent_o[: unique_depth + 1]
            pw = parent_pw[unique_depth + 1 :]
            pw[: unique_depth + 1] = parent_pw[: unique_depth + 1]

            _extend_path(
                feat, zf, of_, pw, unique_depth,
                parent_zero_fraction, parent_one_fraction, parent_feature_index,
            )

            if depth == D:
                # Subtracting tree_expected is the CatBoost-specific
                # adjustment: per-feature contributions carry only the
                # Shapley-attributable mass; the expected value flows
                # into phi[-1] separately.
                leaf_value = leaves[node_idx] - tree_expected  # (K,)
                for i in range(1, unique_depth + 1):
                    w = _unwound_path_sum(feat, zf, of_, pw, unique_depth, i)
                    f_i = int(feat[i])
                    phi[f_i] += w * (of_[i] - zf[i]) * leaf_value * self.scale * condition_fraction
                return

            f_d = int(splits_f[depth])
            b_d = float(splits_b[depth])
            if hot_bits_override is not None:
                hot_bit = int(hot_bits_override[depth])
            else:
                hot_bit = 1 if x[f_d] > b_d else 0
            cold_bit = 1 - hot_bit

            parent_cover = covers[depth][node_idx]
            if parent_cover <= 0:
                return
            hot_child = node_idx | (hot_bit << depth)
            cold_child = node_idx | (cold_bit << depth)
            hot_zero_frac = covers[depth + 1][hot_child] / parent_cover
            cold_zero_frac = covers[depth + 1][cold_child] / parent_cover

            # Already split on this feature? Unwind it first so the
            # child's extend_path re-adds it with combined fractions.
            incoming_zero = 1.0
            incoming_one = 1.0
            path_index = 0
            while path_index <= unique_depth:
                if feat[path_index] == f_d:
                    break
                path_index += 1
            inner_ud = unique_depth
            if path_index != unique_depth + 1:
                incoming_zero = zf[path_index]
                incoming_one = of_[path_index]
                _unwind_path(feat, zf, of_, pw, unique_depth, path_index)
                inner_ud = unique_depth - 1

            recurse(
                depth + 1, hot_child, feat, zf, of_, pw, inner_ud + 1,
                hot_zero_frac * incoming_zero, incoming_one, f_d, condition_fraction,
            )
            recurse(
                depth + 1, cold_child, feat, zf, of_, pw, inner_ud + 1,
                cold_zero_frac * incoming_zero, 0.0, f_d, condition_fraction,
            )

        recurse(0, 0, feature_indexes, zero_fractions, one_fractions, pweights, 0, 1.0, 1.0, -1, 1.0)
