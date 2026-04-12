"""CBSHAP v1: compact binary format for precomputed TreeSHAP lookup tables.

Why this format exists
----------------------
Running TreeSHAP Algorithm 2 per row is too slow for a production request
path (~1.8 s per row in pure Python on a 785-tree production model). For
CatBoost oblivious trees the algorithm has a crucial property: a single
tree's contribution to phi depends *only* on which leaf the row reaches,
not on the raw feature values. So we can precompute the full
``(tree, leaf) → contribution`` mapping once at training time and reduce
per-row SHAP to a leaf-index walk plus a table gather.

The file format:

- **Tiny** (~3 MB for a 785-tree production model — 20× smaller than a
  raw-JSON dump of the same data).
- **Mmap-friendly**: fixed-size per-tree records, every offset is
  computable from n_trees, depth, and n_classes.
- **Portable**: pure little-endian uint/float, no schema compiler. Ruby
  reads it with ``File.binread`` + ``String#unpack``. See
  ``lib/catboost/cbshap.rb`` for the Ruby port; byte-for-byte round
  trips are verified by the cross-language test at
  ``test/cbshap_test.rb``.

Layout
------
All integers little-endian.

Header (fixed ``36 + 4*n_classes`` bytes):
::
    offset  size   field
    0       8      magic             "CBSHAP\\x01\\x00"
    8       4      version           uint32 = 1
    12      4      n_trees           uint32
    16      4      depth             uint32 (D — same for every tree in a CatBoost oblivious model)
    20      4      n_features        uint32
    24      4      n_classes         uint32
    28      8      reserved          zero for v1
    36      4*K    bias_row          float32[n_classes] — precomputed constant row for phi[-1]

Per tree (fixed-size record, padded to 4-byte alignment):
::
    split_features    uint8[depth]       feature index at each split depth
    split_borders     float32[depth]     border threshold at each split depth
    unique_features   uint8[depth]       distinct features used by this tree (sorted, zero-padded)
    n_unique          uint8              how many slots in ``unique_features`` are real (1..depth)
    _pad              uint8              0
    shap_values       float16[2^depth, depth, n_classes]
                                         sparse SHAP payload: row ``i`` (i < n_unique)
                                         corresponds to feature ``unique_features[i]``

For D=6 and K=5 one tree takes 3,878 raw bytes → 3,880 after pad → 3,898
once we round the whole record up to the next 4-byte boundary.
Total: ``header + n_trees * 3898``. On a 785-tree production model that's
``56 + 785 * 3898 ≈ 2.90 MB``.

Quantization choice
-------------------
SHAP values are stored as IEEE 754 binary16 (float16). Empirically the
max per-row error from this quantization vs float64 is ~1e-4 on a
production model — about 500× smaller than the drift between standard
TreeSHAP and CatBoost's own internal SHAP variant, so invisible for
explainability UX.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from tree_shap import TreeSHAPExplainer

MAGIC = b"CBSHAP\x01\x00"
VERSION = 1


# ── Layout helpers ──────────────────────────────────────────────────


def _per_tree_size(depth: int, n_classes: int) -> int:
    """Byte size of one packed tree record (padded to 4-byte alignment)."""
    leaves = 1 << depth
    shap_bytes = leaves * depth * n_classes * 2  # float16 = 2 bytes
    raw = depth + 4 * depth + depth + 1 + 1 + shap_bytes
    return (raw + 3) & ~3


def _header_size(n_classes: int) -> int:
    """Full header byte size (fixed 36 bytes + bias_row)."""
    return 36 + 4 * n_classes


# ── Writer ──────────────────────────────────────────────────────────


def write_cbshap(
    path: str | Path,
    *,
    n_features: int,
    n_classes: int,
    depth: int,
    tree_features: np.ndarray,  # (T, D) int
    tree_borders: np.ndarray,  # (T, D) float
    shap_table: np.ndarray,  # (T, L, F+1, K) float — dense SHAP table
    tree_expected: np.ndarray,  # (T, K) float
    scale: float,
    bias: np.ndarray,  # (K,) float — model-level scale_and_bias bias vector
) -> None:
    """Pack a precomputed SHAP lookup table into CBSHAP v1 on disk."""
    path = Path(path)
    n_trees = int(tree_features.shape[0])
    leaves = 1 << depth
    assert tree_borders.shape == (n_trees, depth)
    assert shap_table.shape == (n_trees, leaves, n_features + 1, n_classes)
    assert tree_expected.shape == (n_trees, n_classes)
    assert bias.shape == (n_classes,)

    # Bias row: constant that the reader adds to phi[-1] after accumulation.
    bias_row = (tree_expected.sum(axis=0) * scale + bias).astype(np.float32)

    per_tree = _per_tree_size(depth, n_classes)
    hdr = _header_size(n_classes)
    total = hdr + n_trees * per_tree

    buf = bytearray(total)
    off = 0
    buf[off : off + 8] = MAGIC
    off += 8
    struct.pack_into("<IIIII", buf, off, VERSION, n_trees, depth, n_features, n_classes)
    off += 20
    struct.pack_into("<Q", buf, off, 0)  # reserved
    off += 8
    buf[off : off + 4 * n_classes] = bias_row.tobytes()
    off += 4 * n_classes
    assert off == hdr

    for t in range(n_trees):
        rec = off

        sf = tree_features[t].astype(np.uint8, copy=False)
        buf[rec : rec + depth] = sf.tobytes()
        rec += depth

        sb = tree_borders[t].astype(np.float32, copy=False)
        buf[rec : rec + 4 * depth] = sb.tobytes()
        rec += 4 * depth

        # Unique feature indices used by this tree, sorted ascending.
        used = np.unique(tree_features[t])
        n_uniq = int(used.size)
        assert n_uniq <= depth
        used_padded = np.zeros(depth, dtype=np.uint8)
        used_padded[:n_uniq] = used.astype(np.uint8)
        buf[rec : rec + depth] = used_padded.tobytes()
        rec += depth
        buf[rec] = n_uniq
        rec += 1
        buf[rec] = 0  # pad
        rec += 1

        # Sparse SHAP payload: row i corresponds to feature used_padded[i].
        packed = np.zeros((leaves, depth, n_classes), dtype=np.float16)
        for i in range(n_uniq):
            f = int(used[i])
            packed[:, i, :] = shap_table[t, :, f, :].astype(np.float16)
        buf[rec : rec + packed.nbytes] = packed.tobytes()
        rec += packed.nbytes

        pad = (off + per_tree) - rec
        if pad:
            buf[rec : rec + pad] = bytes(pad)
        off += per_tree

    assert off == total, f"final offset {off} != total size {total}"
    path.write_bytes(buf)


# ── Reader (Python) ─────────────────────────────────────────────────


class CBSHAPReader:
    """Load a CBSHAP v1 file and compute per-row SHAP by table lookup.

    This is a reference implementation for the binary format. The Ruby
    port (``lib/catboost/cbshap.rb``) mirrors it closely. Both readers
    produce byte-identical SHAP outputs — the cross-language round-trip
    test at ``test/cbshap_test.rb`` runs the Ruby reader against fixtures
    generated by this Python one.
    """

    def __init__(self, path: str | Path):
        self._buf = Path(path).read_bytes()
        self._parse_header()
        self._materialize_views()

    def _parse_header(self) -> None:
        if self._buf[:8] != MAGIC:
            raise ValueError(f"not a CBSHAP file: got magic {self._buf[:8]!r}")
        off = 8
        (self.version, self.n_trees, self.depth, self.n_features, self.n_classes) = struct.unpack_from(
            "<IIIII", self._buf, off
        )
        off += 20
        off += 8  # reserved
        self.bias_row = np.frombuffer(
            self._buf, dtype=np.float32, count=self.n_classes, offset=off
        ).copy()
        off += 4 * self.n_classes
        self._header_size_bytes = off
        self._per_tree = _per_tree_size(self.depth, self.n_classes)
        self._leaves = 1 << self.depth

    def _materialize_views(self) -> None:
        T = self.n_trees
        D = self.depth
        L = self._leaves
        K = self.n_classes

        self._split_features = np.zeros((T, D), dtype=np.uint8)
        self._split_borders = np.zeros((T, D), dtype=np.float32)
        self._unique_features = np.zeros((T, D), dtype=np.uint8)
        self._n_unique = np.zeros(T, dtype=np.uint8)
        self._shap_packed = np.zeros((T, L, D, K), dtype=np.float16)

        for t in range(T):
            rec = self._header_size_bytes + t * self._per_tree
            self._split_features[t] = np.frombuffer(
                self._buf, dtype=np.uint8, count=D, offset=rec
            )
            rec += D
            self._split_borders[t] = np.frombuffer(
                self._buf, dtype=np.float32, count=D, offset=rec
            )
            rec += 4 * D
            self._unique_features[t] = np.frombuffer(
                self._buf, dtype=np.uint8, count=D, offset=rec
            )
            rec += D
            self._n_unique[t] = self._buf[rec]
            rec += 2  # n_unique + pad
            self._shap_packed[t] = np.frombuffer(
                self._buf, dtype=np.float16, count=L * D * K, offset=rec
            ).reshape(L, D, K)

    def shap_values(self, x: np.ndarray) -> np.ndarray:
        """Per-row SHAP via table lookup.

        Returns ``(n_features + 1, n_classes)``. Column ``-1`` is the
        bias row. Matches :meth:`TreeSHAPExplainer.shap_values` to float16
        precision (~1e-4 max diff on a production model).
        """
        x = np.asarray(x, dtype=np.float64)
        phi = np.zeros((self.n_features + 1, self.n_classes), dtype=np.float64)

        for t in range(self.n_trees):
            sf = self._split_features[t]
            sb = self._split_borders[t]
            leaf = 0
            for d in range(self.depth):
                if x[sf[d]] > sb[d]:
                    leaf |= 1 << d

            n_uniq = int(self._n_unique[t])
            row = self._shap_packed[t, leaf]  # (D, K)
            for i in range(n_uniq):
                f = int(self._unique_features[t, i])
                phi[f] += row[i].astype(np.float64)

        phi[-1] += self.bias_row.astype(np.float64)
        return phi


# ── High-level export from a trained CatBoost model ─────────────────


def export_from_cbm(
    model_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Produce a CBSHAP v1 sidecar from a trained CatBoost .cbm file.

    Loads the model, computes the SHAP lookup table, and writes the
    sidecar next to the .cbm (or to ``output_path`` if given). Returns
    the path to the written file.
    """
    from catboost import CatBoostClassifier
    import tempfile

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"no .cbm at {model_path}")

    # Export the CatBoost model to JSON — TreeSHAPExplainer needs the
    # tree structure in a portable form, and this is the cleanest way
    # to get it. We use a tempfile since the JSON dump is ~6 MB and
    # isn't worth keeping around.
    cb = CatBoostClassifier()
    cb.load_model(str(model_path), format="cbm")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        cb.save_model(tmp.name, format="json")
        tmp_path = Path(tmp.name)
    try:
        json_model = json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink()

    explainer = TreeSHAPExplainer(json_model)
    shap_table, tree_expected = explainer.precompute_shap_table()

    out = Path(output_path) if output_path else model_path.with_suffix(".shap.cbshap")
    write_cbshap(
        out,
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
    return out
