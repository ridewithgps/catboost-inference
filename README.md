# catboost-inference

Minimal Ruby FFI bindings for [CatBoost](https://catboost.ai) model inference.

**Applier only.** Train your models in Python, save them as `.cbm`, load
them from Ruby, and run predictions with a clean, Rubyish API. No compiler
at install time, no Python runtime required, no Numo dependency — just
[`ffi`](https://github.com/ffi/ffi) and the official `libcatboostmodel`
shared library (vendored per platform).

Built against **CatBoost v1.2.10** / `libcatboostmodel` v1.2.10.

## Status

v0.1 ships as three platform-specific gems:

| Platform       | Vendored binary                            |
|----------------|--------------------------------------------|
| `x86_64-linux` | `libcatboostmodel.so` (glibc 2.14+)        |
| `arm64-darwin` | `libcatboostmodel.dylib` (universal2)      |
| `x86_64-darwin`| `libcatboostmodel.dylib` (universal2)      |

The two macOS gems share a single Mach-O universal2 `.dylib` from CatBoost
upstream (one file, both architectures). Windows and linux-aarch64 are
planned follow-ups — the binding layer is portable, only the vendored
shared libraries need to change. The public Ruby API does not change
across platforms.

## Install

Add to your `Gemfile`:

```ruby
gem "catboost-inference",
    git: "https://github.com/ridewithgps/catboost-inference.git",
    tag: "v0.1.0"
```

Then `bundle install`. Bundler picks the matching platform-specific gem
automatically based on your OS and CPU — no compiler, no Python, no
CatBoost install required at install time.

**Mixed-platform teams:** if your team spans Linux and macOS (for
example, Linux CI plus Mac dev workstations), declare both platforms in
your app's `Gemfile.lock` once so the same lockfile works everywhere:

```
bundle lock --add-platform x86_64-linux arm64-darwin
```

Commit the updated `Gemfile.lock`.

## Five-minute tour

```ruby
require "catboost"

model = CatBoost::Model.load("iris.cbm")
model                         # => #<CatBoost::Model trees=50 floats=4 cats=0 dim=3>

model.predict([5.1, 3.5, 1.4, 0.2])
# => [4.179, -1.612, -2.567]                         (raw logits by default)

model.predict_proba([5.1, 3.5, 1.4, 0.2])
# => [0.9957, 0.0030, 0.0011]

model.predict([5.1, 3.5, 1.4, 0.2], prediction_type: :class)
# => [0.0]

# Batch — Array of Arrays in, Array of Arrays out
model.predict_proba([
  [5.1, 3.5, 1.4, 0.2],
  [6.2, 3.4, 5.4, 2.3]
])
# => [[0.99, 0.00, 0.00], [0.00, 0.02, 0.97]]
```

With a model trained with explicit feature names:

```ruby
churn = CatBoost::Model.load("churn.cbm")
churn.feature_names
# => { floats: ["tenure_months", "monthly_spend"], cats: ["tier", "region"] }

# Pass features as keyword arguments — no positional-order memorization
churn.predict_proba(
  tenure_months: 12.0,
  monthly_spend: 49.95,
  tier:          :premium,   # Symbol OK
  region:        "US"        # String also OK
)
# => [0.31]
```

## Developer workflow

A [`justfile`](https://github.com/casey/just) ships with the repo. Install
`just` once (`brew install just` / `apt install just` / `cargo install just`),
then:

```
just setup                           # install deps + vendor the shared library
just test                            # run all 179 tests
just test test/multilabel_test.rb    # one file
just test predict_proba              # name pattern
just fixtures                        # regenerate Python-produced test fixtures
just cbshap path/to/model.cbm        # export a CBSHAP sidecar (see Explainability)
just build / build-all / publish     # gem packaging
```

Run `just` with no arguments to list every recipe. If you don't want to install
`just`, the recipes map one-to-one onto `bundle exec rake` targets — read the
`justfile`, it's a thin wrapper.

## Public API

### `CatBoost::Model.load(path, cbshap: nil) → Model`
### `CatBoost::Model.load_from_buffer(bytes, cbshap: nil) → Model`

Loads a `.cbm` file (from path or a binary `String`) and returns a `Model`
with its handle managed by `FFI::AutoPointer` — freed automatically when the
Ruby object is garbage-collected. Raises `CatBoost::LoadError` on failure
with the C API's error string.

Passing `cbshap:` attaches a precomputed SHAP sidecar for explainability
(see [Explainability](#explainability-cbshap) below).

```ruby
model = CatBoost::Model.load("iris.cbm")
model = CatBoost::Model.load("iris.cbm", cbshap: "iris.shap.cbshap")
model = CatBoost::Model.load_from_buffer(File.binread("iris.cbm"))
```

### `Model#predict(input = nil, cat_features: nil, prediction_type: :raw_formula_val, **named_features) → Array`

The main inference method. **Five input shapes are accepted**, listed from
most to least Rubyish:

```ruby
# 1. Keyword arguments (requires a model trained with feature_names)
model.predict(tenure_months: 12.0, monthly_spend: 49.95, tier: :premium, region: "US")

# 2. Explicit Hash (same as above, useful when building rows dynamically)
model.predict({ tenure_months: 12.0, monthly_spend: 49.95, tier: :premium, region: "US" })

# 3. Positional single row (pure float model)
model.predict([5.1, 3.5, 1.4, 0.2])

# 4. Positional single row + cats via kwarg
model.predict([12.0, 49.95], cat_features: [:premium, :us])

# 5. Batch — an Array of any of the above row shapes
model.predict([row1, row2, row3], prediction_type: :probability)
```

**Categorical values may always be passed as Symbols OR Strings** — they
are converted via `#to_s` before being sent to the C API. Unknown or missing
feature names in Hash/kwarg input raise `ArgumentError` / `KeyError` so typos
cannot silently produce wrong predictions.

Return shape: `Array<Float>` for single rows, `Array<Array<Float>>` for
batches. The inner length depends on the prediction type — see below.

### `Model#predict_proba(input = nil, cat_features: nil, **named_features) → Array`

Convenience wrapper: `predict(input, prediction_type: :probability)`.
**Correct for binary and multiclass models.** For multi-label models
(`MultiLogloss`), use `predict_multi_proba` instead — see below. Accepts the
same five input shapes as `predict`. Passing `prediction_type:` as a kwarg
raises `ArgumentError`.

### `Model#predict_multi_proba(input = nil, cat_features: nil, **named_features) → Array`

Convenience wrapper: `predict(input, prediction_type: :multi_probability)`.
**Correct for multi-label (`MultiLogloss`) models** — returns K independent
per-label sigmoid probabilities that do NOT sum to 1. Using `predict_proba`
on a multi-label model silently returns softmax values (wrong for
multi-label); `predict_multi_proba` is the intended method.

### Prediction types

Pass `prediction_type:` as either a **Ruby symbol** or a **Python-style
string** — both work, so code copy-pasted from the Python docs will run:

```ruby
model.predict(row, prediction_type: :probability)       # Ruby idiomatic
model.predict(row, prediction_type: "Probability")      # Python compatible
```

| Symbol                   | Python string          | Valid for                             |
|--------------------------|------------------------|---------------------------------------|
| `:raw_formula_val`       | `"RawFormulaVal"`      | any model (default)                   |
| `:probability`           | `"Probability"`        | classification                        |
| `:multi_probability`     | `"MultiProbability"`   | **multi-label** (`MultiLogloss`)      |
| `:class`                 | `"Class"`              | classification                        |
| `:exponent`              | `"Exponent"`           | regression                            |
| `:log_probability`       | `"LogProbability"`     | multiclass                            |
| `:rmse_with_uncertainty` | `"RMSEWithUncertainty"`| regression with uncertainty           |

Requesting a type the model doesn't support raises `CatBoost::PredictError`
with the C API's error string. Unknown type names raise `ArgumentError`
with the full valid set listed.

### Metadata

```ruby
model.float_features_count   # Integer
model.cat_features_count     # Integer
model.tree_count             # Integer
model.dimensions_count       # Integer — raw model dim (before prediction type)
model.feature_names          # { floats: [...], cats: [...] }
model.schema                 # CatBoost::FeatureSchema (see below)
model.class_names            # Array<String|Integer> | nil — from class_params metadata
model.metadata(:model_guid)  # raw String value for a given CatBoost metadata key, or nil
model.metadata?(:class_params)
model.inspect                # "#<CatBoost::Model trees=50 floats=4 cats=0 dim=3>"
```

`class_names` returns `["setosa", "versicolor", ...]` for models trained with
explicit string labels, integer indices otherwise, `nil` for regression/ranking.
`metadata(key)` exposes CatBoost's built-in key-value store (see the method's
source for the list of standard keys written at training time).

### `CatBoost::FeatureSchema`

The translation layer between Ruby-idiomatic `{name => value}` hashes and
the C API's positional float-array + cat-array layout. Built once per
`Model` at load time from three C API calls
(`GetModelUsedFeaturesNames` + `GetFloatFeatureIndices` +
`GetCatFeatureIndices`).

```ruby
schema = model.schema
schema.float_names   # ["tenure_months", "monthly_spend"]  — in positional order
schema.cat_names     # ["tier", "region"]
schema.all_names     # ["tenure_months", "monthly_spend", "tier", "region"]  — global order
schema.named?        # true if the model was trained with explicit feature_names;
                     # false for synthetic "0"/"1"/... names
schema.to_h          # { floats: [...], cats: [...] }
```

If you call `predict` with a Hash on a model whose `schema.named?` is
false, the Hash must use the synthetic names (`"0"`, `"1"`, ...) as keys —
at that point you almost certainly want the positional API instead.

### `Model#close` / `#closed?`

```ruby
model.close         # Explicitly free the handle now
model.closed?       # true after close
```

`#close` is idempotent. You generally don't need to call it — the handle
is registered with `FFI::AutoPointer` and will be freed automatically when
GC collects the Ruby object. Call `#close` explicitly only if you want
deterministic cleanup before GC runs (tight memory budgets, short-lived
workers, tests).

## Explainability (CBSHAP)

Per-row SHAP explanations via a precomputed binary sidecar. Running
TreeSHAP Algorithm 2 at request time is too slow for a production path,
so the sidecar caches per-(tree, leaf) contributions once at training
time; the Ruby reader is a leaf walk + table gather.

**Generate a sidecar from a `.cbm`:**

```
just cbshap path/to/model.cbm           # writes path/to/model.shap.cbshap
just cbshap model.cbm custom/out.cbshap # explicit output path
```

Under the hood this runs the Python writer in `python/cbshap.py` via `uv
run`. Everything is numpy + catboost — no `shap` library, no sklearn. You
can also call `cbshap.export_from_cbm(path)` directly from your own
Python training pipeline.

**Load a model with a sidecar and call `#explain`:**

```ruby
model = CatBoost::Model.load("model.cbm", cbshap: "model.shap.cbshap")
model.explainer?                         # => true

model.explain(sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2)
# =>
# {
#   "setosa"     => {
#     raw:           4.179,
#     bias:         -0.012,
#     contributions: { "petal_length" => 2.14, "petal_width" => 1.58, ... }
#   },
#   "versicolor" => { ... },
#   "virginica"  => { ... }
# }
```

`contributions` is a Hash whose iteration order is `|phi|` descending
(Ruby Hash preserves insertion order), so `contribs.keys` gives features
ranked by importance and `contribs["petal_length"]` is a direct lookup.

**Limitations:**

- **Float-only models.** CBSHAP has no encoding for categorical splits;
  `Model.load(..., cbshap: ...)` raises `CatBoost::Cbshap::IncompatibleError`
  on models with cat features.
- **Models without a sidecar raise `CatBoost::Error` on `#explain`.** Use
  `model.explainer?` to branch. The `cbshap:` kwarg is always optional.
- **Float16 quantization.** SHAP values are stored as IEEE 754 binary16,
  producing ~1e-4 max drift vs. double-precision TreeSHAP on realistic
  models — invisible for explainability UX, ~500× smaller than the drift
  between standard TreeSHAP and CatBoost's internal variant.

Errors:
- `CatBoost::Cbshap::ParseError` — malformed, truncated, or wrong-magic sidecar
- `CatBoost::Cbshap::IncompatibleError` — sidecar parses fine but doesn't
  match the model (wrong feature count, wrong class count, or cat features)
- `CatBoost::LoadError` — sidecar file not found

All three rescue as `CatBoost::Error`.

## Python CatBoost → Ruby API mapping

If you're coming from Python, here's the correspondence:

| Python                                                 | Ruby                                                    | Notes |
|--------------------------------------------------------|---------------------------------------------------------|---|
| `model = CatBoostClassifier().load_model(path)`        | `model = CatBoost::Model.load(path)`                    | Class method in Ruby; no separate constructor |
| `model.load_model(blob=bytes)`                         | `CatBoost::Model.load_from_buffer(bytes)`               | Explicit method name |
| `model.predict(data, prediction_type="Probability")`   | `model.predict(data, prediction_type: :probability)`    | Or `"Probability"` — both work |
| `model.predict_proba(data)` (binary/multiclass)        | `model.predict_proba(data)`                             | Direct parity |
| `model.predict_proba(data)` (MultiLogloss)             | `model.predict_multi_proba(data)`                       | Explicit split; Ruby doesn't auto-rewrite |
| `model.feature_names_`                                 | `model.feature_names`                                   | No trailing underscore in Ruby |
| `model.tree_count_`                                    | `model.tree_count`                                      | No trailing underscore |
| `model.is_fitted()`                                    | (not needed — models are always loaded in Ruby)         | |
| `model.save_model(path)`                               | *not supported*                                         | Applier only — train + save in Python |
| `model.fit(...)` / `grid_search` / `eval_metrics`      | *not supported*                                         | Training is out of scope |
| `model.get_feature_importance(...)`                    | *not supported*                                         | Training artifact |
| `ntree_start=`, `ntree_end=`, `thread_count=`          | *not exposed*                                           | Niche; open an issue if you need them |
| `task_type="GPU"`                                      | *not exposed*                                           | CPU applier only in v0.1 |

## Binary classifier note

For binary classification (`loss_function="Logloss"`), the C API's
`Probability` returns a single value — the probability of the positive
class. Python's high-level `model.predict(prediction_type="Probability")`
expands this to `[p0, p1]` as a convenience; this gem mirrors the C API
literally and returns `[p1]`. Compute `p0` as `1.0 - p1` if you need it.

## Multi-label (`MultiLogloss`) note — use `predict_multi_proba`

For multi-label models (trained with `loss_function="MultiLogloss"`), the
probability you want is **K independent per-label sigmoids** (each label
can be 0 or 1 independently). There are two ways to get them:

```ruby
model = CatBoost::Model.load("multilabel.cbm")
model.dimensions_count  # => 4  (K labels)

# Preferred: the convenience method (locks to :multi_probability)
model.predict_multi_proba(features)
# => [0.57, 0.29, 0.21, 0.54]     # independent per-label probs, does NOT sum to 1

# Equivalent: explicit prediction type
model.predict(features, prediction_type: :multi_probability)
```

**Do NOT use `predict_proba` on a multi-label model.** `predict_proba`
locks to `prediction_type: :probability`, which for `MultiLogloss` models
applies a **softmax over the K logits** — the values sum to 1.0 as if the
labels were mutually exclusive, which is semantically wrong for
multi-label. Python's CatBoost wrapper silently rewrites `"Probability"`
to `"MultiProbability"` for MultiLogloss models to hide this C API quirk;
this gem keeps them distinct and provides `predict_multi_proba` as the
explicit correct path.

```ruby
# For a 0/1 decision per label, threshold the output yourself:
labels = model.predict_multi_proba(features).map { |p| p >= 0.5 ? 1 : 0 }
```

## Thread safety

`CatBoost::Model` is safe to share across threads **for prediction calls
that use the same prediction type**. `SetPredictionTypeString` mutates
handle-global state, so if one thread wants `:class` predictions while
another wants `:raw_formula_val` on the same Model, you need either a
per-thread Model or a mutex around the predict call. The common case
(one thread, one prediction type, or many threads all using the same
prediction type) is safe without any extra effort.

## Testing

The gem uses [minitest](https://github.com/minitest/minitest) (Ruby stdlib,
no extra dev dependency).

```
just test                                 # all 179 tests
just test test/multilabel_test.rb         # one file
just test predict_proba                   # name pattern

# without just:
bundle exec rake test
```

### How the parity tests work

The `python/` subdirectory is a `uv`-managed harness that trains three
tiny CatBoost models (iris multiclass, churn binary with cats, multilabel)
and writes reference predictions to JSON. The Ruby test suite loads the
same `.cbm` files via FFI and asserts the predictions match Python to
within `1e-6`. Python is the source of truth.

For CBSHAP, the same harness emits `iris.cbshap` plus
`iris_cbshap_reference.json` (per-row SHAP values generated by the Python
reader). `test/cbshap_test.rb` loads the sidecar via the Ruby reader and
asserts byte-parity at `1e-6` against the Python reference, plus a looser
`5e-3` semantic check against the Algorithm-2 oracle.

Regenerate fixtures with `just fixtures`. The Python side seeds every RNG
(`random_seed=1337`) so regeneration is byte-deterministic.

### Overriding the vendored library

Set `CATBOOST_LIB_PATH` to point at a locally-built `libcatboostmodel`
shared library (`.so` on Linux, `.dylib` on macOS) — useful when testing
against an unreleased CatBoost build:

```
CATBOOST_LIB_PATH=/path/to/my/libcatboostmodel.dylib just test
```

## Platform compatibility

### Linux x86_64

The vendored `libcatboostmodel.so` is statically linked against libstdc++
and requires only **glibc 2.14+** (released 2011). No OpenMP, no BLAS, no
CUDA, no compiler runtime. It works on:

- Ubuntu 12.04+, Debian 8+, CentOS/RHEL 7+, Amazon Linux 2+, Fedora, Arch
- Docker: `ruby:3.x`, `ruby:3.x-slim`, `ruby:3.x-bookworm` — any
  glibc-based image

**Alpine Linux** uses musl libc (not glibc) and needs one extra step:

```dockerfile
# Option A (recommended): use a glibc-based image instead of Alpine
FROM ruby:3.3-slim

# Option B: stay on Alpine but add the glibc compatibility layer
FROM ruby:3.3-alpine
RUN apk add gcompat
```

### macOS (arm64 and x86_64)

The vendored `libcatboostmodel.dylib` is a **Mach-O universal2 binary**
containing both `arm64` and `x86_64` slices — a single file works on
Apple Silicon and Intel Macs. Requires macOS 10.15+ (Catalina). No Xcode
Command Line Tools, no Homebrew, no framework dependencies beyond what
ships with the OS.

On Apple Silicon, the gem tagged `arm64-darwin` is installed; on Intel
Macs, the `x86_64-darwin` gem. Both contain the same universal2 dylib —
Bundler's platform resolver picks the right gem automatically from your
`Gemfile.lock`.

## License

Apache-2.0. The vendored `libcatboostmodel.so` (Linux) and
`libcatboostmodel.dylib` (macOS universal2) are binary redistributions of
the official CatBoost v1.2.10 release and are covered by the CatBoost
project's
[Apache-2.0 license](https://github.com/catboost/catboost/blob/master/LICENSE).
