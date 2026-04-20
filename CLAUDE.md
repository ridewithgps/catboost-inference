# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`catboost-inference` is a minimal Ruby FFI gem for CatBoost model inference (applier only — no training). It wraps `libcatboostmodel` v1.2.10 via the `ffi` gem. Ships as four platform-specific gems: `x86_64-linux`, `aarch64-linux`, `arm64-darwin`, and `x86_64-darwin`. The macOS gems share the same universal2 `.dylib` (one file, both archs) but are published as separate gems so Bundler's platform matcher picks the right one cleanly.

## Commands

```bash
just setup                          # install Ruby + Python deps + vendor the .so/.dylib for current host
just test                           # run all 178 minitest tests
just test test/model_test.rb        # run one file (detected as a path)
just test predict_proba             # run tests matching a name pattern
just fixtures                       # regenerate Python-produced test fixtures (+ iris.cbshap sidecar)
just build                          # build one gem for the current host platform
just build-all                      # build all four platform gems (2× linux + 2× darwin)
just publish                        # clean + build-all + push every gem to Cassette (needs CASSETTE_API_KEY)
just console                        # IRB with `require "catboost"` loaded
```

Python-side checks (writer + TreeSHAP): `cd python && uv run pytest test_cbshap.py`.

Without `just`: `bundle exec rake test`, `bundle exec ruby -Ilib -Itest test/file.rb`.

## Architecture

**Six modules, each with one job:**

- `CatBoost` (lib/catboost.rb) — Error classes, library path resolution. At load time it resolves `vendor/#{Gem::Platform.local.cpu}-#{Gem::Platform.local.os}/libcatboostmodel.#{FFI::Platform::LIBSUFFIX}` directly. This matters on the fat gem (or in the dev repo), where multiple vendor subdirs share the same LIBSUFFIX (e.g. `aarch64-linux` and `x86_64-linux` both use `.so`) and a simple glob would pick the wrong one. Falls back to a glob if there's exactly one candidate (thin-gem safety net); raises `LibraryNotFound` otherwise.
- `CatBoost::LibFFI` (lib/catboost/ffi.rb) — Raw `attach_function` declarations binding 20 C API functions; `check!` helper reads thread-local `GetErrorString()` immediately on failure
- `CatBoost::LibC` (lib/catboost/ffi.rb) — Separate FFI module binding libc `free()` for cleanup of C API malloc'd output arrays
- `CatBoost::FeatureSchema` (lib/catboost/feature_schema.rb) — Translates `{name => value}` Hashes into the C API's positional `[float_array, cat_array]` layout; built once per Model at load time
- `CatBoost::Cbshap` (lib/catboost/cbshap.rb) — Pure-Ruby reader for CBSHAP v1 sidecars + `Explainer` wrapper that adds feature/class names. Float-only, stdlib-only, no FFI. The binary format is defined by python/cbshap.py and the Ruby reader is a byte-identical port.
- `CatBoost::Model` (lib/catboost/model.rb) — Public API: load (accepts optional `cbshap:` sidecar), predict, predict_proba, predict_multi_proba, explain, explainer?, feature_names, metadata, metadata?, class_names, close

**Handle lifecycle:** `ManagedHandle < FFI::AutoPointer` wraps the opaque C handle. GC calls `ModelCalcerDelete` automatically; `#close` does it eagerly. No `ObjectSpace.define_finalizer` — AutoPointer is the idiomatic FFI pattern.

**Predict dispatch:** `predict` → `resolve_input` (normalize nil/kwargs/Hash) → `set_prediction_type` (returns per-row dim) → `dispatch_predict` → `dispatch_array` → one of `predict_single`, `predict_batch_flat` (float-only), `predict_batch_mixed` (float+cat). Each maps to a different C API entry point.

**Prediction dim is method-local, never cached on @.** `SetPredictionTypeString` mutates handle-global state; caching the dim on an instance variable would race if threads use different prediction types on the same Model.

## Key design rules

- **prediction_type accepts both Ruby symbols and Python CamelCase strings.** `:probability` and `"Probability"` are equivalent. `resolve_prediction_type_string` normalizes. Unknown types raise `ArgumentError` listing all valid options.
- **predict_proba explicitly rejects prediction_type kwarg.** Before this guard, `**named_features` splat silently overrode the hardcoded `:probability`. Now raises with guidance to use `predict(...)` instead.
- **Cat feature string pointers must be held alive across FFI calls.** `build_cat_ptr` and `build_cat_matrix` return `[ptr, keep]` tuples; the caller holds `_keep` in a local variable until after the C call returns. Dropping it before Calc is a use-after-free.
- **Feature-name C API memory is freed with `begin/ensure`.** `GetModelUsedFeaturesNames` and index getters malloc their output; `FeatureSchema.read_feature_names` and `read_indices` free with `LibC.free` in ensure blocks.
- **Binary classifiers return 1-element probability** (C API literal), not `[p0, p1]` like Python's wrapper. Multiclass returns K softmax. Multi-label: use `predict_multi_proba` (locks `:multi_probability`) for K independent sigmoids — `predict_proba` applies softmax which is wrong for multi-label. Python's wrapper silently rewrites `"Probability"` to `"MultiProbability"` for MultiLogloss; this gem keeps them distinct with two explicit convenience methods.
- **CBSHAP is float-only.** The binary format has no encoding for cat splits; `Model.load(path, cbshap: ...)` raises `CatBoost::Cbshap::ParseError` if the model has categorical features. The writer (python/cbshap.py) also refuses to produce sidecars for mixed models — the failure is surfaced on both ends.
- **CBSHAP class_names fall back to integers when model metadata is missing or mismatched.** For binary models (2 labels, 1 logit dimension) and regression models, the Explainer keys the result hash by integer indices rather than guessing which label to attach to the single SHAP column. Multiclass uses the trained `class_names` directly.
- **Reader lifetime is tied to Model lifetime.** `Model#close` drops the explainer along with the FFI handle — no separate explainer close needed. The Reader holds parsed Ruby arrays, so it survives `close` in theory, but we release it eagerly for consistency.

## Testing approach

Minitest with Python-generated fixtures. `python/generate_fixtures.py` trains three CatBoost models (iris multiclass, churn binary with cats, multilabel) with fixed `random_seed=1337` and writes `.cbm` + reference JSON. Ruby tests load the same `.cbm`, predict, and assert parity within `1e-6`. Python is the source of truth. Fixture regen is byte-deterministic.

Shared fixture loaders (`iris_model`, `churn_model`, `iris_with_cbshap`, etc.) live in `test/test_helper.rb` via `FixtureHelpers` module.

**Cross-language CBSHAP parity:** `python/generate_fixtures.py` also emits `iris.cbshap` (the sidecar) and `iris_cbshap_reference.json` — the latter contains per-row SHAP values generated by the *Python* `CBSHAPReader`. `test/cbshap_test.rb` loads the same sidecar via the Ruby reader and asserts element-wise equality to `1e-6`, which proves the two readers decode the binary format byte-identically (float16 → float, tree layout, sparse unique-feature dispatch). A separate `oracle_rows` block in the reference JSON holds the Algorithm-2 oracle outputs at a looser `5e-3` tolerance so the Ruby side also checks that SHAP is *semantically* correct, not just self-consistent.

There are also Python-native tests in `python/test_cbshap.py` (`uv run pytest`) that round-trip the writer against `TreeSHAPExplainer` on a tiny synthesized model. Those are the fast canary for any change to the writer or tree_shap code.

## Vendoring

`libcatboostmodel.{so,dylib}` is downloaded from CatBoost GitHub releases by `rake vendor:fetch`, SHA-256 pinned in `Rakefile`'s `VENDOR_SOURCES` map. The default `vendor:fetch` resolves the current host platform via `Gem::Platform.local` (producing e.g. `x86_64-linux` / `arm64-darwin`); set `CATBOOST_GEM_PLATFORM=<platform>` to fetch a specific one manually. Both macOS entries point at the *same* universal2 dylib — one download, mirrored into `vendor/arm64-darwin/` and `vendor/x86_64-darwin/`. `rake vendor:fetch_all` grabs everything at once. The vendored binaries are committed to git, so fresh clones skip the fetch entirely.

## Gem build strategy (dual: fat + thin)

The gemspec is **platform-agnostic** — `spec.platform` defaults to `Gem::Platform::RUBY` and `spec.files` includes every committed vendor subdirectory. `gem build catboost-inference.gemspec` (or `just build`) produces a single fat `catboost-inference-VERSION.gem` (~28 MB) containing all four platforms' binaries. This is what `git:` source consumers get — Bundler can resolve one RUBY-platform gem against any `Gemfile.lock` `PLATFORMS` list.

`rake build:all` (via `just build-all`) produces four thin platform-tagged gems in-process using the idiomatic `Gem::Specification.load.dup` + `Gem::Package.build` pattern (see `rake-compiler`'s `Rake::ExtensionTask#define_native_tasks` and `libusb`/`google-protobuf`/`wasmtime-rb` Rakefiles). Each task loads the gemspec, duplicates it, retags the platform, and prunes `spec.files` down to one platform's binary. These thin gems are what `just publish` pushes to Cassette — consumers downloading from the registry get a ~3–6 MB gem with only their own binary.

Both paths use the same runtime `Dir.glob` in `lib/catboost.rb` to locate the shared library, so fat and thin gems are behaviorally identical after install. No `Bundler.with_unbundled_env` or subshell gymnastics in the Rakefile — `Gem::Package.build` runs in-process.

To upgrade: bump `LIBCATBOOSTMODEL_VERSION` in `lib/catboost/version.rb`, update all three SHA-256 entries in `Rakefile` (linux x86_64 .so + linux aarch64 .so + darwin universal2 .dylib), run `rake vendor:fetch_all`. The Linux `.so`s require only glibc (2.14+ for x86_64, 2.17+ for aarch64; no libstdc++, no OpenMP, no CUDA) — do not work on Alpine without `apk add gcompat`. The darwin dylib is a Mach-O universal2 containing both `x86_64` and `arm64` slices.

## Thread safety

Safe if all threads use the same `prediction_type` on a shared Model. Unsafe if threads use different types concurrently — use per-thread Models or a mutex.

`Model#explain` is safe for concurrent reads once the Model is loaded — the CBSHAP reader is immutable after parsing, the explainer holds only frozen Arrays, and no handle state is touched during shap_values / explain calls. Mixing `predict` and `explain` from multiple threads is also safe as long as the `predict` calls themselves stay on one prediction_type (the usual thread-safety rule).

## Explainability (CBSHAP) workflow

The Python writer lives at `python/cbshap.py` + `python/tree_shap.py`. To produce a sidecar for a model from outside the fixture flow:

```python
from cbshap import export_from_cbm
sidecar_path = export_from_cbm("path/to/model.cbm")  # → path/to/model.shap.cbshap
```

The writer takes a `.cbm`, exports it to JSON via the CatBoost library, instantiates `TreeSHAPExplainer`, computes the per-leaf SHAP table, and serializes it. Everything is numpy + catboost — no shap library, no sklearn, no pandas.

On the Ruby side:

```ruby
model = CatBoost::Model.load("path/to/model.cbm", cbshap: "path/to/model.shap.cbshap")
model.explain(name: value, ...)
# → { class_name => { raw:, bias:, contributions: { name => phi, ... } } }
```

`contributions` is a Hash whose iteration order is |phi| descending — Ruby Hash preserves insertion order (since 1.9), so `contribs.keys` gives features ranked by importance and `contribs["sepal_width"]` is a direct lookup. Top-K caveat: `contribs.first(K)` returns an Array of pairs, not a Hash — use `.first(K).to_h` if you need a Hash back.

The writer and reader agree on CBSHAP v1. Bumping the format is a breaking change — update both python/cbshap.py and lib/catboost/cbshap.rb, then regenerate fixtures and run both test suites.
