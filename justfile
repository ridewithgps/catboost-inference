# catboost-inference — developer task runner (https://github.com/casey/just)
#
# Install `just` with: `brew install just`, `apt install just`, or `cargo install just`.

# Cassette (RideWithGPS private gem proxy) endpoint. Override with
#   CASSETTE_HOST=https://cassette.internal.example.com/rubygems just publish
export CASSETTE_HOST := env_var_or_default("CASSETTE_HOST", "https://cassette.ridewithgps.com/rubygems")

# List available commands (this is what `just` runs with no args)
default:
    @just --list --unsorted

# Install deps and download the vendored libcatboostmodel for your host (run once after cloning)
setup:
    bundle install
    cd python && uv sync --quiet
    bundle exec rake vendor:fetch

# Run tests. ARG empty → all tests; ARG is a file path → that file; else → name pattern
test ARG='':
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "{{ARG}}" ]]; then
      bundle exec rake test
    elif [[ -f "{{ARG}}" ]]; then
      bundle exec ruby -Ilib -Itest "{{ARG}}"
    else
      bundle exec rake test TESTOPTS="--name=/{{ARG}}/"
    fi

# Regenerate Python-produced test fixtures (after changing training code)
fixtures:
    cd python && uv run generate_fixtures.py

# Export a CBSHAP sidecar from a .cbm model. OUT defaults to <MODEL>.shap.cbshap next to the input.
cbshap MODEL OUT='':
    #!/usr/bin/env bash
    set -euo pipefail
    model=$(realpath "{{MODEL}}")
    out=""
    [[ -n "{{OUT}}" ]] && out=$(realpath -m "{{OUT}}")
    cd python && uv run python export_cbshap.py "$model" "$out"

# Build the .gem file for the current host platform
build:
    gem build catboost-inference.gemspec

# Build .gem files for every supported platform (x86_64-linux, arm64-darwin, x86_64-darwin)
build-all:
    bundle exec rake build:all

# Build all platform gems and push them to cassette (needs CASSETTE_API_KEY env var)
publish: clean build-all
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${CASSETTE_API_KEY:-}" ]]; then
      echo "error: CASSETTE_API_KEY is not set" >&2
      echo "       export CASSETTE_API_KEY=... (from 1Password / cassette admin) and retry" >&2
      exit 1
    fi
    shopt -s nullglob
    gems=( catboost-inference-*.gem )
    if (( ${#gems[@]} == 0 )); then
      echo "error: no .gem files found after build-all" >&2
      exit 1
    fi
    for gem_file in "${gems[@]}"; do
      echo "Pushing ${gem_file} → ${CASSETTE_HOST}"
      GEM_HOST_API_KEY="$CASSETTE_API_KEY" gem push "${gem_file}" --host "${CASSETTE_HOST}"
    done

# Open an IRB session with `require "catboost"` already loaded
console:
    bundle exec ruby -Ilib -rcatboost -e 'require "irb"; IRB.start(__FILE__)'

# Remove build artifacts
clean:
    rm -f catboost-inference-*.gem
    rm -rf pkg/ tmp/
