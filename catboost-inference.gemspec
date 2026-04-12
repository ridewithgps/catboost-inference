require_relative "lib/catboost/version"

Gem::Specification.new do |spec|
  spec.name          = "catboost-inference"
  spec.version       = CatBoost::VERSION
  spec.authors       = ["Jeremy Castagno"]
  spec.summary       = "Minimal Ruby FFI bindings for CatBoost model inference"
  spec.description   = "Load CatBoost .cbm models and run predictions from Ruby " \
                       "via the official libcatboostmodel C API. Applier only " \
                       "(no training), pure-FFI, no compiler required at install time."
  spec.license       = "Apache-2.0"
  spec.required_ruby_version = ">= 3.1"

  # Platform-agnostic gem. `gem build` produces a single "ruby" platform gem
  # containing every committed vendored binary (used by `git:` source
  # consumers and as the default local-dev build). The per-platform thin
  # gems used by `rake build:all` / `just publish` are produced in the
  # Rakefile, which loads this spec, duplicates it, and prunes spec.files
  # down to one platform's binary before calling Gem::Package.build.
  spec.files = Dir[
    "lib/**/*.rb",
    "vendor/*/libcatboostmodel.*",
    "README.md",
    "LICENSE*"
  ]
  spec.require_paths = ["lib"]

  spec.add_dependency "ffi", "~> 1.16"
end
