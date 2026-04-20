require "ffi"

module CatBoost
  class Error < StandardError; end
  class LibraryNotFound < Error; end
  class LoadError < Error; end
  class PredictError < Error; end

  def self.library_path
    @library_path ||= resolve_library_path
  end

  def self.resolve_library_path
    if (override = ENV["CATBOOST_LIB_PATH"]) && !override.empty?
      raise LibraryNotFound, "CATBOOST_LIB_PATH=#{override.inspect} but no file exists there" unless File.file?(override)
      return override
    end

    suffix = FFI::Platform::LIBSUFFIX
    local = Gem::Platform.local
    platform_dir = "#{local.cpu}-#{local.os}"
    candidate = File.expand_path("../vendor/#{platform_dir}/libcatboostmodel.#{suffix}", __dir__)
    return candidate if File.file?(candidate)

    # Thin platform-specific gems ship with exactly one vendor subdir; if the
    # arch-specific lookup above misses (unusual platform naming, hand-built
    # vendor tree) and there's exactly one candidate, fall back to it.
    candidates = Dir[File.expand_path("../vendor/*/libcatboostmodel.#{suffix}", __dir__)]
    return candidates.first if candidates.size == 1

    raise LibraryNotFound, <<~MSG
      Could not locate libcatboostmodel.#{suffix} for platform #{platform_dir} under #{File.expand_path("../vendor", __dir__)}.
      Run `just setup` (or `bundle exec rake vendor:fetch`) to download the pinned
      release artifact, or set CATBOOST_LIB_PATH to override with your own build.
    MSG
  end
end

require_relative "catboost/version"
require_relative "catboost/ffi"
require_relative "catboost/feature_schema"
require_relative "catboost/cbshap"
require_relative "catboost/model"
