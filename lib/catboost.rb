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
    candidates = Dir[File.expand_path("../vendor/*/libcatboostmodel.#{suffix}", __dir__)]
    return candidates.first if candidates.any?

    raise LibraryNotFound, <<~MSG
      Could not locate libcatboostmodel.#{suffix} under #{File.expand_path("../vendor", __dir__)}.
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
