require "rake/testtask"
require "digest"
require "fileutils"
require "open-uri"
require "rubygems/package"

require_relative "lib/catboost/version"

LIB_VERSION = CatBoost::LIBCATBOOSTMODEL_VERSION

DARWIN_UNIVERSAL2 = {
  url: "https://github.com/catboost/catboost/releases/download/v#{LIB_VERSION}/libcatboostmodel-darwin-universal2-#{LIB_VERSION}.dylib",
  sha256: "c31067d8df69e0be3dd353a06853d64f2e74bab1097512fc2d0db76add75514e",
  filename: "libcatboostmodel.dylib"
}.freeze

VENDOR_SOURCES = {
  "x86_64-linux" => {
    url: "https://github.com/catboost/catboost/releases/download/v#{LIB_VERSION}/libcatboostmodel-linux-x86_64-#{LIB_VERSION}.so",
    sha256: "de32f8e147ee8f969599a00b7145e8561b943e916b0132b811037f8f71952a0e",
    filename: "libcatboostmodel.so"
  },
  "aarch64-linux" => {
    url: "https://github.com/catboost/catboost/releases/download/v#{LIB_VERSION}/libcatboostmodel-linux-aarch64-#{LIB_VERSION}.so",
    sha256: "d1b5d12f57ef72fd43505d9fbc1ea30d93d54ef0f305484c6bbfde8642e99a5b",
    filename: "libcatboostmodel.so"
  },
  "arm64-darwin"  => DARWIN_UNIVERSAL2,
  "x86_64-darwin" => DARWIN_UNIVERSAL2
}.freeze

def target_platform
  return ENV["CATBOOST_GEM_PLATFORM"] if ENV["CATBOOST_GEM_PLATFORM"]
  local = Gem::Platform.local
  "#{local.cpu}-#{local.os}"
end

Rake::TestTask.new(:test) do |t|
  t.libs << "lib" << "test"
  t.test_files = FileList["test/**/*_test.rb"]
  t.warning = false
end

namespace :vendor do
  desc "Download pinned libcatboostmodel release artifact for CATBOOST_GEM_PLATFORM (default: current host)"
  task :fetch do
    platform = target_platform
    spec = VENDOR_SOURCES.fetch(platform) { abort "[vendor:fetch] unknown platform #{platform.inspect}" }
    target_dir = File.expand_path("vendor/#{platform}", __dir__)
    target_file = File.join(target_dir, spec[:filename])

    if File.file?(target_file) && Digest::SHA256.file(target_file).hexdigest == spec[:sha256]
      puts "[vendor:fetch] #{target_file} already present and matches SHA-256"
      next
    end

    FileUtils.mkdir_p(target_dir)
    puts "[vendor:fetch] downloading #{spec[:url]}"
    URI.parse(spec[:url]).open do |io|
      File.binwrite(target_file, io.read)
    end

    actual = Digest::SHA256.file(target_file).hexdigest
    if actual != spec[:sha256]
      File.delete(target_file)
      abort "[vendor:fetch] SHA-256 mismatch!\n  expected #{spec[:sha256]}\n  got      #{actual}"
    end

    puts "[vendor:fetch] wrote #{target_file} (#{File.size(target_file)} bytes)"
  end

  desc "Download release artifacts for every supported platform"
  task :fetch_all do
    VENDOR_SOURCES.each_key do |platform|
      ENV["CATBOOST_GEM_PLATFORM"] = platform
      Rake::Task["vendor:fetch"].reenable
      Rake::Task["vendor:fetch"].invoke
    end
  ensure
    ENV.delete("CATBOOST_GEM_PLATFORM")
  end
end

namespace :build do
  # One task per supported platform: loads the platform-agnostic gemspec,
  # duplicates it, retags the copy with the target platform, and prunes
  # spec.files down to just that platform's vendored binary, then builds
  # the .gem in-process via Gem::Package.build. This is the idiomatic
  # Ruby pattern (see rake-compiler's Rake::ExtensionTask#define_native_tasks
  # and libusb/google-protobuf/wasmtime-rb Rakefiles) — no subshell, no
  # Bundler env wrangling, no env var on spec.platform.
  VENDOR_SOURCES.each do |platform, source|
    desc "Build catboost-inference-#{CatBoost::VERSION}-#{platform}.gem"
    task platform do
      vendor_file = "vendor/#{platform}/#{source[:filename]}"
      unless File.file?(vendor_file)
        ENV["CATBOOST_GEM_PLATFORM"] = platform
        Rake::Task["vendor:fetch"].reenable
        Rake::Task["vendor:fetch"].invoke
        ENV.delete("CATBOOST_GEM_PLATFORM")
      end

      spec = Gem::Specification.load("catboost-inference.gemspec").dup
      spec.platform = Gem::Platform.new(platform)
      spec.files = spec.files.reject { |f| f.start_with?("vendor/") } + [vendor_file]
      Gem::Package.build(spec)
    end
  end

  desc "Build thin platform-specific .gem files for every supported platform"
  task all: VENDOR_SOURCES.keys.map { |p| "build:#{p}" }
end

desc "Regenerate Python-produced fixtures under test/fixtures/"
task :fixtures do
  sh "cd python && uv sync --quiet && uv run generate_fixtures.py"
end

desc "Full verification: vendor:fetch + fixtures + test"
task verify: ["vendor:fetch", :fixtures, :test]

task default: :test
