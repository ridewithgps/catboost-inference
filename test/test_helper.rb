$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))
require "minitest/autorun"
require "json"
require "catboost"

module FixtureHelpers
  FIXTURE_DIR = File.expand_path("fixtures", __dir__)

  def fixture_path(name)
    File.join(FIXTURE_DIR, name)
  end

  def load_reference(name)
    JSON.parse(File.read(fixture_path(name)))
  end

  def assert_floats_close(expected, actual, tol: 1e-6, msg: nil)
    assert_equal expected.size, actual.size, "#{msg}: length mismatch"
    expected.each_with_index do |e, i|
      diff = (e - actual[i]).abs
      assert diff <= tol, "#{msg}: index #{i} diff=#{diff} (expected=#{e}, actual=#{actual[i]})"
    end
  end

  # Memoized fixture loaders. Each test class uses at most a subset of these;
  # minitest creates a fresh test instance per test method, so the memoization
  # is intra-test (not shared across tests).

  def iris_model
    @iris_model ||= CatBoost::Model.load(fixture_path("iris.cbm"))
  end

  def iris_named_model
    @iris_named_model ||= CatBoost::Model.load(fixture_path("iris_named.cbm"))
  end

  def iris_ref
    @iris_ref ||= load_reference("iris_reference.json")
  end

  def churn_model
    @churn_model ||= CatBoost::Model.load(fixture_path("churn_mixed.cbm"))
  end

  def churn_ref
    @churn_ref ||= load_reference("churn_mixed_reference.json")
  end

  def multilabel_model
    @multilabel_model ||= CatBoost::Model.load(fixture_path("multilabel.cbm"))
  end

  def multilabel_ref
    @multilabel_ref ||= load_reference("multilabel_reference.json")
  end

  def iris_cbshap_path
    fixture_path("iris.cbshap")
  end

  def iris_with_cbshap
    @iris_with_cbshap ||= CatBoost::Model.load(fixture_path("iris.cbm"), cbshap: iris_cbshap_path)
  end

  def iris_cbshap_ref
    @iris_cbshap_ref ||= load_reference("iris_cbshap_reference.json")
  end
end
