require_relative "test_helper"

class MetadataTest < Minitest::Test
  include FixtureHelpers

  # ----- Model#metadata (generic key-value accessor) ----------------------

  def test_metadata_reads_standard_keys
    # These keys are always written by CatBoost at training time.
    # Values vary (GUID, timestamp, JSON) so we can only assert type + non-empty.
    %w[class_params params model_guid train_finish_time catboost_version_info].each do |key|
      value = iris_model.metadata(key)
      assert_kind_of String, value, "expected metadata(#{key.inspect}) to be a String"
      refute_empty value, "expected metadata(#{key.inspect}) to be non-empty"
    end
  end

  def test_metadata_returns_nil_for_missing_key
    assert_nil iris_model.metadata("definitely_not_a_real_key_12345")
  end

  def test_metadata_accepts_symbol_keys
    # Symbols are coerced to strings internally — callers using :class_params
    # get the same result as callers using "class_params".
    assert_equal iris_model.metadata("class_params"), iris_model.metadata(:class_params)
  end

  def test_metadata_accepts_empty_string_key
    # Empty key should return nil without crashing. CatBoost's
    # CheckModelMetadataHasKey handles zero-length lookups as absent.
    assert_nil iris_model.metadata("")
  end

  def test_metadata_key_with_embedded_null_byte_raises
    # FFI's :string argtype rejects embedded NUL bytes at the Ruby/FFI
    # boundary, before the C call happens. Otherwise the C-level string
    # would silently truncate at the first NUL, making
    # "class_params\x00evil" match "class_params".
    assert_raises(ArgumentError) { iris_model.metadata("class_params\x00garbage") }
    assert_raises(ArgumentError) { iris_model.metadata?("class_params\x00garbage") }
  end

  def test_metadata_multibyte_utf8_key
    # Verifies bytesize (not length) is passed — a multibyte key is
    # still routed correctly without crashing.
    assert_nil iris_model.metadata("キー")
    refute iris_model.metadata?("キー")
  end

  def test_metadata_raises_when_model_closed
    model = CatBoost::Model.load(fixture_path("iris.cbm"))
    model.close
    assert_raises(CatBoost::Error) { model.metadata("class_params") }
  end

  # ----- Model#metadata? (presence check) ---------------------------------

  def test_metadata_key_presence_check
    assert iris_model.metadata?("class_params")
    assert iris_model.metadata?(:class_params)
    refute iris_model.metadata?("definitely_not_a_real_key_12345")
  end

  def test_metadata_key_presence_raises_when_closed
    model = CatBoost::Model.load(fixture_path("iris.cbm"))
    model.close
    assert_raises(CatBoost::Error) { model.metadata?("class_params") }
  end

  # ----- Model#class_names ------------------------------------------------

  def test_class_names_returns_integers_for_default_trained_model
    # iris.cbm was trained without explicit class_names; CatBoost falls back
    # to integer class indices, which we surface as Integer, not String.
    assert_equal [0, 1, 2], iris_model.class_names
  end

  def test_class_names_returns_strings_for_named_model
    # iris_named.cbm was trained with class_names=["setosa","versicolor","virginica"]
    assert_equal %w[setosa versicolor virginica], iris_named_model.class_names
  end

  def test_class_names_is_memoized
    # Calling twice returns the same array object — proves memoization.
    # Documented as "immutable once the model is loaded" in the class_names
    # docstring; this test guards against accidental removal of the
    # defined?(@class_names) cache.
    first = iris_model.class_names
    second = iris_model.class_names
    assert_same first, second
  end

  def test_class_names_is_index_aligned_with_predict_proba
    # Sanity: the K classes returned by class_names correspond positionally
    # to the K probabilities returned by predict_proba. For the iris_named
    # setosa sample (4.9, 3.0, 1.4, 0.2), predict_proba[0] is the setosa
    # probability, and class_names[0] == "setosa".
    probs = iris_named_model.predict_proba([4.9, 3.0, 1.4, 0.2])
    assert_equal iris_named_model.class_names.size, probs.size
    top_class = iris_named_model.class_names[probs.each_with_index.max_by { |p, _| p }[1]]
    assert_equal "setosa", top_class
  end

  def test_class_names_survives_model_close_if_already_memoized
    # If class_names was read before close, the cached value is still
    # returnable after close — it's a plain Array with no handle refs.
    model = CatBoost::Model.load(fixture_path("iris_named.cbm"))
    before = model.class_names
    model.close
    assert_equal before, model.class_names
  end

  def test_class_names_raises_if_first_access_after_close
    # When @class_names is NOT already memoized, the lookup goes through
    # metadata("class_params") which calls ensure_open! and raises.
    # Complement to the "memoized survives close" test above.
    model = CatBoost::Model.load(fixture_path("iris.cbm"))
    model.close
    assert_raises(CatBoost::Error) { model.class_names }
  end

  def test_class_names_wraps_json_parse_errors_in_catboost_error
    # If CatBoost were ever to write malformed JSON into class_params
    # (it shouldn't), we catch it and wrap in CatBoost::Error so callers
    # only need to rescue one error hierarchy. Stubbed because we can't
    # easily produce a malformed .cbm file.
    model = iris_model
    model.stub :metadata, ->(key) { key == "class_params" ? "{not valid json" : nil } do
      model.remove_instance_variable(:@class_names) if model.instance_variable_defined?(:@class_names)
      err = assert_raises(CatBoost::Error) { model.class_names }
      assert_match(/malformed class_params/, err.message)
    end
  end
end
