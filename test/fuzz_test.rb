require_relative "test_helper"
require "securerandom"
require "tempfile"

# Fuzz / robustness tests. The goal: NOTHING in this file should segfault,
# bus-error, or kill the Ruby process. Every case must either succeed or
# raise a clean Ruby exception. If any test in this file causes a crash
# (you'll see "Segmentation fault" in the terminal instead of a test
# failure), that's a BLOCKER bug in the FFI binding.
#
# Uses the real 654-tree route-classification model for realism.
class FuzzTest < Minitest::Test
  MODEL_PATH = File.expand_path("fixtures/multilabel.cbm", __dir__)

  def model
    @model ||= CatBoost::Model.load(MODEL_PATH)
  end

  def valid_row
    [0.0] * 6
  end

  # Helper: assert that a block either succeeds or raises a Ruby exception
  # (NOT a crash). The test passes either way — we're testing that the
  # process survives, not that the result is correct.
  def assert_no_crash(label, &block)
    block.call
  rescue => e
    # Clean Ruby exception — this is fine
    pass
  rescue Exception => e
    # SignalException, SystemExit, etc. — still Ruby-level, still fine
    # (but weird, so note it)
    pass
  end

  # =====================================================================
  # 1. WRONG FEATURE COUNTS
  # =====================================================================

  def test_too_few_features
    assert_raises(ArgumentError) { model.predict([1.0, 2.0, 3.0]) }
  end

  def test_too_many_features
    assert_raises(ArgumentError) { model.predict([0.0] * 20) }
  end

  def test_zero_features
    assert_raises(ArgumentError) { model.predict([]) }
  end

  def test_one_feature
    assert_raises(ArgumentError) { model.predict([1.0]) }
  end

  def test_39_features
    assert_raises(ArgumentError) { model.predict([0.0] * 5) }
  end

  def test_41_features
    assert_raises(ArgumentError) { model.predict([0.0] * 7) }
  end

  # =====================================================================
  # 2. WRONG TYPES IN FEATURE ARRAYS
  # =====================================================================

  def test_nil_in_features
    row = valid_row.dup
    row[5] = nil
    assert_no_crash("nil in features") { model.predict(row) }
  end

  def test_string_in_features
    row = valid_row.dup
    row[5] = "hello"
    assert_no_crash("string in features") { model.predict(row) }
  end

  def test_symbol_in_features
    row = valid_row.dup
    row[5] = :foo
    assert_no_crash("symbol in features") { model.predict(row) }
  end

  def test_boolean_in_features
    row = valid_row.dup
    row[5] = true
    assert_no_crash("boolean in features") { model.predict(row) }
  end

  def test_array_in_features
    row = valid_row.dup
    row[5] = [1.0, 2.0]
    assert_no_crash("nested array in features") { model.predict(row) }
  end

  def test_hash_in_features
    row = valid_row.dup
    row[5] = { a: 1 }
    assert_no_crash("hash in features") { model.predict(row) }
  end

  # =====================================================================
  # 3. EXTREME NUMERIC VALUES
  # =====================================================================

  def test_nan_in_features
    row = valid_row.dup
    row[0] = Float::NAN
    # CatBoost may handle NaN internally — should not crash
    assert_no_crash("NaN") { model.predict(row) }
  end

  def test_infinity_in_features
    row = valid_row.dup
    row[0] = Float::INFINITY
    assert_no_crash("Infinity") { model.predict(row) }
  end

  def test_negative_infinity_in_features
    row = valid_row.dup
    row[0] = -Float::INFINITY
    assert_no_crash("-Infinity") { model.predict(row) }
  end

  def test_float_max_in_features
    row = [Float::MAX] * 40
    assert_no_crash("Float::MAX") { model.predict(row) }
  end

  def test_float_min_in_features
    row = [-Float::MAX] * 40
    assert_no_crash("-Float::MAX") { model.predict(row) }
  end

  def test_very_small_floats
    row = [Float::EPSILON] * 40
    assert_no_crash("Float::EPSILON") { model.predict(row) }
  end

  def test_integers_instead_of_floats
    row = [0] * 6
    # Ruby Integers are Numeric, should convert via to_f
    result = model.predict(row)
    assert_equal 4, result.size
  end

  def test_large_integers
    row = [10**18] * 40
    assert_no_crash("huge integers") { model.predict(row) }
  end

  def test_rationals
    row = [Rational(1, 3)] * 40
    assert_no_crash("Rationals") { model.predict(row) }
  end

  # =====================================================================
  # 4. CORRUPTED MODEL DATA
  # =====================================================================

  def test_load_empty_file
    Tempfile.create("empty.cbm") do |f|
      f.close
      assert_raises(CatBoost::LoadError) { CatBoost::Model.load(f.path) }
    end
  end

  def test_load_random_bytes
    Tempfile.create("random.cbm") do |f|
      f.binmode
      f.write(SecureRandom.random_bytes(1024))
      f.close
      assert_raises(CatBoost::LoadError) { CatBoost::Model.load(f.path) }
    end
  end

  def test_load_text_file
    Tempfile.create("text.cbm") do |f|
      f.write("this is not a catboost model\n" * 100)
      f.close
      assert_raises(CatBoost::LoadError) { CatBoost::Model.load(f.path) }
    end
  end

  def test_load_truncated_model
    bytes = File.binread(MODEL_PATH)
    # Take only the first 1KB of a 4MB model
    truncated = bytes[0, 1024]
    assert_raises(CatBoost::LoadError) { CatBoost::Model.load_from_buffer(truncated) }
  end

  def test_load_corrupted_model
    bytes = File.binread(MODEL_PATH)
    # Flip some bytes in the middle
    corrupted = bytes.dup
    (1000..1100).each { |i| corrupted.setbyte(i, corrupted.getbyte(i) ^ 0xFF) }
    assert_no_crash("corrupted model bytes") { CatBoost::Model.load_from_buffer(corrupted) }
  end

  def test_load_zero_bytes_buffer
    assert_raises(CatBoost::LoadError) { CatBoost::Model.load_from_buffer("") }
  end

  def test_load_one_byte_buffer
    assert_raises(CatBoost::LoadError) { CatBoost::Model.load_from_buffer("\x00") }
  end

  def test_load_nonexistent_file
    assert_raises(CatBoost::LoadError) { CatBoost::Model.load("/tmp/does_not_exist_#{$$}.cbm") }
  end

  def test_load_directory_as_file
    assert_raises(CatBoost::LoadError) { CatBoost::Model.load("/tmp") }
  end

  # =====================================================================
  # 5. LIFECYCLE ABUSE
  # =====================================================================

  def test_predict_after_close
    m = CatBoost::Model.load(MODEL_PATH)
    m.close
    assert_raises(CatBoost::Error) { m.predict(valid_row) }
  end

  def test_double_close
    m = CatBoost::Model.load(MODEL_PATH)
    m.close
    m.close # should be idempotent, not crash
    assert m.closed?
  end

  def test_triple_close
    m = CatBoost::Model.load(MODEL_PATH)
    3.times { m.close }
    assert m.closed?
  end

  def test_metadata_after_close
    m = CatBoost::Model.load(MODEL_PATH)
    m.close
    # Metadata is cached on the Ruby side — should still work
    assert_equal 6, m.float_features_count
    assert_equal 4, m.dimensions_count
  end

  def test_feature_names_after_close
    m = CatBoost::Model.load(MODEL_PATH)
    m.close
    # Schema is cached — should still be accessible
    names = m.feature_names
    assert_equal 6, names[:floats].size
  end

  def test_inspect_after_close
    m = CatBoost::Model.load(MODEL_PATH)
    m.close
    assert_match(/CLOSED/, m.inspect)
  end

  # =====================================================================
  # 6. EMPTY AND DEGENERATE INPUTS
  # =====================================================================

  def test_predict_nil
    assert_raises(ArgumentError) { model.predict(nil) }
  end

  def test_predict_empty_array
    # Empty array → treated as single row with 0 features → wrong arity
    assert_raises(ArgumentError) { model.predict([]) }
  end

  def test_predict_empty_hash
    # Empty Hash → split_row finds no keys → KeyError on first missing feature
    assert_raises(KeyError) { model.predict({}) }
  end

  def test_predict_nested_empty_arrays
    assert_raises(ArgumentError) { model.predict([[]]) }
  end

  def test_predict_deeply_nested
    # [[[[1.0]]]] — first element is Array, so batch path, then inner is Array, so batch again...
    assert_no_crash("deeply nested") { model.predict([[[0.0] * 6]]) }
  end

  def test_predict_batch_of_one_empty
    assert_raises(ArgumentError) { model.predict([[]], prediction_type: :multi_probability) }
  end

  def test_predict_with_no_args
    assert_raises(ArgumentError) { model.predict }
  end

  def test_predict_integer_input
    assert_raises(ArgumentError) { model.predict(42) }
  end

  def test_predict_string_input
    assert_raises(ArgumentError) { model.predict("hello") }
  end

  # =====================================================================
  # 7. BATCH EDGE CASES
  # =====================================================================

  def test_batch_inconsistent_row_sizes
    rows = [[0.0] * 6, [0.0] * 5, [0.0] * 6]
    assert_raises(ArgumentError) { model.predict(rows) }
  end

  def test_batch_with_nil_row
    rows = [[0.0] * 6, nil, [0.0] * 6]
    assert_no_crash("nil row in batch") { model.predict(rows) }
  end

  def test_batch_single_row
    result = model.predict([[0.0] * 6], prediction_type: :multi_probability)
    assert_equal 1, result.size
    assert_equal 4, result[0].size
  end

  def test_batch_1000_rows
    rows = Array.new(1000) { valid_row }
    result = model.predict(rows, prediction_type: :multi_probability)
    assert_equal 1000, result.size
    result.each { |r| assert_equal 4, r.size }
  end

  def test_batch_mixed_types_in_rows
    rows = [[0.0] * 6, "not an array"]
    assert_no_crash("mixed types in batch") { model.predict(rows) }
  end

  # =====================================================================
  # 8. PREDICTION TYPE ABUSE
  # =====================================================================

  def test_all_valid_prediction_types
    valid_types = [:raw_formula_val, :probability, :multi_probability, :class, :exponent]
    valid_types.each do |pt|
      result = model.predict(valid_row, prediction_type: pt)
      assert result.is_a?(Array), "prediction_type #{pt} should return Array"
    end
  end

  def test_invalid_prediction_type_symbol
    assert_raises(ArgumentError) { model.predict(valid_row, prediction_type: :bogus) }
  end

  def test_invalid_prediction_type_string
    assert_raises(ArgumentError) { model.predict(valid_row, prediction_type: "NotReal") }
  end

  def test_prediction_type_nil
    assert_raises(ArgumentError) { model.predict(valid_row, prediction_type: nil) }
  end

  def test_prediction_type_integer
    assert_raises(ArgumentError) { model.predict(valid_row, prediction_type: 42) }
  end

  def test_log_probability_on_multilabel_model
    # LogProbability is multiclass-only; this is a MultiLogloss model
    assert_raises(CatBoost::PredictError) do
      model.predict(valid_row, prediction_type: :log_probability)
    end
  end

  def test_rmse_with_uncertainty_on_classification_model
    # CatBoost accepts this without error (returns raw-ish values) — the
    # important thing is it doesn't crash.
    assert_no_crash("rmse_with_uncertainty on classifier") do
      model.predict(valid_row, prediction_type: :rmse_with_uncertainty)
    end
  end

  # =====================================================================
  # 9. NAMED PREDICT ABUSE
  # =====================================================================

  def test_named_predict_unknown_feature
    # Multilabel fixture has synthetic names "0".."5"; "bogus" is unknown
    assert_raises(ArgumentError) do
      model.predict({"0" => 1.0, "bogus" => 2.0})
    end
  end

  def test_named_predict_missing_feature
    assert_raises(KeyError) do
      model.predict({"0" => 1.0}) # missing features "1" through "5"
    end
  end

  def test_named_predict_extra_plus_missing
    assert_raises(ArgumentError) do
      model.predict(bogus: 1.0, also_bogus: 2.0)
    end
  end

  def test_named_predict_empty_hash_arg
    assert_raises(KeyError) { model.predict({}) }
  end

  def test_named_predict_with_nil_values
    row = model.schema.float_names.each_with_object({}) { |n, h| h[n.to_sym] = nil }
    assert_no_crash("nil values in named predict") { model.predict(row, prediction_type: :multi_probability) }
  end

  def test_named_predict_with_string_values
    row = model.schema.float_names.each_with_object({}) { |n, h| h[n.to_sym] = "oops" }
    assert_no_crash("string values in named predict") { model.predict(row, prediction_type: :multi_probability) }
  end

  # =====================================================================
  # 10. CAT FEATURES ON FLOAT-ONLY MODEL
  # =====================================================================

  def test_passing_cat_features_to_float_only_model
    # This model has 0 cat features; passing cat_features: should raise or be ignored
    assert_no_crash("cats on float-only model") do
      model.predict(valid_row, cat_features: ["hello", "world"])
    end
  end

  def test_passing_empty_cat_features
    result = model.predict(valid_row, cat_features: [])
    assert_equal 4, result.size
  end

  # =====================================================================
  # 11. GC STRESS — run the most dangerous operations under GC pressure
  # =====================================================================

  def test_gc_stress_single_predict
    GC.stress = true
    begin
      5.times { model.predict(valid_row, prediction_type: :multi_probability) }
    ensure
      GC.stress = false
    end
  end

  def test_gc_stress_batch_predict
    batch = Array.new(50) { valid_row }
    GC.stress = true
    begin
      3.times { model.predict(batch, prediction_type: :multi_probability) }
    ensure
      GC.stress = false
    end
  end

  def test_gc_stress_named_predict
    row = model.schema.float_names.each_with_object({}) { |n, h| h[n.to_sym] = 0.0 }
    GC.stress = true
    begin
      5.times { model.predict(row, prediction_type: :multi_probability) }
    ensure
      GC.stress = false
    end
  end

  def test_gc_stress_load_and_close
    GC.stress = true
    begin
      3.times do
        m = CatBoost::Model.load(MODEL_PATH)
        m.predict(valid_row)
        m.close
      end
    ensure
      GC.stress = false
    end
  end

  def test_gc_stress_load_from_buffer
    bytes = File.binread(MODEL_PATH)
    GC.stress = true
    begin
      3.times do
        m = CatBoost::Model.load_from_buffer(bytes)
        m.predict(valid_row)
        m.close
      end
    ensure
      GC.stress = false
    end
  end

  # =====================================================================
  # 12. RAPID-FIRE MIXED OPERATIONS
  # =====================================================================

  def test_rapid_fire_mixed_prediction_types
    100.times do |i|
      type = [:raw_formula_val, :multi_probability, :probability, :class, :exponent][i % 5]
      result = model.predict(valid_row, prediction_type: type)
      assert result.is_a?(Array)
    end
  end

  def test_rapid_fire_load_predict_close
    10.times do
      m = CatBoost::Model.load(MODEL_PATH)
      m.predict(valid_row)
      m.predict_proba(valid_row)
      m.close
    end
  end

  def test_many_models_open_simultaneously
    models = 10.times.map { CatBoost::Model.load(MODEL_PATH) }
    models.each { |m| m.predict(valid_row) }
    models.each(&:close)
  end
end
