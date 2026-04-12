require_relative "test_helper"

# Public-API surface tests — the things that aren't about prediction correctness
# but about how the gem FEELS to use. Exercises Python-compatibility conveniences,
# Ruby debugging hygiene, and the full-schema accessors.
class ApiSurfaceTest < Minitest::Test
  include FixtureHelpers

  # ----- Python-compatible string prediction types -----------------------

  def test_accepts_python_string_prediction_type_probability
    row = iris_ref["rows"].first
    sym_result = iris_model.predict(row["features"], prediction_type: :probability)
    str_result = iris_model.predict(row["features"], prediction_type: "Probability")
    assert_floats_close sym_result, str_result, msg: "symbol and string should give identical results"
    assert_floats_close row["prob"], str_result, msg: "string prediction_type should match reference"
  end

  def test_accepts_python_string_prediction_type_raw_formula_val
    row = iris_ref["rows"].first
    str_result = iris_model.predict(row["features"], prediction_type: "RawFormulaVal")
    assert_floats_close row["raw"], str_result, msg: "'RawFormulaVal' string"
  end

  def test_accepts_python_string_prediction_type_class
    row = iris_ref["rows"].first
    str_result = iris_model.predict(row["features"], prediction_type: "Class")
    assert_equal 1, str_result.size
    assert_equal row["class"], str_result.first.to_i
  end

  def test_unknown_string_prediction_type_raises_with_helpful_message
    err = assert_raises(ArgumentError) do
      iris_model.predict([5.1, 3.5, 1.4, 0.2], prediction_type: "NotAThing")
    end
    assert_match(/NotAThing/, err.message)
    assert_match(/Probability/, err.message) # shows valid options
  end

  def test_unknown_symbol_prediction_type_raises_with_helpful_message
    err = assert_raises(ArgumentError) do
      iris_model.predict([5.1, 3.5, 1.4, 0.2], prediction_type: :bogus)
    end
    assert_match(/bogus/, err.message)
    assert_match(/probability/, err.message)
  end

  def test_wrong_type_prediction_type_raises
    assert_raises(ArgumentError) do
      iris_model.predict([5.1, 3.5, 1.4, 0.2], prediction_type: 42)
    end
  end

  # ----- predict_proba convenience ---------------------------------------

  def test_predict_proba_matches_predict_with_probability
    iris_ref["rows"].each_with_index do |row, i|
      direct = iris_model.predict(row["features"], prediction_type: :probability)
      proba  = iris_model.predict_proba(row["features"])
      assert_floats_close direct, proba, msg: "predict_proba row=#{i}"
      assert_floats_close row["prob"], proba, msg: "predict_proba vs reference row=#{i}"
    end
  end

  def test_predict_proba_works_on_batch
    rows = iris_ref["rows"].map { |r| r["features"] }
    expected = iris_ref["rows"].map { |r| r["prob"] }
    actual = iris_model.predict_proba(rows)
    assert_equal expected.size, actual.size
    expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "batch predict_proba row=#{i}" }
  end

  def test_predict_proba_works_with_named_features
    row = churn_ref["rows"].first
    actual = churn_model.predict_proba(
      tenure_months: row["floats"][0],
      monthly_spend: row["floats"][1],
      tier:          row["cats"][0],
      region:        row["cats"][1]
    )
    assert_floats_close row["prob"], actual, msg: "predict_proba with kwargs"
  end

  def test_predict_proba_works_positional_with_cats
    row = churn_ref["rows"].first
    actual = churn_model.predict_proba(row["floats"], cat_features: row["cats"])
    assert_floats_close row["prob"], actual, msg: "predict_proba positional with cats"
  end

  def test_predict_proba_rejects_prediction_type_override
    err = assert_raises(ArgumentError) do
      iris_model.predict_proba([5.1, 3.5, 1.4, 0.2], prediction_type: :class)
    end
    assert_match(/predict_proba.*locks prediction_type/, err.message)
    assert_match(/predict.*:class/, err.message)
  end

  # ----- predict_multi_proba -----------------------------------------------

  def test_predict_multi_proba_returns_independent_sigmoids
    # Use the multilabel fixture — multi_probability gives independent
    # per-label sigmoids that do NOT sum to 1.
    ref = load_reference("multilabel_reference.json")
    m = multilabel_model
    ref["rows"].each_with_index do |row, i|
      actual = m.predict_multi_proba(row["features"])
      assert_floats_close row["multi_prob"], actual, msg: "predict_multi_proba row=#{i}"
    end
  end

  def test_predict_multi_proba_matches_predict_with_multi_probability
    m = multilabel_model
    ref = load_reference("multilabel_reference.json")
    row = ref["rows"].first["features"]
    direct = m.predict(row, prediction_type: :multi_probability)
    convenience = m.predict_multi_proba(row)
    assert_floats_close direct, convenience, msg: "predict_multi_proba matches predict"
  end

  def test_predict_multi_proba_works_on_batch
    m = multilabel_model
    ref = load_reference("multilabel_reference.json")
    rows = ref["rows"].map { |r| r["features"] }
    expected = ref["rows"].map { |r| r["multi_prob"] }
    actual = m.predict_multi_proba(rows)
    expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "batch multi_proba row=#{i}" }
  end

  def test_predict_multi_proba_rejects_prediction_type_override
    m = multilabel_model
    err = assert_raises(ArgumentError) do
      m.predict_multi_proba([0.0] * 6, prediction_type: :class)
    end
    assert_match(/predict_multi_proba.*locks prediction_type/, err.message)
  end

  # ----- Model#inspect ---------------------------------------------------

  def test_inspect_shows_meaningful_metadata
    str = iris_model.inspect
    assert_match(/CatBoost::Model/, str)
    assert_match(/trees=\d+/, str)
    assert_match(/floats=4/, str)
    assert_match(/cats=0/, str)
    assert_match(/dim=3/, str)
    refute_match(/CLOSED/, str)
  end

  def test_inspect_marks_closed_models
    m = CatBoost::Model.load(fixture_path("iris.cbm"))
    m.close
    assert_match(/CLOSED/, m.inspect)
  end

  def test_to_s_matches_inspect
    assert_equal iris_model.inspect, iris_model.to_s
  end

  # ----- schema.all_names ------------------------------------------------

  def test_schema_all_names_for_named_model
    assert_equal(
      ["tenure_months", "monthly_spend", "tier", "region"],
      churn_model.schema.all_names
    )
  end

  def test_schema_all_names_for_synthetic_model
    # Iris was trained without feature_names, so all_names is the synthetic set.
    assert_equal 4, iris_model.schema.all_names.size
    iris_model.schema.all_names.each { |n| assert_match(/\A\d+\z/, n) }
  end

  def test_schema_all_names_is_frozen
    assert churn_model.schema.all_names.frozen?
  end
end
