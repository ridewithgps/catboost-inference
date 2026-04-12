require_relative "test_helper"

# Exercises the Hash-keyed predict API and the FeatureSchema abstraction.
#
# The churn_mixed fixture is trained with explicit feature_names, so its
# schema is "named" and users can call predict with a Hash. The iris and
# multilabel fixtures have synthetic numeric names ("0", "1", ...), so
# schema.named? is false there and the positional API is the only sensible
# option. Both cases are tested so regressions on either path surface.
class NamedFeaturesTest < Minitest::Test
  include FixtureHelpers

  # ----- Schema metadata --------------------------------------------------

  def test_churn_schema_exposes_named_float_and_cat_features
    schema = churn_model.schema
    assert_equal ["tenure_months", "monthly_spend"], schema.float_names
    assert_equal ["tier", "region"], schema.cat_names
    assert schema.named?, "churn_mixed was trained with explicit feature_names; schema should report named?=true"
  end

  def test_churn_feature_names_helper
    assert_equal(
      { floats: ["tenure_months", "monthly_spend"], cats: ["tier", "region"] },
      churn_model.feature_names
    )
  end

  def test_iris_schema_reports_synthetic_names
    schema = iris_model.schema
    assert_equal 4, schema.float_names.size
    assert_equal 0, schema.cat_names.size
    refute schema.named?, "iris was trained without feature_names; schema should be synthetic"
  end

  # ----- Hash-keyed single predict ----------------------------------------

  def test_hash_predict_matches_positional_predict_raw
    churn_ref["rows"].each_with_index do |row, i|
      hash_input = {
        tenure_months: row["floats"][0],
        monthly_spend: row["floats"][1],
        tier:          row["cats"][0],
        region:        row["cats"][1]
      }

      actual = churn_model.predict(hash_input, prediction_type: :raw_formula_val)
      assert_floats_close row["raw"], actual, msg: "hash predict raw row=#{i}"
    end
  end

  def test_hash_predict_accepts_symbol_cat_values
    row = churn_ref["rows"].first
    sym_input = {
      tenure_months: row["floats"][0],
      monthly_spend: row["floats"][1],
      tier:          row["cats"][0].to_sym, # Symbol instead of String
      region:        row["cats"][1].to_sym
    }

    actual = churn_model.predict(sym_input, prediction_type: :raw_formula_val)
    assert_floats_close row["raw"], actual, msg: "hash predict with symbol cat values"
  end

  def test_hash_predict_accepts_string_keys
    row = churn_ref["rows"].first
    str_input = {
      "tenure_months" => row["floats"][0],
      "monthly_spend" => row["floats"][1],
      "tier"          => row["cats"][0],
      "region"        => row["cats"][1]
    }

    actual = churn_model.predict(str_input, prediction_type: :raw_formula_val)
    assert_floats_close row["raw"], actual, msg: "hash predict with string keys"
  end

  def test_hash_predict_mixed_symbol_and_string_keys
    row = churn_ref["rows"].first
    mixed_input = {
      :tenure_months => row["floats"][0],
      "monthly_spend" => row["floats"][1],
      :tier => row["cats"][0],
      "region" => row["cats"][1]
    }

    actual = churn_model.predict(mixed_input, prediction_type: :raw_formula_val)
    assert_floats_close row["raw"], actual, msg: "hash predict with mixed symbol/string keys"
  end

  # ----- Hash-keyed batch predict -----------------------------------------

  def test_hash_batch_predict_matches_positional_batch
    hash_rows = churn_ref["rows"].map do |row|
      {
        tenure_months: row["floats"][0],
        monthly_spend: row["floats"][1],
        tier:          row["cats"][0],
        region:        row["cats"][1]
      }
    end
    expected = churn_ref["rows"].map { |r| r["raw"] }

    actual = churn_model.predict(hash_rows, prediction_type: :raw_formula_val)
    assert_equal expected.size, actual.size
    expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "hash batch row=#{i}" }
  end

  # ----- Error cases ------------------------------------------------------

  def test_hash_predict_unknown_key_raises
    err = assert_raises(ArgumentError) do
      churn_model.predict(
        tenure_months: 12.0,
        monthly_spend: 49.95,
        tier:          "premium",
        region:        "US",
        bogus:         42
      )
    end
    assert_match(/unknown feature/, err.message)
  end

  def test_hash_predict_missing_key_raises
    assert_raises(KeyError) do
      churn_model.predict(
        tenure_months: 12.0,
        monthly_spend: 49.95,
        tier:          "premium"
        # region missing
      )
    end
  end

  def test_hash_predict_wrong_input_type_raises
    assert_raises(ArgumentError) { churn_model.predict(42) }
    assert_raises(ArgumentError) { churn_model.predict("not a row") }
  end

  def test_batch_with_non_array_non_hash_row_raises
    err = assert_raises(ArgumentError) do
      churn_model.predict([["a"], 42, []])
    end
    # Whatever the exact error — user should not get a silent wrong answer.
    assert err.message.length > 0
  end

  # ----- Mixed Array/Hash safety ------------------------------------------

  def test_positional_predict_still_works_on_named_model
    # Having a named schema does NOT break the positional API; users who
    # prefer raw arrays keep that path.
    row = churn_ref["rows"].first
    actual = churn_model.predict(row["floats"], cat_features: row["cats"], prediction_type: :raw_formula_val)
    assert_floats_close row["raw"], actual, msg: "positional predict on named model"
  end

  def test_positional_predict_on_iris_accepts_ints_and_floats
    # Regression: Numeric (not just Float) should be accepted.
    [5, 3, 1, 0].tap do |ints|
      # We don't care about the prediction value — just that it doesn't raise.
      out = iris_model.predict(ints, prediction_type: :raw_formula_val)
      assert_equal 3, out.size
    end
  end

  # ----- GC safety under the new path -------------------------------------

  def test_hash_predict_survives_gc_stress
    row = churn_ref["rows"].first
    hash_input = {
      tenure_months: row["floats"][0],
      monthly_spend: row["floats"][1],
      tier:          row["cats"][0].to_sym,
      region:        row["cats"][1].to_sym
    }
    GC.stress = true
    begin
      actual = churn_model.predict(hash_input, prediction_type: :raw_formula_val)
      assert_floats_close row["raw"], actual, msg: "hash predict under GC.stress"
    ensure
      GC.stress = false
    end
  end

  def test_schema_read_does_not_leak_under_repeated_loads
    # Load + drop 25 times; if read_feature_names / read_indices leaked,
    # this would show up as growth or a crash under GC stress.
    GC.stress = true
    begin
      25.times do
        m = CatBoost::Model.load(fixture_path("churn_mixed.cbm"))
        assert_equal ["tenure_months", "monthly_spend"], m.schema.float_names
        m.close
      end
    ensure
      GC.stress = false
    end
  end
end
