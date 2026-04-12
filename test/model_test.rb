require_relative "test_helper"

class ModelTest < Minitest::Test
  include FixtureHelpers

  # ------- Iris (float-only multiclass) -------

  def test_iris_metadata
    m = iris_model
    assert_equal 4, m.float_features_count
    assert_equal 0, m.cat_features_count
    assert_equal 3, m.dimensions_count
    assert m.tree_count > 0
  end

  def test_iris_single_row_raw
    iris_ref["rows"].each_with_index do |row, i|
      actual = iris_model.predict(row["features"], prediction_type: :raw_formula_val)
      assert_floats_close row["raw"], actual, msg: "iris raw row=#{i}"
    end
  end

  def test_iris_single_row_probability
    iris_ref["rows"].each_with_index do |row, i|
      actual = iris_model.predict(row["features"], prediction_type: :probability)
      assert_floats_close row["prob"], actual, msg: "iris prob row=#{i}"
    end
  end

  def test_iris_single_row_class
    iris_ref["rows"].each_with_index do |row, i|
      actual = iris_model.predict(row["features"], prediction_type: :class)
      # For multiclass :class, CatBoost returns a single-element vector with the class label.
      assert_equal 1, actual.size, "expected 1-element class vector"
      assert_equal row["class"], actual.first.to_i, "iris class row=#{i}"
    end
  end

  def test_iris_batch_flat_raw
    rows = iris_ref["rows"].map { |r| r["features"] }
    expected = iris_ref["rows"].map { |r| r["raw"] }

    actual = iris_model.predict(rows, prediction_type: :raw_formula_val)
    assert_equal expected.size, actual.size
    expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "iris batch raw row=#{i}" }
  end

  def test_iris_batch_matches_single
    rows = iris_ref["rows"].map { |r| r["features"] }
    batch = iris_model.predict(rows, prediction_type: :probability)
    rows.each_with_index do |r, i|
      single = iris_model.predict(r, prediction_type: :probability)
      assert_floats_close single, batch[i], msg: "iris single vs batch row=#{i}"
    end
  end

  def test_iris_wrong_arity_raises
    assert_raises(ArgumentError) { iris_model.predict([1.0, 2.0, 3.0]) } # only 3 floats
    assert_raises(ArgumentError) { iris_model.predict([1.0, 2.0, 3.0, 4.0, 5.0]) } # too many
  end

  def test_iris_empty_batch_raises
    err = assert_raises(ArgumentError) { iris_model.predict([[]]) }
    assert_match(/(expected 4 float features|batch must not be empty)/, err.message)

    assert_raises(ArgumentError) { iris_model.predict([[5.1, 3.5, 1.4, 0.2]] * 0) }
  end

  def test_iris_load_from_buffer
    bytes = File.binread(fixture_path("iris.cbm"))
    m = CatBoost::Model.load_from_buffer(bytes)
    assert_equal 4, m.float_features_count
    assert_equal 3, m.dimensions_count

    row = iris_ref["rows"].first
    actual = m.predict(row["features"], prediction_type: :raw_formula_val)
    assert_floats_close row["raw"], actual, msg: "buffer-loaded iris"
    m.close
  end

  def test_iris_missing_file_raises
    err = assert_raises(CatBoost::LoadError) do
      CatBoost::Model.load(fixture_path("does_not_exist.cbm"))
    end
    assert_match(/No such file/, err.message)
  end

  def test_iris_close_then_predict_raises
    m = CatBoost::Model.load(fixture_path("iris.cbm"))
    m.close
    assert m.closed?
    assert_raises(CatBoost::Error) { m.predict([5.1, 3.5, 1.4, 0.2]) }
    m.close # idempotent
  end

  def test_iris_gc_stress_single_predict
    row = iris_ref["rows"].first
    GC.stress = true
    begin
      actual = iris_model.predict(row["features"], prediction_type: :probability)
      assert_floats_close row["prob"], actual, msg: "iris under GC.stress"
    ensure
      GC.stress = false
    end
  end

  # ------- Churn mixed (float + string categorical) -------

  def test_churn_metadata
    m = churn_model
    assert_equal 2, m.float_features_count
    assert_equal 2, m.cat_features_count
    assert_equal 1, m.dimensions_count
  end

  def test_churn_single_row_raw
    churn_ref["rows"].each_with_index do |row, i|
      actual = churn_model.predict(row["floats"], cat_features: row["cats"], prediction_type: :raw_formula_val)
      assert_floats_close row["raw"], actual, msg: "churn raw row=#{i}"
    end
  end

  def test_churn_single_row_probability
    churn_ref["rows"].each_with_index do |row, i|
      actual = churn_model.predict(row["floats"], cat_features: row["cats"], prediction_type: :probability)
      assert_floats_close row["prob"], actual, msg: "churn prob row=#{i}"
    end
  end

  def test_churn_batch_mixed
    floats = churn_ref["rows"].map { |r| r["floats"] }
    cats = churn_ref["rows"].map { |r| r["cats"] }
    expected = churn_ref["rows"].map { |r| r["raw"] }

    actual = churn_model.predict(floats, cat_features: cats, prediction_type: :raw_formula_val)
    assert_equal expected.size, actual.size
    expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "churn batch raw row=#{i}" }
  end

  def test_churn_batch_matches_single
    floats = churn_ref["rows"].map { |r| r["floats"] }
    cats = churn_ref["rows"].map { |r| r["cats"] }
    batch = churn_model.predict(floats, cat_features: cats, prediction_type: :probability)
    floats.each_with_index do |f, i|
      single = churn_model.predict(f, cat_features: cats[i], prediction_type: :probability)
      assert_floats_close single, batch[i], msg: "churn single vs batch row=#{i}"
    end
  end

  def test_churn_missing_cats_for_batch_raises
    floats = churn_ref["rows"].map { |r| r["floats"] }
    assert_raises(ArgumentError) { churn_model.predict(floats, prediction_type: :raw_formula_val) }
  end

  def test_churn_wrong_cat_arity_raises
    assert_raises(ArgumentError) do
      churn_model.predict([10.0, 50.0], cat_features: ["free"], prediction_type: :raw_formula_val)
    end
  end

  def test_churn_gc_stress_batch_with_cats
    floats = churn_ref["rows"].map { |r| r["floats"] }
    cats = churn_ref["rows"].map { |r| r["cats"] }
    expected = churn_ref["rows"].map { |r| r["raw"] }

    GC.stress = true
    begin
      actual = churn_model.predict(floats, cat_features: cats, prediction_type: :raw_formula_val)
      expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "churn batch under GC.stress row=#{i}" }
    ensure
      GC.stress = false
    end
  end
end
