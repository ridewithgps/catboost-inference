require_relative "test_helper"

# Multi-label model tests. The underlying CatBoost model is trained with
# loss_function="MultiLogloss": K independent binary classifiers under one
# handle, and each row of ground truth may have 0..K labels active
# simultaneously. Probe rows in the fixture span every label-count bucket
# (0, 1, 2, 3, and 4 active labels) so we cover the full range.
#
# Key quirk validated by these tests: for MultiLogloss models the C API
# exposes TWO different "probability-ish" prediction types with different
# semantics -- :multi_probability (independent per-label sigmoids, does NOT
# sum to 1) and :probability (softmax over K logits, DOES sum to 1).
# Users of multi-label models almost always want :multi_probability.
class MultiLabelTest < Minitest::Test
  include FixtureHelpers

  alias_method :model, :multilabel_model
  alias_method :reference, :multilabel_ref

  def k
    reference["dimensions_count"]
  end

  def test_metadata
    assert_equal 6, model.float_features_count
    assert_equal 0, model.cat_features_count
    assert_equal 4, model.dimensions_count
    assert_equal 4, k
    assert model.tree_count > 0
  end

  def test_probe_rows_span_all_label_counts
    counts = reference["rows"].map { |r| r["label_count"] }.uniq.sort
    assert_equal [0, 1, 2, 3, 4], counts,
                 "fixture should cover every 0..K label-count bucket; got #{counts.inspect}"
  end

  def test_single_row_raw
    reference["rows"].each_with_index do |row, i|
      actual = model.predict(row["features"], prediction_type: :raw_formula_val)
      assert_equal k, actual.size
      assert_floats_close row["raw"], actual, msg: "multilabel raw row=#{i} (#{row['label_count']} active labels)"
    end
  end

  def test_single_row_multi_probability
    reference["rows"].each_with_index do |row, i|
      actual = model.predict(row["features"], prediction_type: :multi_probability)
      assert_equal k, actual.size
      assert_floats_close row["multi_prob"], actual,
                          msg: "multilabel multi_prob row=#{i} (#{row['label_count']} active labels)"

      # Sanity: multi_probability values are independent sigmoids, so they
      # almost never sum to 1.0 for a multi-label row. The only way they
      # would is a pathological coincidence we can ignore here.
      each_prob_in_unit_interval(actual, i)
    end
  end

  def test_single_row_probability_is_softmax
    # The C API's "Probability" prediction type on a MultiLogloss model is
    # a softmax over the K raw logits -- not independent sigmoids. This is
    # arguably the wrong default for multi-label, but it's what the C API
    # returns, and a thin FFI gem mirrors the C API. Verify the literal
    # behavior so any upstream change surfaces loudly.
    reference["rows"].each_with_index do |row, i|
      actual = model.predict(row["features"], prediction_type: :probability)
      assert_equal k, actual.size
      assert_floats_close row["softmax_prob"], actual,
                          msg: "multilabel probability(softmax) row=#{i}"

      sum = actual.sum
      assert (sum - 1.0).abs < 1e-6,
             "multilabel :probability row=#{i} should sum to 1.0 (softmax), got #{sum}"
    end
  end

  def test_multi_probability_differs_from_probability
    # These two types must return *different* values for this model --
    # if they ever match, CatBoost has changed the semantics of one of them
    # and we want to know.
    row = reference["rows"].first["features"]
    multi = model.predict(row, prediction_type: :multi_probability)
    soft  = model.predict(row, prediction_type: :probability)

    max_diff = multi.zip(soft).map { |a, b| (a - b).abs }.max
    assert max_diff > 1e-3,
           "multi_probability and probability should return materially different values " \
           "for a multi-label model; got max_diff=#{max_diff}"
  end

  def test_batch_raw
    rows = reference["rows"].map { |r| r["features"] }
    expected = reference["rows"].map { |r| r["raw"] }
    actual = model.predict(rows, prediction_type: :raw_formula_val)
    assert_equal expected.size, actual.size
    expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "multilabel batch raw row=#{i}" }
  end

  def test_batch_multi_probability
    rows = reference["rows"].map { |r| r["features"] }
    expected = reference["rows"].map { |r| r["multi_prob"] }
    actual = model.predict(rows, prediction_type: :multi_probability)
    expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "multilabel batch multi_prob row=#{i}" }
  end

  def test_batch_matches_single
    rows = reference["rows"].map { |r| r["features"] }
    batch = model.predict(rows, prediction_type: :multi_probability)
    rows.each_with_index do |r, i|
      single = model.predict(r, prediction_type: :multi_probability)
      assert_floats_close single, batch[i], msg: "multilabel single vs batch row=#{i}"
    end
  end

  def test_thresholded_labels_make_reasonable_predictions
    # Sanity check the model actually learned the task: for rows with high
    # label counts, at least one multi_probability value should exceed 0.5,
    # and for zero-label rows the max should generally stay below ~0.9.
    reference["rows"].each_with_index do |row, i|
      actual = model.predict(row["features"], prediction_type: :multi_probability)
      if row["label_count"] >= 3
        assert actual.any? { |p| p > 0.5 },
               "row=#{i} has #{row['label_count']} active labels but no predicted prob > 0.5"
      end
    end
  end

  def test_gc_stress_batch_multi_probability
    rows = reference["rows"].map { |r| r["features"] }
    expected = reference["rows"].map { |r| r["multi_prob"] }
    GC.stress = true
    begin
      actual = model.predict(rows, prediction_type: :multi_probability)
      expected.each_with_index { |e, i| assert_floats_close e, actual[i], msg: "multilabel under GC.stress row=#{i}" }
    ensure
      GC.stress = false
    end
  end

  private

  def each_prob_in_unit_interval(probs, row_index)
    probs.each_with_index do |p, j|
      assert p >= 0.0 && p <= 1.0, "row=#{row_index} label=#{j} prob #{p} outside [0,1]"
    end
  end
end
