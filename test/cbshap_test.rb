require_relative "test_helper"
require "tempfile"

class CbshapTest < Minitest::Test
  include FixtureHelpers

  # ----- Cross-language reader parity --------------------------------------

  def test_reader_matches_python_reference_shap_values_element_wise
    reader = CatBoost::Cbshap::Reader.new(iris_cbshap_path)

    assert_equal iris_cbshap_ref["n_features"], reader.n_features
    assert_equal iris_cbshap_ref["n_classes"], reader.n_classes

    iris_cbshap_ref["rows"].each_with_index do |row, i|
      phi = reader.shap_values(row["features"])
      expected = row["phi"]
      assert_equal expected.size, phi.size, "row #{i}: phi length mismatch"
      phi.each_with_index do |phi_row, fi|
        assert_floats_close(expected[fi], phi_row, tol: 1e-6,
          msg: "row #{i} feature/bias row #{fi}")
      end
    end
  end

  def test_reader_header_fields_match_model
    reader = CatBoost::Cbshap::Reader.new(iris_cbshap_path)
    assert_equal 1, reader.version
    assert_equal iris_model.float_features_count, reader.n_features
    assert_equal iris_model.dimensions_count, reader.n_classes
    assert_operator reader.n_trees, :>, 0
    assert_operator reader.depth, :>, 0
  end

  # ----- Reader format guards ---------------------------------------------

  def test_reader_rejects_non_cbshap_file
    Tempfile.create(["not-cbshap", ".bin"]) do |f|
      f.binmode
      f.write("RANDOM__" + ("\x00" * 100))
      f.close
      err = assert_raises(CatBoost::Cbshap::ParseError) do
        CatBoost::Cbshap::Reader.new(f.path)
      end
      assert_match(/not a CBSHAP file/, err.message)
    end
  end

  def test_reader_rejects_unknown_version
    bytes = File.binread(iris_cbshap_path)
    bytes.setbyte(8, 99) # first byte of the version u32

    Tempfile.create(["bad-version", ".cbshap"]) do |f|
      f.binmode
      f.write(bytes)
      f.close
      err = assert_raises(CatBoost::Cbshap::ParseError) do
        CatBoost::Cbshap::Reader.new(f.path)
      end
      assert_match(/unsupported CBSHAP version/, err.message)
    end
  end

  def test_reader_parse_error_rescues_as_catboost_error
    Tempfile.create(["not-cbshap", ".bin"]) do |f|
      f.binmode
      f.write("NOPE" * 16)
      f.close
      assert_raises(CatBoost::Error) { CatBoost::Cbshap::Reader.new(f.path) }
    end
  end

  # ----- Model#explain end-to-end -----------------------------------------

  def test_explain_returns_hash_keyed_by_class_name
    model = iris_with_cbshap
    result = model.explain([5.1, 3.5, 1.4, 0.2])

    assert_equal model.class_names, result.keys
    result.each_value do |explanation|
      assert_kind_of Numeric, explanation[:raw]
      assert_kind_of Numeric, explanation[:bias]
      contribs = explanation[:contributions]
      assert_kind_of Hash, contribs
      assert_equal model.float_features_count, contribs.size
      contribs.each do |name, value|
        assert_kind_of String, name
        assert_kind_of Numeric, value
      end
    end
  end

  def test_contributions_support_lookup_by_name_and_cover_every_feature
    result = iris_with_cbshap.explain([5.1, 3.5, 1.4, 0.2])
    result.each_value do |explanation|
      contribs = explanation[:contributions]
      iris_with_cbshap.schema.float_names.each do |name|
        assert contribs.key?(name), "expected #{name.inspect} in contributions"
        assert_kind_of Numeric, contribs[name]
      end
    end
  end

  def test_explain_contributions_hash_iteration_order_matches_absolute_magnitude
    result = iris_with_cbshap.explain([6.5, 3.0, 5.2, 2.0])

    result.each_value do |explanation|
      contribs = explanation[:contributions]
      abs_values = contribs.values.map(&:abs)
      assert_equal abs_values.sort.reverse, abs_values,
        "contributions Hash should iterate in |phi| descending order"
    end
  end

  def test_explain_reconstructs_raw_from_contributions_plus_bias
    result = iris_with_cbshap.explain([5.9, 3.0, 4.2, 1.5])
    result.each_value do |explanation|
      sum = explanation[:contributions].values.sum + explanation[:bias]
      assert_in_delta explanation[:raw], sum, 1e-9
    end
  end

  def test_explain_matches_algorithm2_oracle_within_float16_tolerance
    # 5e-3 leaves headroom for float16 per-tree drift on small models.
    iris_cbshap_ref["oracle_rows"].each_with_index do |row, i|
      result = iris_with_cbshap.explain(row["features"])
      iris_with_cbshap.class_names.each_with_index do |cls, k|
        contribs = result[cls][:contributions]
        iris_with_cbshap.schema.float_names.each_with_index do |name, fi|
          oracle_value = row["oracle_phi"][fi][k]
          diff = (oracle_value - contribs[name]).abs
          assert diff < 5e-3,
            "row #{i} class #{cls} feature #{name}: drift #{diff.inspect}"
        end
      end
    end
  end

  def test_explain_accepts_hash_and_array_inputs_equivalently
    model = iris_with_cbshap
    feature_names = model.schema.float_names
    array_input = [5.1, 3.5, 1.4, 0.2]
    hash_input = feature_names.zip(array_input).to_h

    array_result = model.explain(array_input)
    hash_result = model.explain(hash_input)

    assert_equal array_result, hash_result
  end

  def test_explain_accepts_symbol_keyed_hash
    model = iris_with_cbshap
    hash_input = model.schema.float_names.map(&:to_sym).zip([5.1, 3.5, 1.4, 0.2]).to_h
    result = model.explain(hash_input)
    assert_equal model.class_names, result.keys
  end

  def test_explain_raises_helpful_error_without_cbshap
    model = CatBoost::Model.load(fixture_path("iris.cbm"))
    refute model.explainer?
    err = assert_raises(CatBoost::Error) { model.explain([5.1, 3.5, 1.4, 0.2]) }
    assert_match(/no CBSHAP explainer/, err.message)
  end

  def test_explain_raises_after_close
    model = CatBoost::Model.load(fixture_path("iris.cbm"), cbshap: iris_cbshap_path)
    assert model.explainer?
    model.close
    assert_raises(CatBoost::Error) { model.explain([5.1, 3.5, 1.4, 0.2]) }
  end

  def test_explainer_released_on_close
    model = CatBoost::Model.load(fixture_path("iris.cbm"), cbshap: iris_cbshap_path)
    assert model.explainer?
    model.close
    refute model.explainer?
  end

  def test_load_raises_when_sidecar_file_missing
    assert_raises(CatBoost::LoadError) do
      CatBoost::Model.load(fixture_path("iris.cbm"), cbshap: fixture_path("does_not_exist.cbshap"))
    end
  end

  def test_load_rejects_model_with_categorical_features
    err = assert_raises(CatBoost::Cbshap::IncompatibleError) do
      CatBoost::Model.load(fixture_path("churn_mixed.cbm"), cbshap: iris_cbshap_path)
    end
    assert_match(/categorical features/, err.message)
  end

  def test_load_rejects_sidecar_with_wrong_feature_count
    # n_features lives at offset 20: magic(8) + version(4) + n_trees(4) + depth(4).
    bytes = File.binread(iris_cbshap_path)
    bytes.setbyte(20, 99)

    Tempfile.create(["bad-dims", ".cbshap"]) do |f|
      f.binmode
      f.write(bytes)
      f.close
      err = assert_raises(CatBoost::Cbshap::IncompatibleError) do
        CatBoost::Model.load(fixture_path("iris.cbm"), cbshap: f.path)
      end
      assert_match(/n_features/, err.message)
    end
  end

  def test_incompatible_error_rescues_as_catboost_error
    assert_raises(CatBoost::Error) do
      CatBoost::Model.load(fixture_path("churn_mixed.cbm"), cbshap: iris_cbshap_path)
    end
  end

  # ----- Standalone Explainer (composable path) ----------------------------

  def test_standalone_explainer_works_with_manually_built_reader
    reader = CatBoost::Cbshap::Reader.new(iris_cbshap_path)
    explainer = CatBoost::Cbshap::Explainer.new(
      reader,
      feature_names: iris_model.schema.float_names,
      class_names: iris_model.class_names
    )
    manual = explainer.explain([5.1, 3.5, 1.4, 0.2])
    auto = iris_with_cbshap.explain([5.1, 3.5, 1.4, 0.2])
    assert_equal manual, auto
  end

  def test_standalone_explainer_rejects_feature_name_size_mismatch
    reader = CatBoost::Cbshap::Reader.new(iris_cbshap_path)
    err = assert_raises(ArgumentError) do
      CatBoost::Cbshap::Explainer.new(reader, feature_names: %w[a b c], class_names: [0, 1, 2])
    end
    assert_match(/feature_names size/, err.message)
  end

  def test_standalone_explainer_rejects_class_name_size_mismatch
    reader = CatBoost::Cbshap::Reader.new(iris_cbshap_path)
    err = assert_raises(ArgumentError) do
      CatBoost::Cbshap::Explainer.new(
        reader,
        feature_names: iris_model.schema.float_names,
        class_names: %w[only_one]
      )
    end
    assert_match(/class_names size/, err.message)
  end
end
