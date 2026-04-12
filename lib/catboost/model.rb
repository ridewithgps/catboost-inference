require "ffi"
require "json"

module CatBoost
  # FFI::AutoPointer subclass whose release method frees the CatBoost model
  # handle when GC collects the Ruby object. An explicit #free marks the
  # releaser done so GC won't double-free.
  class ManagedHandle < ::FFI::AutoPointer
    def self.release(ptr)
      LibFFI.ModelCalcerDelete(ptr) unless ptr.null?
    end
  end

  class Model
    # Prediction types the C API accepts. Keys are the Ruby-idiomatic symbol
    # form, values are the CamelCase strings the C API expects. Callers may
    # pass either form. Not every type is valid for every model:
    # :log_probability is multiclass-only, :multi_probability is for
    # MultiLogloss, :rmse_with_uncertainty is for regression with uncertainty.
    # Unsupported combinations raise CatBoost::PredictError at call time.
    PREDICTION_TYPES = {
      raw_formula_val:       "RawFormulaVal",
      probability:           "Probability",
      multi_probability:     "MultiProbability",
      class:                 "Class",
      exponent:              "Exponent",
      log_probability:       "LogProbability",
      rmse_with_uncertainty: "RMSEWithUncertainty"
    }.freeze

    def self.load(path, cbshap: nil)
      raise LoadError, "No such file: #{path}" unless File.file?(path)

      handle = allocate_handle
      unless LibFFI.LoadFullModelFromFile(handle, path)
        err = LibFFI.GetErrorString || "(unknown)"
        handle.free
        raise LoadError, "LoadFullModelFromFile(#{path}) failed: #{err}"
      end
      new(handle, cbshap: cbshap)
    end

    def self.load_from_buffer(bytes, cbshap: nil)
      raise ArgumentError, "bytes must be a String" unless bytes.is_a?(String)

      handle = allocate_handle
      buf = ::FFI::MemoryPointer.new(:uint8, bytes.bytesize)
      buf.write_bytes(bytes)

      unless LibFFI.LoadFullModelFromBuffer(handle, buf, bytes.bytesize)
        err = LibFFI.GetErrorString || "(unknown)"
        handle.free
        raise LoadError, "LoadFullModelFromBuffer failed: #{err}"
      end
      new(handle, cbshap: cbshap)
    end

    def self.allocate_handle
      raw = LibFFI.ModelCalcerCreate
      if raw.null?
        err = LibFFI.GetErrorString || "ModelCalcerCreate returned null"
        raise LoadError, "ModelCalcerCreate failed: #{err}"
      end
      ManagedHandle.new(raw)
    end

    def initialize(handle, cbshap: nil)
      @handle = handle
      @float_features_count = LibFFI.GetFloatFeaturesCount(@handle)
      @cat_features_count   = LibFFI.GetCatFeaturesCount(@handle)
      @tree_count           = LibFFI.GetTreeCount(@handle)
      @dimensions_count     = LibFFI.GetDimensionsCount(@handle)
      @schema               = FeatureSchema.from_handle(@handle)
      @explainer            = Cbshap::Explainer.for_model(self, cbshap) if cbshap
    end

    attr_reader :float_features_count, :cat_features_count, :tree_count, :dimensions_count, :schema

    # Convenience: { floats: [...], cats: [...] } name listing.
    def feature_names
      @schema.to_h
    end

    # ----- Metadata ---------------------------------------------------------

    # Reads a value from CatBoost's built-in model-metadata key-value store.
    # Mirrors Python's `model.get_metadata()["key"]`. Standard keys written
    # at training time:
    #
    #   class_params           JSON: classifier label list + label type
    #   params                 JSON: full training hyperparameters
    #   model_guid             Unique ID generated at training
    #   train_finish_time      ISO-8601 timestamp
    #   catboost_version_info  libcatboost version + git commit
    #   output_options         JSON: output options struct
    #   training               JSON: training stats
    #
    # Returns the raw String value (JSON blobs are left un-parsed by
    # design — callers that want structured access parse themselves).
    # Returns nil if the key is absent. Symbol and String keys are
    # equivalent. Raises CatBoost::Error if the Model has been closed.
    #
    # There is no key-enumeration function in the v1.2 C API, so there
    # is no #each / #keys / Hash return — callers must know which key
    # they want.
    def metadata(key)
      ensure_open!
      key = key.to_s
      return nil unless LibFFI.CheckModelMetadataHasKey(@handle, key, key.bytesize)

      size = LibFFI.GetModelInfoValueSize(@handle, key, key.bytesize)
      return nil if size.zero?

      ptr = LibFFI.GetModelInfoValue(@handle, key, key.bytesize)
      return nil if ptr.null?

      ptr.read_string(size)
    end

    # Presence check for a metadata key without reading the value.
    # Matches `Hash#key?` in shape — `model.metadata?(:class_params)`.
    def metadata?(key)
      ensure_open!
      key = key.to_s
      LibFFI.CheckModelMetadataHasKey(@handle, key, key.bytesize)
    end

    # Ordered list of class labels for classification models, read from
    # the `class_params` metadata key and parsed out of its JSON payload.
    # Returns:
    #   - Array<String> if the model was trained with explicit string
    #     labels (e.g. `class_names=["setosa","versicolor","virginica"]`)
    #   - Array<Integer> if no explicit labels were passed at training
    #     time (CatBoost falls back to integer class indices)
    #   - nil for regression or ranking models with no `class_params` key
    #
    # Memoized — the result is immutable once the model is loaded.
    # A malformed `class_params` JSON payload (which should never happen
    # for a legitimately-trained CatBoost model) is wrapped in
    # CatBoost::Error so all gem-visible errors stay in one hierarchy.
    def class_names
      return @class_names if defined?(@class_names)

      raw = metadata("class_params")
      @class_names = raw ? JSON.parse(raw)["class_names"] : nil
    rescue JSON::ParserError => e
      raise CatBoost::Error, "malformed class_params metadata: #{e.message}"
    end

    # ----- Prediction -------------------------------------------------------

    # predict accepts five input shapes: a positional Array (single row or
    # batch), an explicit Hash, keyword arguments, or a batch of Hashes.
    # Categorical values may be Symbol or String — they're converted via #to_s.
    def predict(input = nil, cat_features: nil, prediction_type: :raw_formula_val, **named_features)
      ensure_open!
      input = resolve_input(input, named_features)
      dim = set_prediction_type(prediction_type)
      dispatch_predict(input, cat_features, dim)
    end

    # Locks prediction_type to :probability (softmax). Correct for binary and
    # multiclass models. For multi-label (MultiLogloss) models, use
    # predict_multi_proba instead — see README "Multi-label note".
    def predict_proba(input = nil, cat_features: nil, **named_features)
      reject_prediction_type_kwarg!("predict_proba", :probability, named_features)
      predict(input, cat_features: cat_features, prediction_type: :probability, **named_features)
    end

    # Locks prediction_type to :multi_probability (independent per-label
    # sigmoids). This is the correct method for MultiLogloss (multi-label)
    # models — each probability is independent and they do NOT sum to 1.
    # For multiclass or binary models, use predict_proba instead.
    def predict_multi_proba(input = nil, cat_features: nil, **named_features)
      reject_prediction_type_kwarg!("predict_multi_proba", :multi_probability, named_features)
      predict(input, cat_features: cat_features, prediction_type: :multi_probability, **named_features)
    end

    # ----- Explanation ------------------------------------------------------

    # Per-class SHAP explanation for a single row. Requires the Model
    # was loaded with a `cbshap:` sidecar — see CatBoost::Cbshap.
    def explain(features)
      ensure_open!
      raise Error, "Model has no CBSHAP explainer attached; load with `cbshap: path` to enable #explain" if @explainer.nil?

      @explainer.explain(features)
    end

    def explainer?
      !@explainer.nil?
    end

    # ----- Lifecycle --------------------------------------------------------

    def closed?
      @handle.nil?
    end

    def close
      return if @handle.nil?

      @handle.free
      @handle = nil
      @explainer = nil
    end

    def inspect
      status = closed? ? " CLOSED" : ""
      "#<CatBoost::Model trees=#{@tree_count} floats=#{@float_features_count} " \
        "cats=#{@cat_features_count} dim=#{@dimensions_count}#{status}>"
    end
    alias_method :to_s, :inspect

    private

    def ensure_open!
      raise Error, "Model is closed" if @handle.nil?
    end

    def reject_prediction_type_kwarg!(method_name, locked_type, named_features)
      return unless named_features.key?(:prediction_type)

      raise ArgumentError,
            "#{method_name} locks prediction_type to :#{locked_type}; " \
            "call predict(..., prediction_type: #{named_features[:prediction_type].inspect}) instead"
    end

    def resolve_input(input, named_features)
      return named_features if input.nil? && !named_features.empty?
      raise ArgumentError, "predict requires a positional input or named feature kwargs" if input.nil?
      raise ArgumentError, "cannot mix positional input with named feature kwargs (#{named_features.keys.inspect})" unless named_features.empty?

      input
    end

    def dispatch_predict(input, cat_features, dim)
      case input
      when Hash
        floats, cats = @schema.split_row(input)
        predict_single(floats, cats, dim)
      when Array
        dispatch_array(input, cat_features, dim)
      else
        raise ArgumentError, "input must be an Array or Hash, got #{input.class}"
      end
    end

    def dispatch_array(input, cat_features, dim)
      return predict_single(input, cat_features, dim) if input.empty? || input.first.is_a?(Numeric)

      case input.first
      when Hash
        floats_rows, cats_rows = input.map { |row| @schema.split_row(row) }.transpose
        predict_batch(floats_rows, cats_rows, dim)
      when Array
        predict_batch(input, cat_features, dim)
      else
        raise ArgumentError, "batch rows must be Arrays or Hashes, got #{input.first.class} at index 0"
      end
    end

    # Reads the per-row output dim IMMEDIATELY after setting the type — do not
    # cache on @, because the handle is mutated by SetPredictionTypeString and
    # another thread could switch it before our Calc runs.
    def set_prediction_type(type)
      str = resolve_prediction_type_string(type)
      LibFFI.check!(LibFFI.SetPredictionTypeString(@handle, str), "SetPredictionTypeString(#{str})")
      LibFFI.GetPredictionDimensionsCount(@handle)
    end

    # Accepts either a Ruby symbol (`:probability`) or a Python-style string
    # (`"Probability"`), returns the CamelCase C API string.
    def resolve_prediction_type_string(type)
      case type
      when Symbol
        PREDICTION_TYPES[type] || raise_unknown_prediction_type(type)
      when String
        PREDICTION_TYPES.value?(type) ? type : raise_unknown_prediction_type(type)
      else
        raise ArgumentError, "prediction_type must be a Symbol or String, got #{type.class}"
      end
    end

    def raise_unknown_prediction_type(type)
      raise ArgumentError,
            "Unknown prediction_type #{type.inspect}; expected one of " \
            "#{PREDICTION_TYPES.keys.inspect} or #{PREDICTION_TYPES.values.inspect}"
    end

    def predict_single(floats, cats, dim)
      validate_row!(floats, @float_features_count, "float")
      cats ||= []
      validate_row!(cats, @cat_features_count, "cat")

      float_ptr = write_float_pointer(floats)
      cat_ptr, _keep = build_cat_ptr(cats)
      result = ::FFI::MemoryPointer.new(:double, dim)

      LibFFI.check!(
        LibFFI.CalcModelPredictionSingle(
          @handle,
          float_ptr, @float_features_count,
          cat_ptr,   @cat_features_count,
          result,    dim
        ),
        "CalcModelPredictionSingle"
      )

      result.read_array_of_double(dim)
    end

    def predict_batch(rows, cat_rows, dim)
      n = rows.size
      rows.each_with_index { |r, i| validate_row!(r, @float_features_count, "float", row_index: i) }

      return predict_batch_flat(rows, dim) if @cat_features_count.zero?

      raise ArgumentError, "model has #{@cat_features_count} cat features; cat_features: must be provided for batch" if cat_rows.nil?
      raise ArgumentError, "cat_features batch size #{cat_rows.size} != float batch size #{n}" unless cat_rows.size == n

      cat_rows.each_with_index { |c, i| validate_row!(c, @cat_features_count, "cat", row_index: i) }
      predict_batch_mixed(rows, cat_rows, dim)
    end

    def predict_batch_flat(rows, dim)
      n = rows.size
      f = @float_features_count
      flat = write_flat_float_matrix(rows)
      row_ptrs = build_row_pointer_array(flat, n, f)
      result = ::FFI::MemoryPointer.new(:double, n * dim)

      LibFFI.check!(
        LibFFI.CalcModelPredictionFlat(@handle, n, row_ptrs, f, result, n * dim),
        "CalcModelPredictionFlat"
      )

      result.read_array_of_double(n * dim).each_slice(dim).to_a
    end

    def predict_batch_mixed(rows, cat_rows, dim)
      n = rows.size
      f = @float_features_count
      c = @cat_features_count

      flat = write_flat_float_matrix(rows)
      row_ptrs = build_row_pointer_array(flat, n, f)
      cat_matrix, _keep = build_cat_matrix(cat_rows, n, c)
      result = ::FFI::MemoryPointer.new(:double, n * dim)

      LibFFI.check!(
        LibFFI.CalcModelPrediction(@handle, n, row_ptrs, f, cat_matrix, c, result, n * dim),
        "CalcModelPrediction"
      )

      result.read_array_of_double(n * dim).each_slice(dim).to_a
    end

    def write_float_pointer(floats)
      ptr = ::FFI::MemoryPointer.new(:float, @float_features_count)
      ptr.write_array_of_float(floats.map(&:to_f)) unless @float_features_count.zero?
      ptr
    end

    def write_flat_float_matrix(rows)
      f = @float_features_count
      ptr = ::FFI::MemoryPointer.new(:float, rows.size * f)
      ptr.write_array_of_float(rows.flatten.map(&:to_f)) unless f.zero?
      ptr
    end

    def build_row_pointer_array(flat, n, f)
      row_ptrs = ::FFI::MemoryPointer.new(:pointer, n)
      stride = f * ::FFI.type_size(:float)
      n.times { |i| row_ptrs[i].write_pointer(flat + i * stride) }
      row_ptrs
    end

    # Returns [char**, keep] where `keep` is the Ruby array of underlying
    # string pointers. The caller MUST hold `keep` in a local variable until
    # after the FFI call returns — dropping it before Calc is a use-after-free.
    def build_cat_ptr(cats)
      return [::FFI::Pointer::NULL, nil] if @cat_features_count.zero?

      str_ptrs = cats.map { |s| ::FFI::MemoryPointer.from_string(s.to_s) }
      arr = ::FFI::MemoryPointer.new(:pointer, @cat_features_count)
      str_ptrs.each_with_index { |sp, i| arr[i].write_pointer(sp) }
      [arr, str_ptrs]
    end

    # Returns [char***, keep] where `keep` holds every per-row char** and
    # every underlying string pointer so GC cannot collect them before the
    # FFI call returns.
    def build_cat_matrix(cat_rows, n, c)
      keep = { strings: [], inner: [] }
      outer = ::FFI::MemoryPointer.new(:pointer, n)

      cat_rows.each_with_index do |row, i|
        row_strs = row.map { |s| ::FFI::MemoryPointer.from_string(s.to_s) }
        keep[:strings].concat(row_strs)

        inner = ::FFI::MemoryPointer.new(:pointer, c)
        row_strs.each_with_index { |sp, j| inner[j].write_pointer(sp) }
        keep[:inner] << inner

        outer[i].write_pointer(inner)
      end

      [outer, keep]
    end

    def validate_row!(row, expected_size, feature_kind, row_index: nil)
      raise ArgumentError, "#{feature_kind} row must be an Array, got #{row.class}" unless row.is_a?(Array)
      return if row.size == expected_size

      where = row_index ? " at index #{row_index}" : ""
      raise ArgumentError, "expected #{expected_size} #{feature_kind} features#{where}, got #{row.size}"
    end
  end
end
