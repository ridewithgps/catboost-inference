module CatBoost
  # Translates between the CatBoost C API's positional feature layout and
  # Ruby-idiomatic {name => value} Hashes.
  #
  # A CatBoost model has N total feature positions. Some are float features,
  # others categorical. The C API wants predictions with floats and cats in
  # two SEPARATE arrays, each with its own per-kind index. A user thinking in
  # names ("tier", "monthly_spend") needs a translation layer. FeatureSchema
  # is that layer, built once per Model at load time from three C API calls:
  # GetModelUsedFeaturesNames, GetFloatFeatureIndices, GetCatFeatureIndices.
  #
  # Hash keys may be Symbol or String — `row[:tier]` and `row["tier"]` are
  # equivalent. Unknown keys raise ArgumentError; missing keys raise KeyError.
  # Positional callers (Array input to Model#predict) skip this class entirely.
  class FeatureSchema
    attr_reader :float_names, :cat_names, :all_names

    # Reads the schema from a live handle via the C API.
    def self.from_handle(handle)
      names = read_feature_names(handle)
      float_positions = read_indices(handle, :GetFloatFeatureIndices)
      cat_positions   = read_indices(handle, :GetCatFeatureIndices)

      new(
        float_names: float_positions.map { |pos| names[pos] },
        cat_names:   cat_positions.map   { |pos| names[pos] },
        all_names:   names
      )
    end

    def initialize(float_names:, cat_names:, all_names: nil)
      @float_names = float_names.freeze
      @cat_names   = cat_names.freeze
      @all_names   = (all_names || (float_names + cat_names)).freeze

      @name_index = {}
      float_names.each_with_index { |name, i| @name_index[name] = [:float, i] }
      cat_names.each_with_index   { |name, i| @name_index[name] = [:cat,   i] }
    end

    def to_h
      { floats: float_names, cats: cat_names }
    end

    # True if the model was trained with explicit feature_names. False for
    # synthetic "0", "1", ... names — the Hash-keyed predict path isn't
    # useful for those; use the positional API.
    def named?
      (@float_names + @cat_names).none? { |n| n =~ /\A\d+\z/ }
    end

    # Translates a Hash row (mixing Symbol and String keys is fine) into the
    # pair of positional arrays [floats, cats] the C API wants.
    def split_row(row)
      raise ArgumentError, "row must be a Hash, got #{row.class}" unless row.is_a?(Hash)

      canonicalized = {}
      row.each do |key, value|
        name = key.to_s
        raise ArgumentError, "unknown feature #{key.inspect}; known: #{@name_index.keys.inspect}" unless @name_index.key?(name)
        canonicalized[name] = value
      end

      [@float_names.map { |name| fetch_feature(canonicalized, name, "float") },
       @cat_names.map   { |name| fetch_feature(canonicalized, name, "cat")   }]
    end

    private

    def fetch_feature(row, name, kind)
      raise KeyError, "missing #{kind} feature #{name.inspect}" unless row.key?(name)
      row.fetch(name)
    end

    def self.read_feature_names(handle)
      out_names = ::FFI::MemoryPointer.new(:pointer, 1)
      out_count = ::FFI::MemoryPointer.new(:size_t, 1)
      LibFFI.check!(LibFFI.GetModelUsedFeaturesNames(handle, out_names, out_count),
                    "GetModelUsedFeaturesNames")

      count = out_count.read(:size_t)
      arr_ptr = out_names.read_pointer
      pointer_size = ::FFI.type_size(:pointer)

      begin
        Array.new(count) do |i|
          str_ptr = arr_ptr[i * pointer_size].read_pointer
          begin
            str_ptr.read_string
          ensure
            LibC.free(str_ptr)
          end
        end
      ensure
        LibC.free(arr_ptr)
      end
    end

    def self.read_indices(handle, getter)
      out_indices = ::FFI::MemoryPointer.new(:pointer, 1)
      out_count   = ::FFI::MemoryPointer.new(:size_t, 1)
      LibFFI.check!(LibFFI.send(getter, handle, out_indices, out_count), getter.to_s)

      count = out_count.read(:size_t)
      arr_ptr = out_indices.read_pointer
      size_t_size = ::FFI.type_size(:size_t)

      begin
        Array.new(count) { |i| arr_ptr[i * size_t_size].read(:size_t) }
      ensure
        LibC.free(arr_ptr)
      end
    end

    private_class_method :read_feature_names, :read_indices
  end
end
