module CatBoost
  # Reader and Explainer for CBSHAP v1 sidecars (format defined by python/cbshap.py).
  module Cbshap
    # Raised when a sidecar file is malformed: bad magic, truncated, unknown version.
    class ParseError < CatBoost::Error; end

    # Raised when a sidecar parses fine but does not match the model it
    # was paired with — wrong feature count, wrong class count, or a
    # model with categorical features (which CBSHAP cannot encode).
    class IncompatibleError < CatBoost::Error; end

    class Reader
      MAGIC = "CBSHAP\x01\x00".b

      attr_reader :n_trees, :depth, :n_features, :n_classes, :version, :bias_row

      def initialize(path)
        @buf = File.binread(path)
        parse_header!
        parse_trees!
      end

      # Return the (n_features + 1) x n_classes SHAP matrix for row x.
      # Rows 0..n_features-1 are per-feature contributions; the last row
      # is the precomputed bias (model expected value per class).
      def shap_values(x)
        if x.size != @n_features
          raise ArgumentError, "expected #{@n_features} features, got #{x.size}"
        end

        phi = Array.new(@n_features + 1) { Array.new(@n_classes, 0.0) }

        @n_trees.times do |t|
          split_features = @split_features[t]
          split_borders  = @split_borders[t]
          leaf = 0
          @depth.times do |d|
            leaf |= (1 << d) if x[split_features[d]] > split_borders[d]
          end

          n_uniq = @n_unique[t]
          uniq_features = @unique_features[t]
          packed = @shap_packed[t]

          row_offset = leaf * @depth * @n_classes
          n_uniq.times do |i|
            feat = uniq_features[i]
            base = row_offset + i * @n_classes
            phi_row = phi[feat]
            @n_classes.times do |k|
              phi_row[k] += packed[base + k]
            end
          end
        end

        phi[@n_features] = @bias_row.dup
        phi
      end

      private

      def parse_header!
        raise ParseError, "file too small for CBSHAP header (#{@buf.bytesize} bytes)" if @buf.bytesize < 36

        magic = @buf.byteslice(0, 8)
        raise ParseError, "not a CBSHAP file: got magic #{magic.inspect}" unless magic == MAGIC

        @version, @n_trees, @depth, @n_features, @n_classes =
          @buf.byteslice(8, 20).unpack("V5")

        raise ParseError, "unsupported CBSHAP version #{@version}" unless @version == 1

        # Magic (8) + V5 header (20) + reserved u64 (8) = 36 bytes before bias_row.
        bias_off = 36
        @header_size = bias_off + 4 * @n_classes
        raise ParseError, "truncated CBSHAP header (need #{@header_size} bytes, got #{@buf.bytesize})" if @buf.bytesize < @header_size

        @bias_row = @buf.byteslice(bias_off, 4 * @n_classes).unpack("e#{@n_classes}")
      end

      def parse_trees!
        d = @depth
        k = @n_classes
        leaves = 1 << d
        shap_floats_per_tree = leaves * d * k
        shap_bytes_per_tree = shap_floats_per_tree * 2 # Float16 = 2 bytes each

        raw_record = d + 4 * d + d + 1 + 1 + shap_bytes_per_tree
        per_tree = (raw_record + 3) & ~3 # round up to 4-byte alignment

        total_size = @header_size + @n_trees * per_tree
        raise ParseError, "truncated CBSHAP body (need #{total_size} bytes, got #{@buf.bytesize})" if @buf.bytesize < total_size

        @split_features  = Array.new(@n_trees)
        @split_borders   = Array.new(@n_trees)
        @unique_features = Array.new(@n_trees)
        @n_unique        = Array.new(@n_trees)
        @shap_packed     = Array.new(@n_trees)

        off = @header_size
        @n_trees.times do |t|
          rec = off
          @split_features[t] = @buf.byteslice(rec, d).unpack("C#{d}")
          rec += d
          @split_borders[t] = @buf.byteslice(rec, 4 * d).unpack("e#{d}")
          rec += 4 * d
          @unique_features[t] = @buf.byteslice(rec, d).unpack("C#{d}")
          rec += d
          @n_unique[t] = @buf.getbyte(rec)
          rec += 1
          rec += 1 # pad

          # Ruby stdlib has no IEEE 754 binary16 unpack — Array#pack "e" is
          # float32. Pull raw 16-bit words and decode by hand.
          raw16 = @buf.byteslice(rec, shap_bytes_per_tree).unpack("v#{shap_floats_per_tree}")
          @shap_packed[t] = raw16.map! { |bits| float16_to_f(bits) }
          off += per_tree
        end
      end

      # IEEE 754 binary16 decoder. Layout: 1 sign | 5 exponent | 10 mantissa.
      def float16_to_f(bits)
        sign = (bits >> 15) & 0x1
        exp  = (bits >> 10) & 0x1F
        frac = bits & 0x3FF

        if exp == 0
          return frac == 0 ? (sign == 1 ? -0.0 : 0.0) : (sign == 1 ? -1.0 : 1.0) * frac * 2.0**-24
        end

        if exp == 0x1F
          return Float::NAN if frac != 0

          return sign == 1 ? -Float::INFINITY : Float::INFINITY
        end

        value = (1.0 + frac / 1024.0) * 2.0**(exp - 15)
        sign == 1 ? -value : value
      end
    end

    # Wraps Reader with feature/class names, returning a Hash of per-class
    # explanations. Built automatically by CatBoost::Model.load when
    # `cbshap:` is passed; can also be constructed manually to share one
    # Reader across models.
    class Explainer
      attr_reader :feature_names, :class_names, :reader

      # Build an Explainer for an already-loaded Model, validating the
      # sidecar against the model's dimensions before wrapping.
      def self.for_model(model, cbshap_path)
        raise CatBoost::LoadError, "cbshap sidecar not found: #{cbshap_path}" unless File.file?(cbshap_path)

        unless model.cat_features_count.zero?
          raise IncompatibleError,
                "CBSHAP does not support models with categorical features " \
                "(model has #{model.cat_features_count} cat features)"
        end

        reader = Reader.new(cbshap_path)

        if reader.n_features != model.float_features_count
          raise IncompatibleError,
                "cbshap n_features=#{reader.n_features} does not match model " \
                "float_features_count=#{model.float_features_count}"
        end
        if reader.n_classes != model.dimensions_count
          raise IncompatibleError,
                "cbshap n_classes=#{reader.n_classes} does not match model " \
                "dimensions_count=#{model.dimensions_count}"
        end

        # Use trained class_names when their count matches the sidecar's
        # n_classes; fall back to integer indices for binary models
        # (2 labels, 1 logit dim) and regression/ranking models.
        names = model.class_names
        labels = names && names.size == reader.n_classes ? names : Array.new(reader.n_classes) { |i| i }

        new(reader, feature_names: model.schema.float_names, class_names: labels)
      end

      def initialize(reader, feature_names:, class_names:)
        @reader = reader
        @feature_names = feature_names.freeze
        @class_names = class_names.freeze

        if feature_names.size != reader.n_features
          raise ArgumentError,
                "feature_names size #{feature_names.size} != reader.n_features #{reader.n_features}"
        end
        if class_names.size != reader.n_classes
          raise ArgumentError,
                "class_names size #{class_names.size} != reader.n_classes #{reader.n_classes}"
        end
      end

      # features is an Array<Numeric> in feature_names order or a
      # Hash{String|Symbol => Numeric} keyed by name.
      #
      # Returns {class_name => {raw:, bias:, contributions: {name => phi, ...}}}.
      # contributions iterates in |phi| descending order.
      def explain(features)
        vec = vectorize(features)
        phi = @reader.shap_values(vec)

        result = {}
        @class_names.each_with_index do |cls, k|
          pairs = @feature_names.each_with_index.map { |name, fi| [name, phi[fi][k]] }
          bias = @reader.bias_row[k]
          raw = pairs.sum { |_, v| v } + bias
          result[cls] = {
            raw: raw,
            bias: bias,
            contributions: pairs.sort_by { |_, v| -v.abs }.to_h
          }
        end
        result
      end

      private

      def vectorize(features)
        case features
        when Array
          if features.size != @feature_names.size
            raise ArgumentError,
                  "expected #{@feature_names.size} features, got #{features.size}"
          end
          features
        when Hash
          @feature_names.map do |name|
            if features.key?(name)
              features[name]
            elsif features.key?(name.to_sym)
              features[name.to_sym]
            else
              raise KeyError, "missing feature #{name.inspect}; " \
                              "caller must provide every feature the model was trained on"
            end
          end
        else
          raise ArgumentError, "features must be an Array or Hash, got #{features.class}"
        end
      end
    end
  end
end
