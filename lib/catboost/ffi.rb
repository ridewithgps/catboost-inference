require "ffi"

module CatBoost
  module LibFFI
    extend ::FFI::Library

    ffi_lib CatBoost.library_path

    attach_function :ModelCalcerCreate, [], :pointer
    attach_function :ModelCalcerDelete, [:pointer], :void
    attach_function :GetErrorString, [], :string

    attach_function :LoadFullModelFromFile, [:pointer, :string], :bool
    attach_function :LoadFullModelFromBuffer, [:pointer, :pointer, :size_t], :bool

    attach_function :GetFloatFeaturesCount,        [:pointer], :size_t
    attach_function :GetCatFeaturesCount,          [:pointer], :size_t
    attach_function :GetTreeCount,                 [:pointer], :size_t
    attach_function :GetDimensionsCount,           [:pointer], :size_t
    attach_function :GetPredictionDimensionsCount, [:pointer], :size_t

    # Feature metadata. These three functions write into caller-provided
    # output pointers, and they malloc their outputs — the caller owns the
    # memory and must free it with libc free. See CatBoost::FeatureSchema for
    # the wrapper that handles that lifecycle.
    attach_function :GetModelUsedFeaturesNames, [:pointer, :pointer, :pointer], :bool
    attach_function :GetFloatFeatureIndices,    [:pointer, :pointer, :pointer], :bool
    attach_function :GetCatFeatureIndices,      [:pointer, :pointer, :pointer], :bool

    # Generic model-info key-value store. Every .cbm file ships with a set
    # of keys baked in at training time (class_params, params, model_guid,
    # train_finish_time, catboost_version_info, output_options, training).
    # GetModelInfoValue returns a pointer into the model's internal buffer —
    # it must NOT be freed. There is no key enumeration function in the
    # v1.2 C API; callers must know which key they want. See
    # CatBoost::Model#metadata for the public wrapper.
    attach_function :CheckModelMetadataHasKey, [:pointer, :string, :size_t], :bool
    attach_function :GetModelInfoValueSize,    [:pointer, :string, :size_t], :size_t
    attach_function :GetModelInfoValue,        [:pointer, :string, :size_t], :pointer

    attach_function :SetPredictionTypeString, [:pointer, :string], :bool

    attach_function :CalcModelPredictionSingle,
                    [:pointer,
                     :pointer, :size_t,
                     :pointer, :size_t,
                     :pointer, :size_t],
                    :bool

    attach_function :CalcModelPredictionFlat,
                    [:pointer, :size_t,
                     :pointer, :size_t,
                     :pointer, :size_t],
                    :bool

    attach_function :CalcModelPrediction,
                    [:pointer, :size_t,
                     :pointer, :size_t,
                     :pointer, :size_t,
                     :pointer, :size_t],
                    :bool

    def self.check!(ok, context)
      return if ok

      err = GetErrorString() || "(no error string)"
      raise CatBoost::PredictError, "#{context}: #{err}"
    end
  end

  # Used to free malloc'd memory returned by GetModelUsedFeaturesNames and
  # the feature-index getters. Bound against a null-handle library
  # (RTLD_DEFAULT) rather than libc.so.6 so the symbol resolves through the
  # process's own loader order. On Rubies built --with-jemalloc (RVM's
  # default), jemalloc interposes malloc/free globally and libcatboostmodel
  # ends up allocating through it — pinning this to glibc's free would then
  # hand jemalloc-owned chunks to glibc and abort with "free(): invalid
  # pointer" on the first model load.
  module LibC
    extend ::FFI::Library
    # ffi_lib takes a path string; to bind against the process's own symbol
    # table we poke @ffi_libs directly with a null-handle DynamicLibrary
    # (dlopen(NULL) ≈ RTLD_DEFAULT). attach_function then resolves through
    # the same symbol table the main Ruby binary uses.
    @ffi_libs = [
      ::FFI::DynamicLibrary.open(
        nil,
        ::FFI::DynamicLibrary::RTLD_LAZY | ::FFI::DynamicLibrary::RTLD_GLOBAL
      )
    ]
    attach_function :free, [:pointer], :void
  end
end
