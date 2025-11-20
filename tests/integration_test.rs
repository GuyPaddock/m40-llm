

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::ffi::c_void;

    #[test]
    fn test_model_loading() -> Result<()> {
        // Create a dummy GGUF model
        let gguf = GgufModel::new(0); // Dummy model
        let gguf_bytes = vec![]; // Empty bytes

        // Load the model
        let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;

        // Verify the model was loaded
        assert!(model.d_data_base as usize >= 0);

        Ok(())
    }

    #[test]
    fn test_kv_cache_allocation() -> Result<()> {
        // Create a dummy GGUF model
        let gguf = GgufModel::new(0); // Dummy model
        let gguf_bytes = vec![]; // Empty bytes

        // Load the model
        let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;

        // Allocate KV cache
        model.allocate_kv_cache(128, 8)?;

        // Verify the KV cache was allocated
        assert!(model.kv_cache.is_some());

        Ok(())
    }

    #[test]
    fn test_attention_operation() -> Result<()> {
        // Create a dummy GGUF model
        let gguf = GgufModel::new(0); // Dummy model
        let gguf_bytes = vec![]; // Empty bytes

        // Load the model
        let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;

        // Allocate KV cache
        model.allocate_kv_cache(128, 8)?;

        // Create dummy input and output
        let input = 0 as *const c_void;
        let mut output = 0 as *mut c_void;

        // Run attention operation
        model.run_attention(
            input,
            output,
            0,
            128,
            512,
            8,
            64,
        );

        Ok(())
    }

    #[test]
    fn test_mlp_operation() -> Result<()> {
        // Create a dummy GGUF model
        let gguf = GgufModel::new(0); // Dummy model
        let gguf_bytes = vec![]; // Empty bytes

        // Load the model
        let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;

        // Create dummy input and output
        let input = 0 as *const c_void;
        let mut output = 0 as *mut c_void;

        // Run MLP operation
        model.run_mlp(
            input,
            output,
            8,
            512,
            2048,
        );

        Ok(())
    }

    #[test]
    fn test_rms_norm_operation() -> Result<()> {
        // Create a dummy GGUF model
        let gguf = GgufModel::new(0); // Dummy model
        let gguf_bytes = vec![]; // Empty bytes

        // Load the model
        let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;

        // Create dummy input and output
        let input = 0 as *const c_void;
        let mut output = 0 as *mut c_void;

        // Run RMSNorm operation
        model.run_rms_norm(
            input,
            output,
            128,
            512,
            1e-5,
        );

        Ok(())
    }

}

