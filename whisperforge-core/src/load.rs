// Model loading utilities for Whisper models saved in MessagePack format
// Compatible with whisper-burn pre-converted models from HuggingFace

use anyhow::{Context, Result};
use burn::{
    module::{Module, ModuleMapper, Param},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{
        DType, Tensor, backend::Backend, ops::QuantizedTensor, quantization::QTensorPrimitive,
    },
};
use burn_flex::{Flex, FlexDevice};
use serde::Deserialize;

#[cfg(feature = "file-io")]
use std::path::Path;

use crate::model::{AudioEncoderConfig, TextDecoderConfig, Whisper, WhisperConfig};

/// Model precision for quantization
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelPrecision {
    Fp32,
    Int8,
}

/// Configuration file format for whisper-burn models (.cfg files)
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperModelConfig {
    pub audio_encoder_config: AudioEncoderConfigFile,
    pub text_decoder_config: TextDecoderConfigFile,
    #[serde(default)]
    pub precision: Option<ModelPrecision>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfigFile {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextDecoderConfigFile {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

impl From<WhisperModelConfig> for WhisperConfig {
    fn from(cfg: WhisperModelConfig) -> Self {
        WhisperConfig {
            audio_encoder_config: AudioEncoderConfig {
                n_mels: cfg.audio_encoder_config.n_mels,
                n_audio_ctx: cfg.audio_encoder_config.n_audio_ctx,
                n_audio_state: cfg.audio_encoder_config.n_audio_state,
                n_audio_head: cfg.audio_encoder_config.n_audio_head,
                n_audio_layer: cfg.audio_encoder_config.n_audio_layer,
            },
            text_decoder_config: TextDecoderConfig {
                n_vocab: cfg.text_decoder_config.n_vocab,
                n_text_ctx: cfg.text_decoder_config.n_text_ctx,
                n_text_state: cfg.text_decoder_config.n_text_state,
                n_text_head: cfg.text_decoder_config.n_text_head,
                n_text_layer: cfg.text_decoder_config.n_text_layer,
            },
        }
    }
}

/// Load a Whisper config from raw JSON bytes.
///
/// Accepts the contents of a `.cfg` file as a byte slice. This is the
/// WASM-compatible path — no filesystem access required.
pub fn load_config_from_bytes(bytes: &[u8]) -> Result<WhisperConfig> {
    let file_config: WhisperModelConfig =
        serde_json::from_slice(bytes).with_context(|| "Failed to parse config JSON from bytes")?;
    Ok(file_config.into())
}

/// Dequantizes INT8 tensor params back to FP32, used inside [`dequantize_weights_to_fp32`].
struct Dequantizer;

impl<B: Backend> ModuleMapper<B> for Dequantizer {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.dequantize(), mapper)
    }
}

/// Load an INT8-quantized model on the CPU (burn-flex), dequantize all weights to FP32,
/// and re-serialize to in-memory bytes. The returned bytes are a plain FP32 NamedMpk
/// that any backend can load without needing INT8 kernel support.
///
/// Used as a fallback by [`load_whisper_from_bytes`] when the target backend reports
/// no native QFloat support (e.g. WGPU/WGSL, which has no i8 element type). Backends
/// that *do* support QFloat (e.g. burn-cuda) skip this indirection and load the
/// quantized weights directly.
fn dequantize_weights_to_fp32(config: &WhisperConfig, bytes: Vec<u8>) -> Result<Vec<u8>> {
    type Cpu = Flex<f32>;
    let device = FlexDevice;
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();

    let cpu_model: Whisper<Cpu> = config.init::<Cpu>(&device);
    let record = recorder
        .load(bytes, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load quantized weights on CPU: {:?}", e))?;
    let cpu_model = cpu_model.load_record(record).map(&mut Dequantizer);

    recorder
        .record(cpu_model.into_record(), ())
        .map_err(|e| anyhow::anyhow!("Failed to re-serialize dequantized weights: {:?}", e))
}

/// Load a Whisper model from in-memory NamedMpk bytes.
///
/// `config` is the parsed model config (from [`load_config_from_bytes`]).
/// `weights` is the raw contents of a `.mpk` file.
/// `precision` is the optional precision from the `.cfg` sidecar; pass `None` when unknown.
///
/// INT8-quantized models are loaded natively on backends that report QFloat support
/// (e.g. burn-cuda → routes quantized matmul through cubecl's INT8 kernels). On
/// backends that don't (WGPU has no INT8 element type in WGSL), the bytes are
/// transparently dequantized to FP32 on the CPU first.
pub fn load_whisper_from_bytes<B: Backend>(
    config: &WhisperConfig,
    weights: Vec<u8>,
    precision: Option<ModelPrecision>,
    device: &B::Device,
) -> Result<Whisper<B>> {
    let weights = match precision {
        Some(ModelPrecision::Int8) => {
            let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme();
            if B::supports_dtype(device, DType::QFloat(scheme)) {
                println!("INT8 model: backend reports QFloat support, loading natively");
                weights
            } else {
                println!("INT8 model: backend has no QFloat support, dequantizing to FP32 first");
                dequantize_weights_to_fp32(config, weights)?
            }
        }
        _ => weights,
    };
    let model = config.init::<B>(device);
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
    let model = model.load_record(
        recorder
            .load(weights, device)
            .map_err(|e| anyhow::anyhow!("Failed to load model weights from bytes: {:?}", e))?,
    );
    Ok(model)
}

/// Load a Whisper model from whisper-burn format (.mpk + .cfg files).
///
/// # Arguments
/// * `model_path` - Path to the model files (without extension); `.cfg` and `.mpk` are appended
/// * `device` - The device to load the model onto
#[cfg(feature = "file-io")]
pub fn load_whisper<B: Backend>(model_path: &str, device: &B::Device) -> Result<Whisper<B>> {
    let model_path = Path::new(model_path);

    let base_path = if model_path.extension().is_some() {
        model_path.with_extension("")
    } else {
        model_path.to_path_buf()
    };

    let config_path = base_path.with_extension("cfg");
    let weights_path = base_path.with_extension("mpk");

    let config_bytes = std::fs::read(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
    let weights_bytes = std::fs::read(&weights_path)
        .with_context(|| format!("Failed to read weights file: {}", weights_path.display()))?;

    let file_config: WhisperModelConfig =
        serde_json::from_slice(&config_bytes).with_context(|| "Failed to parse config JSON")?;
    let precision = file_config.precision;
    let config: WhisperConfig = file_config.into();
    load_whisper_from_bytes(&config, weights_bytes, precision, device)
}

/// Load just the config from a `.cfg` file path.
#[cfg(feature = "file-io")]
pub fn load_config(config_path: &str) -> Result<WhisperConfig> {
    let config_path = Path::new(config_path);

    let config_path = if config_path.extension().map(|e| e == "cfg").unwrap_or(false) {
        config_path.to_path_buf()
    } else {
        config_path.with_extension("cfg")
    };

    let bytes = std::fs::read(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
    load_config_from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn models_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models")
    }

    #[test]
    #[ignore = "slow: initialises base model (6-layer 512-dim) on NdArray CPU (~10 min)"]
    fn test_layer_norm_dims_match_loaded_config() {
        use burn_flex::Flex;
        use burn_flex::FlexDevice;

        let device = FlexDevice;
        // base has n_audio_state=512; tiny_en has 384. The bug hardcoded tiny_en, so a base
        // model would get ln_post with gamma shape [384] instead of [512] and panic on forward.
        let config = WhisperConfig::base();
        assert_ne!(
            config.audio_encoder_config.n_audio_state,
            WhisperConfig::tiny_en().audio_encoder_config.n_audio_state,
            "test precondition: base and tiny_en must have different state dims"
        );

        let mut model = config.init::<Flex<f32>>(&device);
        let fresh = config.init::<Flex<f32>>(&device);
        model.decoder.ln = fresh.decoder.ln;
        model.encoder.ln_post = fresh.encoder.ln_post;

        // Encoder forward would panic if ln_post had wrong dims (384 vs expected 512).
        let mel = burn::tensor::Tensor::<Flex<f32>, 3>::zeros([1, 80, 3000], &device);
        let out = model.forward_encoder(mel);
        assert_eq!(out.dims(), [1, 1500, 512]);
    }

    #[test]
    fn test_load_config() {
        let config_path = models_dir().join("tiny_en.cfg");
        if !config_path.exists() {
            eprintln!("Skipping test: model files not found at {:?}", config_path);
            return;
        }

        let config = load_config(config_path.to_str().unwrap()).unwrap();

        assert_eq!(config.audio_encoder_config.n_mels, 80);
        assert_eq!(config.audio_encoder_config.n_audio_state, 384);
        assert_eq!(config.audio_encoder_config.n_audio_head, 6);
        assert_eq!(config.audio_encoder_config.n_audio_layer, 4);
        assert_eq!(config.text_decoder_config.n_vocab, 51864);
        assert_eq!(config.text_decoder_config.n_text_state, 384);
    }

    #[test]
    fn test_load_whisper_model() {
        use burn_flex::Flex;
        use burn_flex::FlexDevice;

        // Use converted model (Burn 0.20 format)
        let model_path = models_dir().join("tiny_en_converted");
        if !model_path.with_extension("mpk").exists() {
            eprintln!("Skipping test: model files not found at {:?}", model_path);
            eprintln!(
                "Run `cargo test -p whisperforge test_convert_tiny_en` first to generate the model"
            );
            return;
        }

        let device = FlexDevice;
        let model = load_whisper::<Flex<f32>>(model_path.to_str().unwrap(), &device);

        match model {
            Ok(m) => {
                assert_eq!(m.encoder.n_mels, 80);
                assert_eq!(m.decoder.n_vocab, 51864);
                println!("Model loaded successfully!");
            }
            Err(e) => {
                // Print detailed error for debugging
                panic!("Failed to load model: {:?}", e);
            }
        }
    }

    /// Load base weights (6-layer, n_audio_state=512) and verify encoder output shape.
    ///
    /// Run locally with model files present:
    /// `cargo test --release -p whisperforge-core -- --ignored test_load_base_model_and_encoder_forward --nocapture`
    ///
    /// medium and large-v2 are excluded from CI (too large for automated test infra).
    #[test]
    #[ignore = "requires models/base_converted.{mpk,cfg} — git-ignored; convert from HuggingFace first"]
    fn test_load_base_model_and_encoder_forward() -> Result<()> {
        use burn_flex::Flex;
        use burn_flex::FlexDevice;

        let model_path = models_dir().join("base_converted");
        if !model_path.with_extension("mpk").exists() {
            eprintln!(
                "Skipping: {:?}.mpk not found. Convert from HuggingFace first.",
                model_path
            );
            return Ok(());
        }

        let device = FlexDevice;
        let m = load_whisper::<Flex<f32>>(model_path.to_str().unwrap(), &device)?;
        assert_eq!(m.encoder.n_mels, 80);

        let mel = burn::tensor::Tensor::<Flex<f32>, 3>::zeros([1, 80, 3000], &device);
        let out = m.forward_encoder(mel);
        // base: n_audio_ctx=1500, n_audio_state=512
        assert_eq!(out.dims(), [1, 1500, 512]);

        Ok(())
    }

    /// Load small weights (12-layer, n_audio_state=768) and verify encoder output shape.
    ///
    /// Run locally with model files present:
    /// `cargo test --release -p whisperforge-core -- --ignored test_load_small_model_and_encoder_forward --nocapture`
    ///
    /// medium and large-v2 are excluded from CI (too large for automated test infra).
    #[test]
    #[ignore = "requires models/small_converted.{mpk,cfg} — git-ignored; convert from HuggingFace first"]
    fn test_load_small_model_and_encoder_forward() -> Result<()> {
        use burn_flex::Flex;
        use burn_flex::FlexDevice;

        let model_path = models_dir().join("small_converted");
        if !model_path.with_extension("mpk").exists() {
            eprintln!(
                "Skipping: {:?}.mpk not found. Convert from HuggingFace first.",
                model_path
            );
            return Ok(());
        }

        let device = FlexDevice;
        let m = load_whisper::<Flex<f32>>(model_path.to_str().unwrap(), &device)?;
        assert_eq!(m.encoder.n_mels, 80);

        let mel = burn::tensor::Tensor::<Flex<f32>, 3>::zeros([1, 80, 3000], &device);
        let out = m.forward_encoder(mel);
        // small: n_audio_ctx=1500, n_audio_state=768
        assert_eq!(out.dims(), [1, 1500, 768]);

        Ok(())
    }
}
