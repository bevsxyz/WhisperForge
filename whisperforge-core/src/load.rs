// Model loading utilities for Whisper models saved in MessagePack format
// Compatible with whisper-burn pre-converted models from HuggingFace

use std::path::Path;

use anyhow::{Context, Result};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use serde::Deserialize;

use crate::model::{AudioEncoderConfig, TextDecoderConfig, Whisper, WhisperConfig};

/// Configuration file format for whisper-burn models (.cfg files)
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperModelConfig {
    pub audio_encoder_config: AudioEncoderConfigFile,
    pub text_decoder_config: TextDecoderConfigFile,
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

/// Load a Whisper model from whisper-burn format (.mpk + .cfg files)
///
/// # Arguments
/// * `model_path` - Path to the .mpk model file (without extension)
/// * `device` - The device to load the model onto
///
/// # Returns
/// The loaded Whisper model
///
/// # Example
/// ```ignore
/// use whisperforge_core::load::load_whisper;
/// use burn::backend::NdArray;
/// use burn_ndarray::NdArrayDevice;
///
/// let device = NdArrayDevice::default();
/// let model = load_whisper::<NdArray<f32>>("models/tiny_en", &device)?;
/// ```
pub fn load_whisper<B: Backend>(model_path: &str, device: &B::Device) -> Result<Whisper<B>> {
    let model_path = Path::new(model_path);

    // Determine paths - handle both with and without extension
    let base_path = if model_path.extension().is_some() {
        model_path.with_extension("")
    } else {
        model_path.to_path_buf()
    };

    let config_path = base_path.with_extension("cfg");
    let weights_path = base_path.with_extension("mpk");

    // Load config
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let file_config: WhisperModelConfig = serde_json::from_str(&config_str)
        .with_context(|| format!("Failed to parse config JSON: {}", config_path.display()))?;

    let config: WhisperConfig = file_config.into();

    // Initialize model with random weights
    let model = config.init::<B>(device);

    // Load weights from .mpk file
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    // Remove .mpk extension for load_file (it adds it automatically)
    let weights_path_str = base_path.to_str().context("Invalid model path encoding")?;

    let model = model
        .load_file(weights_path_str, &recorder, device)
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to load model weights from {}: {:?}",
                weights_path.display(),
                e
            )
        })?;

    Ok(model)
}

/// Load just the config without loading model weights
pub fn load_config(config_path: &str) -> Result<WhisperConfig> {
    let config_path = Path::new(config_path);

    // Handle .cfg or base path
    let config_path = if config_path.extension().map(|e| e == "cfg").unwrap_or(false) {
        config_path.to_path_buf()
    } else {
        config_path.with_extension("cfg")
    };

    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let file_config: WhisperModelConfig = serde_json::from_str(&config_str)
        .with_context(|| format!("Failed to parse config JSON: {}", config_path.display()))?;

    Ok(file_config.into())
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
        use burn::backend::NdArray;
        use burn_ndarray::NdArrayDevice;

        // Use converted model (Burn 0.20 format)
        let model_path = models_dir().join("tiny_en_converted");
        if !model_path.with_extension("mpk").exists() {
            eprintln!("Skipping test: model files not found at {:?}", model_path);
            eprintln!(
                "Run `cargo test -p whisperforge-convert test_convert_tiny_en` first to generate the model"
            );
            return;
        }

        let device = NdArrayDevice::default();
        let model = load_whisper::<NdArray<f32>>(model_path.to_str().unwrap(), &device);

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
}
