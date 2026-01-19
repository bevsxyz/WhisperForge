// Convert OpenAI Whisper models to Burn 0.20 format
//
// This module handles loading safetensors files and converting them
// to our Whisper model structure.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{backend::Backend, Tensor, TensorData},
};
use safetensors::SafeTensors;

use whisperforge_core::{Whisper, WhisperConfig};

/// Convert an OpenAI Whisper model to Burn format
///
/// # Arguments
/// * `input_path` - Path to the OpenAI safetensors file
/// * `output_path` - Path to save the converted model (without extension)
/// * `device` - Device to use for conversion
pub fn convert_openai_to_burn<B: Backend>(
    input_path: &Path,
    output_path: &Path,
    device: &B::Device,
) -> Result<()> {
    // Load safetensors
    let data = std::fs::read(input_path)
        .with_context(|| format!("Failed to read {}", input_path.display()))?;
    let tensors = SafeTensors::deserialize(&data).context("Failed to parse safetensors file")?;

    // Detect model size from tensor shapes
    let config = detect_config(&tensors)?;
    println!("Detected config: {:?}", config);

    // Initialize model with random weights
    let model = config.init::<B>(device);

    // Load weights from safetensors into model
    let model = load_weights_into_model(model, &tensors, device)?;

    // Save in Burn format
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(output_path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model: {:?}", e))?;

    // Save config
    let config_path = output_path.with_extension("cfg");
    let config_json = serde_json::to_string_pretty(&WhisperConfigFile::from(&config))?;
    std::fs::write(&config_path, config_json)?;

    println!("Model saved to {}", output_path.display());
    println!("Config saved to {}", config_path.display());

    Ok(())
}

/// Detect model configuration from tensor shapes
fn detect_config(tensors: &SafeTensors) -> Result<WhisperConfig> {
    // HuggingFace models have "model." prefix, handle both formats
    let prefix = if tensors.tensor("model.encoder.conv1.weight").is_ok() {
        "model."
    } else {
        ""
    };

    // Get encoder embedding dimension from conv1 weight shape
    // encoder.conv1.weight has shape [n_audio_state, n_mels, 3]
    let conv1_tensor = tensors
        .tensor(&format!("{}encoder.conv1.weight", prefix))
        .context("Missing encoder.conv1.weight")?;
    let conv1_shape = conv1_tensor.shape();

    let n_audio_state = conv1_shape[0];
    let n_mels = conv1_shape[1];

    // Get vocab size from decoder embed_tokens
    // decoder.embed_tokens.weight has shape [n_vocab, n_text_state]
    let embed_tensor = tensors
        .tensor(&format!("{}decoder.embed_tokens.weight", prefix))
        .context("Missing decoder.embed_tokens.weight")?;
    let embed_shape = embed_tensor.shape();

    let n_vocab = embed_shape[0];
    let n_text_state = embed_shape[1];

    // Count encoder layers
    let mut n_audio_layer = 0;
    while tensors
        .tensor(&format!(
            "{}encoder.layers.{}.self_attn.q_proj.weight",
            prefix, n_audio_layer
        ))
        .is_ok()
    {
        n_audio_layer += 1;
    }

    // Count decoder layers
    let mut n_text_layer = 0;
    while tensors
        .tensor(&format!(
            "{}decoder.layers.{}.self_attn.q_proj.weight",
            prefix, n_text_layer
        ))
        .is_ok()
    {
        n_text_layer += 1;
    }

    // Get number of heads from q_proj weight shape
    // q_proj.weight has shape [n_state, n_state], heads = n_state / head_dim
    // For Whisper, head_dim is typically 64
    let n_audio_head = n_audio_state / 64;
    let n_text_head = n_text_state / 64;

    // Get context sizes from positional embeddings
    let encoder_pos_tensor = tensors
        .tensor(&format!("{}encoder.embed_positions.weight", prefix))
        .context("Missing encoder.embed_positions.weight")?;
    let encoder_pos_shape = encoder_pos_tensor.shape();
    let n_audio_ctx = encoder_pos_shape[0];

    let decoder_pos_tensor = tensors
        .tensor(&format!("{}decoder.embed_positions.weight", prefix))
        .context("Missing decoder.embed_positions.weight")?;
    let decoder_pos_shape = decoder_pos_tensor.shape();
    let n_text_ctx = decoder_pos_shape[0];

    Ok(WhisperConfig {
        audio_encoder_config: whisperforge_core::AudioEncoderConfig {
            n_mels,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
        },
        text_decoder_config: whisperforge_core::TextDecoderConfig {
            n_vocab,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
        },
    })
}

/// Load safetensor weights into our model structure
fn load_weights_into_model<B: Backend>(
    model: Whisper<B>,
    tensors: &SafeTensors,
    device: &B::Device,
) -> Result<Whisper<B>> {
    // Map OpenAI layer names to our model structure
    // OpenAI: encoder.conv1.weight -> Our: encoder.conv1.weight
    // OpenAI: encoder.layers.0.self_attn.q_proj.weight -> Our: encoder.blocks.0.attn.query.weight

    let tensor_map = build_tensor_map(tensors)?;

    // Load encoder
    let encoder = load_encoder(&model.encoder, &tensor_map, device)?;

    // Load decoder
    let decoder = load_decoder(&model.decoder, &tensor_map, device)?;

    Ok(Whisper { encoder, decoder })
}

/// Build a map of tensor name -> raw tensor data
/// Strips "model." prefix if present (HuggingFace format)
fn build_tensor_map(tensors: &SafeTensors) -> Result<HashMap<String, RawTensorData>> {
    let mut map = HashMap::new();

    for (name, tensor) in tensors.tensors() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let data = tensor.data();

        // Convert f16 to f32 if needed
        let f32_data: Vec<f32> = match tensor.dtype() {
            safetensors::Dtype::F16 => data
                .chunks(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F32 => data
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::BF16 => data
                .chunks(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            other => anyhow::bail!("Unsupported dtype: {:?}", other),
        };

        // Strip "model." prefix if present (HuggingFace format)
        let key = name.strip_prefix("model.").unwrap_or(&name).to_string();

        map.insert(
            key,
            RawTensorData {
                shape,
                data: f32_data,
            },
        );
    }

    Ok(map)
}

struct RawTensorData {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl RawTensorData {
    fn to_tensor<B: Backend, const D: usize>(&self, device: &B::Device) -> Tensor<B, D> {
        let shape_arr: [usize; D] = self
            .shape
            .clone()
            .try_into()
            .unwrap_or_else(|_| panic!("Shape {:?} doesn't match dimension {}", self.shape, D));
        let tensor_data = TensorData::new(self.data.clone(), shape_arr);
        Tensor::from_data(tensor_data, device)
    }
}

use burn::module::Param;
use whisperforge_core::model::{AudioEncoder, TextDecoder};

fn load_encoder<B: Backend>(
    encoder: &AudioEncoder<B>,
    tensors: &HashMap<String, RawTensorData>,
    device: &B::Device,
) -> Result<AudioEncoder<B>> {
    // Clone the encoder structure and load weights
    let mut encoder = encoder.clone();

    // Load conv1
    let conv1_weight = tensors
        .get("encoder.conv1.weight")
        .context("Missing encoder.conv1.weight")?;
    let conv1_bias = tensors
        .get("encoder.conv1.bias")
        .context("Missing encoder.conv1.bias")?;
    encoder.conv1 = load_conv1d(&encoder.conv1, conv1_weight, conv1_bias, device);

    // Load conv2
    let conv2_weight = tensors
        .get("encoder.conv2.weight")
        .context("Missing encoder.conv2.weight")?;
    let conv2_bias = tensors
        .get("encoder.conv2.bias")
        .context("Missing encoder.conv2.bias")?;
    encoder.conv2 = load_conv1d(&encoder.conv2, conv2_weight, conv2_bias, device);

    // Load positional embedding
    let pos_emb = tensors
        .get("encoder.embed_positions.weight")
        .context("Missing encoder.embed_positions.weight")?;
    encoder.positional_embedding = Param::from_tensor(pos_emb.to_tensor(device));

    // Load encoder blocks
    for (i, block) in encoder.blocks.iter_mut().enumerate() {
        // Self attention
        let prefix = format!("encoder.layers.{}", i);

        // attn.query
        let q_weight = tensors
            .get(&format!("{}.self_attn.q_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.q_proj.weight", prefix))?;
        let q_bias = tensors
            .get(&format!("{}.self_attn.q_proj.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn.q_proj.bias", prefix))?;
        block.attn.query = load_linear(&block.attn.query, q_weight, Some(q_bias), device);

        // attn.key (no bias in Whisper)
        let k_weight = tensors
            .get(&format!("{}.self_attn.k_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.k_proj.weight", prefix))?;
        block.attn.key = load_linear_no_bias(&block.attn.key, k_weight, device);

        // attn.value
        let v_weight = tensors
            .get(&format!("{}.self_attn.v_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.v_proj.weight", prefix))?;
        let v_bias = tensors
            .get(&format!("{}.self_attn.v_proj.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn.v_proj.bias", prefix))?;
        block.attn.value = load_linear(&block.attn.value, v_weight, Some(v_bias), device);

        // attn.out
        let out_weight = tensors
            .get(&format!("{}.self_attn.out_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.out_proj.weight", prefix))?;
        let out_bias = tensors
            .get(&format!("{}.self_attn.out_proj.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn.out_proj.bias", prefix))?;
        block.attn.out = load_linear(&block.attn.out, out_weight, Some(out_bias), device);

        // attn_ln
        let ln_weight = tensors
            .get(&format!("{}.self_attn_layer_norm.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn_layer_norm.weight", prefix))?;
        let ln_bias = tensors
            .get(&format!("{}.self_attn_layer_norm.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn_layer_norm.bias", prefix))?;
        block.attn_ln = load_layer_norm(&block.attn_ln, ln_weight, ln_bias, device);

        // mlp.lin1 (fc1)
        let fc1_weight = tensors
            .get(&format!("{}.fc1.weight", prefix))
            .with_context(|| format!("Missing {}.fc1.weight", prefix))?;
        let fc1_bias = tensors
            .get(&format!("{}.fc1.bias", prefix))
            .with_context(|| format!("Missing {}.fc1.bias", prefix))?;
        block.mlp.lin1 = load_linear(&block.mlp.lin1, fc1_weight, Some(fc1_bias), device);

        // mlp.lin2 (fc2)
        let fc2_weight = tensors
            .get(&format!("{}.fc2.weight", prefix))
            .with_context(|| format!("Missing {}.fc2.weight", prefix))?;
        let fc2_bias = tensors
            .get(&format!("{}.fc2.bias", prefix))
            .with_context(|| format!("Missing {}.fc2.bias", prefix))?;
        block.mlp.lin2 = load_linear(&block.mlp.lin2, fc2_weight, Some(fc2_bias), device);

        // mlp_ln (final_layer_norm)
        let mlp_ln_weight = tensors
            .get(&format!("{}.final_layer_norm.weight", prefix))
            .with_context(|| format!("Missing {}.final_layer_norm.weight", prefix))?;
        let mlp_ln_bias = tensors
            .get(&format!("{}.final_layer_norm.bias", prefix))
            .with_context(|| format!("Missing {}.final_layer_norm.bias", prefix))?;
        block.mlp_ln = load_layer_norm(&block.mlp_ln, mlp_ln_weight, mlp_ln_bias, device);
    }

    // Load ln_post (encoder.layer_norm)
    let ln_post_weight = tensors
        .get("encoder.layer_norm.weight")
        .context("Missing encoder.layer_norm.weight")?;
    let ln_post_bias = tensors
        .get("encoder.layer_norm.bias")
        .context("Missing encoder.layer_norm.bias")?;
    encoder.ln_post = load_layer_norm(&encoder.ln_post, ln_post_weight, ln_post_bias, device);

    Ok(encoder)
}

fn load_decoder<B: Backend>(
    decoder: &TextDecoder<B>,
    tensors: &HashMap<String, RawTensorData>,
    device: &B::Device,
) -> Result<TextDecoder<B>> {
    let mut decoder = decoder.clone();

    // Load token embedding
    let token_emb = tensors
        .get("decoder.embed_tokens.weight")
        .context("Missing decoder.embed_tokens.weight")?;
    decoder.token_embedding = Param::from_tensor(token_emb.to_tensor(device));

    // Load positional embedding
    let pos_emb = tensors
        .get("decoder.embed_positions.weight")
        .context("Missing decoder.embed_positions.weight")?;
    decoder.positional_embedding = Param::from_tensor(pos_emb.to_tensor(device));

    // Load decoder blocks
    for (i, block) in decoder.blocks.iter_mut().enumerate() {
        let prefix = format!("decoder.layers.{}", i);

        // Self attention
        let q_weight = tensors
            .get(&format!("{}.self_attn.q_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.q_proj.weight", prefix))?;
        let q_bias = tensors
            .get(&format!("{}.self_attn.q_proj.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn.q_proj.bias", prefix))?;
        block.attn.query = load_linear(&block.attn.query, q_weight, Some(q_bias), device);

        let k_weight = tensors
            .get(&format!("{}.self_attn.k_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.k_proj.weight", prefix))?;
        block.attn.key = load_linear_no_bias(&block.attn.key, k_weight, device);

        let v_weight = tensors
            .get(&format!("{}.self_attn.v_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.v_proj.weight", prefix))?;
        let v_bias = tensors
            .get(&format!("{}.self_attn.v_proj.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn.v_proj.bias", prefix))?;
        block.attn.value = load_linear(&block.attn.value, v_weight, Some(v_bias), device);

        let out_weight = tensors
            .get(&format!("{}.self_attn.out_proj.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn.out_proj.weight", prefix))?;
        let out_bias = tensors
            .get(&format!("{}.self_attn.out_proj.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn.out_proj.bias", prefix))?;
        block.attn.out = load_linear(&block.attn.out, out_weight, Some(out_bias), device);

        // attn_ln (self_attn_layer_norm)
        let ln_weight = tensors
            .get(&format!("{}.self_attn_layer_norm.weight", prefix))
            .with_context(|| format!("Missing {}.self_attn_layer_norm.weight", prefix))?;
        let ln_bias = tensors
            .get(&format!("{}.self_attn_layer_norm.bias", prefix))
            .with_context(|| format!("Missing {}.self_attn_layer_norm.bias", prefix))?;
        block.attn_ln = load_layer_norm(&block.attn_ln, ln_weight, ln_bias, device);

        // Cross attention (encoder_attn)
        let cross_q_weight = tensors
            .get(&format!("{}.encoder_attn.q_proj.weight", prefix))
            .with_context(|| format!("Missing {}.encoder_attn.q_proj.weight", prefix))?;
        let cross_q_bias = tensors
            .get(&format!("{}.encoder_attn.q_proj.bias", prefix))
            .with_context(|| format!("Missing {}.encoder_attn.q_proj.bias", prefix))?;
        block.cross_attn.query = load_linear(
            &block.cross_attn.query,
            cross_q_weight,
            Some(cross_q_bias),
            device,
        );

        let cross_k_weight = tensors
            .get(&format!("{}.encoder_attn.k_proj.weight", prefix))
            .with_context(|| format!("Missing {}.encoder_attn.k_proj.weight", prefix))?;
        block.cross_attn.key = load_linear_no_bias(&block.cross_attn.key, cross_k_weight, device);

        let cross_v_weight = tensors
            .get(&format!("{}.encoder_attn.v_proj.weight", prefix))
            .with_context(|| format!("Missing {}.encoder_attn.v_proj.weight", prefix))?;
        let cross_v_bias = tensors
            .get(&format!("{}.encoder_attn.v_proj.bias", prefix))
            .with_context(|| format!("Missing {}.encoder_attn.v_proj.bias", prefix))?;
        block.cross_attn.value = load_linear(
            &block.cross_attn.value,
            cross_v_weight,
            Some(cross_v_bias),
            device,
        );

        let cross_out_weight = tensors
            .get(&format!("{}.encoder_attn.out_proj.weight", prefix))
            .with_context(|| format!("Missing {}.encoder_attn.out_proj.weight", prefix))?;
        let cross_out_bias = tensors
            .get(&format!("{}.encoder_attn.out_proj.bias", prefix))
            .with_context(|| format!("Missing {}.encoder_attn.out_proj.bias", prefix))?;
        block.cross_attn.out = load_linear(
            &block.cross_attn.out,
            cross_out_weight,
            Some(cross_out_bias),
            device,
        );

        // cross_attn_ln (encoder_attn_layer_norm)
        let cross_ln_weight = tensors
            .get(&format!("{}.encoder_attn_layer_norm.weight", prefix))
            .with_context(|| format!("Missing {}.encoder_attn_layer_norm.weight", prefix))?;
        let cross_ln_bias = tensors
            .get(&format!("{}.encoder_attn_layer_norm.bias", prefix))
            .with_context(|| format!("Missing {}.encoder_attn_layer_norm.bias", prefix))?;
        block.cross_attn_ln =
            load_layer_norm(&block.cross_attn_ln, cross_ln_weight, cross_ln_bias, device);

        // mlp (fc1, fc2)
        let fc1_weight = tensors
            .get(&format!("{}.fc1.weight", prefix))
            .with_context(|| format!("Missing {}.fc1.weight", prefix))?;
        let fc1_bias = tensors
            .get(&format!("{}.fc1.bias", prefix))
            .with_context(|| format!("Missing {}.fc1.bias", prefix))?;
        block.mlp.lin1 = load_linear(&block.mlp.lin1, fc1_weight, Some(fc1_bias), device);

        let fc2_weight = tensors
            .get(&format!("{}.fc2.weight", prefix))
            .with_context(|| format!("Missing {}.fc2.weight", prefix))?;
        let fc2_bias = tensors
            .get(&format!("{}.fc2.bias", prefix))
            .with_context(|| format!("Missing {}.fc2.bias", prefix))?;
        block.mlp.lin2 = load_linear(&block.mlp.lin2, fc2_weight, Some(fc2_bias), device);

        // mlp_ln (final_layer_norm)
        let mlp_ln_weight = tensors
            .get(&format!("{}.final_layer_norm.weight", prefix))
            .with_context(|| format!("Missing {}.final_layer_norm.weight", prefix))?;
        let mlp_ln_bias = tensors
            .get(&format!("{}.final_layer_norm.bias", prefix))
            .with_context(|| format!("Missing {}.final_layer_norm.bias", prefix))?;
        block.mlp_ln = load_layer_norm(&block.mlp_ln, mlp_ln_weight, mlp_ln_bias, device);
    }

    // Load ln (decoder.layer_norm)
    let ln_weight = tensors
        .get("decoder.layer_norm.weight")
        .context("Missing decoder.layer_norm.weight")?;
    let ln_bias = tensors
        .get("decoder.layer_norm.bias")
        .context("Missing decoder.layer_norm.bias")?;
    decoder.ln = load_layer_norm(&decoder.ln, ln_weight, ln_bias, device);

    Ok(decoder)
}

// Helper functions to load individual layer types

fn load_conv1d<B: Backend>(
    conv: &burn::nn::conv::Conv1d<B>,
    weight: &RawTensorData,
    bias: &RawTensorData,
    device: &B::Device,
) -> burn::nn::conv::Conv1d<B> {
    let mut conv = conv.clone();
    conv.weight = Param::from_tensor(weight.to_tensor(device));
    conv.bias = Some(Param::from_tensor(bias.to_tensor(device)));
    conv
}

fn load_linear<B: Backend>(
    linear: &burn::nn::Linear<B>,
    weight: &RawTensorData,
    bias: Option<&RawTensorData>,
    device: &B::Device,
) -> burn::nn::Linear<B> {
    let mut linear = linear.clone();
    // PyTorch stores weights as [out_features, in_features]
    // Burn expects [in_features, out_features] for input.matmul(weight)
    let weight_tensor: Tensor<B, 2> = weight.to_tensor(device);
    linear.weight = Param::from_tensor(weight_tensor.transpose());
    if let Some(b) = bias {
        linear.bias = Some(Param::from_tensor(b.to_tensor(device)));
    }
    linear
}

fn load_linear_no_bias<B: Backend>(
    linear: &burn::nn::Linear<B>,
    weight: &RawTensorData,
    device: &B::Device,
) -> burn::nn::Linear<B> {
    let mut linear = linear.clone();
    // PyTorch stores weights as [out_features, in_features]
    // Burn expects [in_features, out_features] for input.matmul(weight)
    let weight_tensor: Tensor<B, 2> = weight.to_tensor(device);
    linear.weight = Param::from_tensor(weight_tensor.transpose());
    linear.bias = None;
    linear
}

fn load_layer_norm<B: Backend>(
    ln: &burn::nn::LayerNorm<B>,
    weight: &RawTensorData,
    bias: &RawTensorData,
    device: &B::Device,
) -> burn::nn::LayerNorm<B> {
    let mut ln = ln.clone();
    ln.gamma = Param::from_tensor(weight.to_tensor(device));
    ln.beta = Some(Param::from_tensor(bias.to_tensor(device)));
    ln
}

// Config file format for saving
#[derive(serde::Serialize)]
struct WhisperConfigFile {
    audio_encoder_config: AudioEncoderConfigFile,
    text_decoder_config: TextDecoderConfigFile,
}

#[derive(serde::Serialize)]
struct AudioEncoderConfigFile {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
}

#[derive(serde::Serialize)]
struct TextDecoderConfigFile {
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
}

impl From<&WhisperConfig> for WhisperConfigFile {
    fn from(config: &WhisperConfig) -> Self {
        WhisperConfigFile {
            audio_encoder_config: AudioEncoderConfigFile {
                n_mels: config.audio_encoder_config.n_mels,
                n_audio_ctx: config.audio_encoder_config.n_audio_ctx,
                n_audio_state: config.audio_encoder_config.n_audio_state,
                n_audio_head: config.audio_encoder_config.n_audio_head,
                n_audio_layer: config.audio_encoder_config.n_audio_layer,
            },
            text_decoder_config: TextDecoderConfigFile {
                n_vocab: config.text_decoder_config.n_vocab,
                n_text_ctx: config.text_decoder_config.n_text_ctx,
                n_text_state: config.text_decoder_config.n_text_state,
                n_text_head: config.text_decoder_config.n_text_head,
                n_text_layer: config.text_decoder_config.n_text_layer,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn_ndarray::NdArrayDevice;
    use std::path::PathBuf;

    fn models_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models")
    }

    #[test]
    fn test_detect_config() {
        let model_path = models_dir().join("tiny_en_openai.safetensors");
        if !model_path.exists() {
            eprintln!("Skipping: model not found at {:?}", model_path);
            return;
        }

        let data = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&data).unwrap();

        let config = detect_config(&tensors).unwrap();

        println!("Detected config:");
        println!("  n_mels: {}", config.audio_encoder_config.n_mels);
        println!(
            "  n_audio_state: {}",
            config.audio_encoder_config.n_audio_state
        );
        println!(
            "  n_audio_layer: {}",
            config.audio_encoder_config.n_audio_layer
        );
        println!(
            "  n_audio_head: {}",
            config.audio_encoder_config.n_audio_head
        );
        println!("  n_vocab: {}", config.text_decoder_config.n_vocab);
        println!(
            "  n_text_state: {}",
            config.text_decoder_config.n_text_state
        );
        println!(
            "  n_text_layer: {}",
            config.text_decoder_config.n_text_layer
        );

        // tiny.en expected values
        assert_eq!(config.audio_encoder_config.n_mels, 80);
        assert_eq!(config.audio_encoder_config.n_audio_state, 384);
        assert_eq!(config.audio_encoder_config.n_audio_layer, 4);
        assert_eq!(config.text_decoder_config.n_text_layer, 4);
    }

    #[test]
    fn test_convert_tiny_en() {
        let input_path = models_dir().join("tiny_en_openai.safetensors");
        let output_path = models_dir().join("tiny_en_converted");

        if !input_path.exists() {
            eprintln!("Skipping: model not found at {:?}", input_path);
            return;
        }

        let device = NdArrayDevice::default();

        let result = convert_openai_to_burn::<NdArray<f32>>(&input_path, &output_path, &device);

        match result {
            Ok(()) => println!("Conversion successful!"),
            Err(e) => panic!("Conversion failed: {:?}", e),
        }
    }
}
