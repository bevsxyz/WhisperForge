use anyhow::Result;
use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        transformer::{
            TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput,
            TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
        },
        Embedding, EmbeddingConfig, LayerNorm, PaddingConfig1d,
    },
    tensor::{backend::Backend, Bool, Int, Tensor},
};

#[derive(Config, Debug)]
pub struct WhisperConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
    pub n_vocab: usize,
}

impl WhisperConfig {
    pub fn tiny_en() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_vocab: 51864,
        }
    }

    pub fn base() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 512,
            n_audio_head: 8,
            n_audio_layer: 6,
            n_text_ctx: 448,
            n_text_state: 512,
            n_text_head: 8,
            n_text_layer: 6,
            n_vocab: 51864,
        }
    }

    pub fn small() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 768,
            n_audio_head: 12,
            n_audio_layer: 12,
            n_text_ctx: 448,
            n_text_state: 768,
            n_text_head: 12,
            n_text_layer: 12,
            n_vocab: 51864,
        }
    }

    pub fn medium() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 1024,
            n_audio_head: 16,
            n_audio_layer: 24,
            n_text_ctx: 448,
            n_text_state: 1024,
            n_text_head: 16,
            n_text_layer: 24,
            n_vocab: 51864,
        }
    }

    pub fn large_v2() -> Self {
        Self {
            n_mels: 128,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
            n_vocab: 51864,
        }
    }

    pub fn large_v3() -> Self {
        Self {
            n_mels: 128,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
            n_vocab: 51865, // One extra token for multilingual
        }
    }
}

#[derive(Module, Debug)]
pub struct WhisperEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub conv2: Conv1d<B>,
    pub pos_emb: Embedding<B>,
    pub blocks: Vec<TransformerEncoder<B>>,
    pub ln_post: LayerNorm<B>,
}

impl<B: Backend> WhisperEncoder<B> {
    pub fn new(config: &WhisperConfig, device: &B::Device) -> Self {
        let conv1_config = Conv1dConfig::new(config.n_mels, config.n_audio_state, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_bias(true);
        let conv1 = conv1_config.init(device);

        let conv2_config = Conv1dConfig::new(config.n_audio_state, config.n_audio_state, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_bias(true);
        let conv2 = conv2_config.init(device);

        let pos_emb_config = EmbeddingConfig::new(config.n_audio_ctx, config.n_audio_state);
        let pos_emb = pos_emb_config.init(device);

        let mut blocks = Vec::with_capacity(config.n_audio_layer);
        for _ in 0..config.n_audio_layer {
            let block_config = TransformerEncoderConfig::new(
                config.n_audio_state,
                config.n_audio_head,
                config.n_audio_state / config.n_audio_head,
                config.n_audio_state * 4,
            )
            .with_norm_first(true);
            blocks.push(block_config.init(device));
        }

        let ln_post_config = burn::nn::LayerNormConfig::new(config.n_audio_state);
        let ln_post = ln_post_config.init(device);

        Self {
            conv1,
            conv2,
            pos_emb,
            blocks,
            ln_post,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let x = x.swap_dims(1, 2);
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::relu(x);

        let x = x.swap_dims(1, 2);

        // Add positional embeddings
        let pos_ids = Tensor::arange(0..seq_len as i64, &x.device())
            .reshape([1, seq_len])
            .repeat(&[batch_size, 1]);
        let x = x + self.pos_emb.forward(pos_ids);

        // Pass through transformer blocks
        let mut hidden_states = x;
        for block in &self.blocks {
            let mut input = TransformerEncoderInput::new(hidden_states);
            if let Some(ref m) = mask {
                input = input.mask_pad(m.clone());
            }
            hidden_states = block.forward(input);
        }

        self.ln_post.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct WhisperDecoder<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub pos_emb: Embedding<B>,
    pub blocks: Vec<TransformerDecoder<B>>,
    pub ln: LayerNorm<B>,
}

impl<B: Backend> WhisperDecoder<B> {
    pub fn new(config: &WhisperConfig, device: &B::Device) -> Self {
        let token_emb_config = EmbeddingConfig::new(config.n_vocab, config.n_text_state);
        let token_embedding = token_emb_config.init(device);

        let pos_emb_config = EmbeddingConfig::new(config.n_text_ctx, config.n_text_state);
        let pos_emb = pos_emb_config.init(device);

        let mut blocks = Vec::with_capacity(config.n_text_layer);
        for _ in 0..config.n_text_layer {
            let block_config = TransformerDecoderConfig::new(
                config.n_text_state,
                config.n_text_head,
                config.n_text_state / config.n_text_head,
                config.n_text_state * 4,
            )
            .with_norm_first(true);
            blocks.push(block_config.init(device));
        }

        let ln_config = burn::nn::LayerNormConfig::new(config.n_text_state);
        let ln = ln_config.init(device);

        Self {
            token_embedding,
            pos_emb,
            blocks,
            ln,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = x.dims();

        let x = self.token_embedding.forward(x);
        let pos_ids = Tensor::arange(0..seq_len as i64, &x.device())
            .reshape([1, seq_len])
            .repeat(&[batch_size, 1]);
        let x = x + self.pos_emb.forward(pos_ids);

        let mut hidden_states = x;

        for block in &self.blocks {
            // Self-attention input (Query = Key = Value = hidden_states)
            // Cross-attention input (Key = Value = encoder_output)

            // In Burn 0.13/0.14+, TransformerDecoderInput::new takes (target, memory).
            // It automatically handles creating MhaInput internally or expects raw tensors.
            // Based on error: "expected Tensor, found MhaInput".

            let mut input =
                TransformerDecoderInput::new(hidden_states.clone(), encoder_output.clone());

            if let Some(ref m) = mask {
                input = input.memory_mask_pad(m.clone());
            }

            hidden_states = block.forward(input);
        }

        self.ln.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct WhisperModel<B: Backend> {
    pub encoder: WhisperEncoder<B>,
    pub decoder: WhisperDecoder<B>,
    pub config: Ignored<WhisperConfig>,
}

impl<B: Backend> WhisperModel<B> {
    pub fn new(config: WhisperConfig, device: &B::Device) -> Self {
        let encoder = WhisperEncoder::new(&config, device);
        let decoder = WhisperDecoder::new(&config, device);

        Self {
            encoder,
            decoder,
            config: Ignored(config),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let encoder_output = self.encoder.forward(x, None);

        let seq_len = tokens.dims()[1];
        let decoder_input = Tensor::<B, 2, Int>::zeros([1, seq_len], &tokens.device());
        self.decoder.forward(decoder_input, encoder_output, None)
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config.0
    }

    pub fn forward_decoder_logits(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let hidden = self.decoder.forward(tokens, encoder_output, None);
        // Project to logits: hidden @ embedding.weight.T
        // hidden: [batch, seq, state] (rank 3)
        // embedding.weight: [vocab, state]
        // transpose: [state, vocab]
        // Need to unsqueeze to [1, state, vocab] for broadcast
        let embedding_weight = self.decoder.token_embedding.weight.val();
        let weight_t = embedding_weight.transpose().unsqueeze_dim(0);
        hidden.matmul(weight_t)
    }

    pub fn generate_greedy(
        &self,
        mel: Tensor<B, 3>,
        start_tokens: Vec<u32>,
        max_len: usize,
        eot_token: u32,
    ) -> Result<Vec<Vec<u32>>> {
        let device = mel.device();
        let [batch_size, _, _] = mel.dims();

        // 1. Encode
        let encoder_output = self.encoder.forward(mel, None);

        // 2. Initialize decoder input
        // [batch_size, start_len]
        let start_tokens_i32: Vec<i32> = start_tokens.iter().map(|&t| t as i32).collect();
        let mut tokens = Tensor::<B, 1, Int>::from_ints(&start_tokens_i32[..], &device)
            .reshape([1, start_tokens.len()])
            .repeat(&[batch_size, 1]);

        // 3. Loop
        for _ in 0..max_len {
            // Forward pass to get logits
            let logits = self.forward_decoder_logits(tokens.clone(), encoder_output.clone());

            // Get logits for the last token: [batch, 1, vocab]
            let seq_len = tokens.dims()[1];
            let last_logits = logits.slice([0..batch_size, seq_len - 1..seq_len]);

            // Greedy: argmax along vocab dimension (dim 2)
            let next_token_logits = last_logits.argmax(2); // Returns [batch, 1, 1] (indices)
            let next_token = next_token_logits.squeeze::<2>(); // [batch, 1]

            // Concatenate
            tokens = Tensor::cat(vec![tokens, next_token.clone()], 1);

            // Check for EOT (simplified for batch=1)
            // In a real implementation, we'd handle batch termination masks
            if batch_size == 1 {
                let token_data = next_token.into_data();
                let token_scalar = token_data.as_slice::<i32>().unwrap()[0] as u32;
                if token_scalar == eot_token {
                    break;
                }
            }
        }

        // Convert tensors to Vec<Vec<u32>>
        let mut results = Vec::with_capacity(batch_size);
        let token_data = tokens.into_data();
        let flat_tokens: Vec<u32> = token_data
            .as_slice::<i32>()
            .unwrap()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let seq_len = flat_tokens.len() / batch_size;

        for i in 0..batch_size {
            let start = i * seq_len;
            let end = start + seq_len;
            results.push(flat_tokens[start..end].to_vec());
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn_ndarray::NdArrayDevice;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_model_creation() -> Result<()> {
        let device = NdArrayDevice::default();
        let config = WhisperConfig::tiny_en();
        let model = WhisperModel::<NdArray>::new(config.clone(), &device);

        // Test that model creates successfully
        assert_eq!(model.config.0.n_audio_state, 384);
        assert_eq!(model.config.0.n_text_state, 384);
        Ok(())
    }

    #[test]
    fn test_encoder_forward() -> Result<()> {
        let device = NdArrayDevice::default();
        let config = WhisperConfig::tiny_en();
        let encoder = WhisperEncoder::<NdArray>::new(&config, &device);

        // Test with dummy input [batch=1, time=100, mels=80]
        let x = Tensor::random(
            [1, 100, 80],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let output = encoder.forward(x, None);

        // Check output shape [batch=1, time=100, state=384]
        assert_eq!(output.dims(), [1, 100, 384]);
        Ok(())
    }
}
