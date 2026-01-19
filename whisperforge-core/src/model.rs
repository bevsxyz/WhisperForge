// Whisper model implementation adapted from whisper-burn (MIT License)
// https://github.com/Gadersd/whisper-burn

use std::f32::NEG_INFINITY;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        conv::{Conv1d, Conv1dConfig},
        PaddingConfig1d,
    },
    tensor::{activation::softmax, backend::Backend, Distribution, Int, Tensor},
};

/// Configuration for the Whisper model
#[derive(Config, Debug)]
pub struct WhisperConfig {
    pub audio_encoder_config: AudioEncoderConfig,
    pub text_decoder_config: TextDecoderConfig,
}

impl WhisperConfig {
    /// Create config for tiny.en model
    pub fn tiny_en() -> Self {
        Self {
            audio_encoder_config: AudioEncoderConfig {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 384,
                n_audio_head: 6,
                n_audio_layer: 4,
            },
            text_decoder_config: TextDecoderConfig {
                n_vocab: 51864,
                n_text_ctx: 448,
                n_text_state: 384,
                n_text_head: 6,
                n_text_layer: 4,
            },
        }
    }

    /// Create config for base model
    pub fn base() -> Self {
        Self {
            audio_encoder_config: AudioEncoderConfig {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 512,
                n_audio_head: 8,
                n_audio_layer: 6,
            },
            text_decoder_config: TextDecoderConfig {
                n_vocab: 51864,
                n_text_ctx: 448,
                n_text_state: 512,
                n_text_head: 8,
                n_text_layer: 6,
            },
        }
    }

    /// Create config for small model
    pub fn small() -> Self {
        Self {
            audio_encoder_config: AudioEncoderConfig {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 768,
                n_audio_head: 12,
                n_audio_layer: 12,
            },
            text_decoder_config: TextDecoderConfig {
                n_vocab: 51864,
                n_text_ctx: 448,
                n_text_state: 768,
                n_text_head: 12,
                n_text_layer: 12,
            },
        }
    }

    /// Create config for medium model
    pub fn medium() -> Self {
        Self {
            audio_encoder_config: AudioEncoderConfig {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 1024,
                n_audio_head: 16,
                n_audio_layer: 24,
            },
            text_decoder_config: TextDecoderConfig {
                n_vocab: 51864,
                n_text_ctx: 448,
                n_text_state: 1024,
                n_text_head: 16,
                n_text_layer: 24,
            },
        }
    }

    /// Create config for large-v2 model
    pub fn large_v2() -> Self {
        Self {
            audio_encoder_config: AudioEncoderConfig {
                n_mels: 128,
                n_audio_ctx: 1500,
                n_audio_state: 1280,
                n_audio_head: 20,
                n_audio_layer: 32,
            },
            text_decoder_config: TextDecoderConfig {
                n_vocab: 51864,
                n_text_ctx: 448,
                n_text_state: 1280,
                n_text_head: 20,
                n_text_layer: 32,
            },
        }
    }

    /// Create config for large-v3 model
    pub fn large_v3() -> Self {
        Self {
            audio_encoder_config: AudioEncoderConfig {
                n_mels: 128,
                n_audio_ctx: 1500,
                n_audio_state: 1280,
                n_audio_head: 20,
                n_audio_layer: 32,
            },
            text_decoder_config: TextDecoderConfig {
                n_vocab: 51865,
                n_text_ctx: 448,
                n_text_state: 1280,
                n_text_head: 20,
                n_text_layer: 32,
            },
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Whisper<B> {
        let n_audio_state = self.audio_encoder_config.n_audio_state;
        let n_text_state = self.text_decoder_config.n_text_state;

        assert!(
            n_audio_state == n_text_state,
            "Audio encoder state size {} must be equal to text decoder state size {}.",
            n_audio_state,
            n_text_state
        );

        let encoder = self.audio_encoder_config.init(device);
        let decoder = self.text_decoder_config.init(device);

        Whisper { encoder, decoder }
    }
}

/// The main Whisper model
#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    pub encoder: AudioEncoder<B>,
    pub decoder: TextDecoder<B>,
}

impl<B: Backend> Whisper<B> {
    /// Full forward pass: encode audio and decode tokens
    pub fn forward(&self, mel: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.decoder.forward(tokens, self.encoder.forward(mel))
    }

    /// Encode audio mel spectrogram
    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    /// Decode tokens given encoder output
    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn encoder_ctx_size(&self) -> usize {
        self.encoder.ctx_size()
    }

    pub fn decoder_ctx_size(&self) -> usize {
        self.decoder.ctx_size()
    }
}

// ============================================================================
// Text Decoder
// ============================================================================

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextDecoder<B> {
        let token_embedding = Param::from_tensor(Tensor::random(
            [self.n_vocab, self.n_text_state],
            Distribution::Normal(0.0, 0.02),
            device,
        ));
        let positional_embedding = Param::from_tensor(Tensor::random(
            [self.n_text_ctx, self.n_text_state],
            Distribution::Normal(0.0, 0.01),
            device,
        ));
        let blocks: Vec<_> = (0..self.n_text_layer)
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                    .init(device)
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init(device);

        let mask = Param::from_tensor(attn_decoder_mask(self.n_text_ctx, device));

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    pub token_embedding: Param<Tensor<B, 2>>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    pub ln: nn::LayerNorm<B>,
    pub mask: Param<Tensor<B, 2>>,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
}

impl<B: Backend> TextDecoder<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        // Token embedding lookup
        let x = burn::tensor::module::embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, xa.clone(), self.mask.val());
        }

        let x = self.ln.forward(x);
        // Project to vocabulary logits
        x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>())
    }

    pub fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }
}

// ============================================================================
// Audio Encoder
// ============================================================================

#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}

impl AudioEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioEncoder<B> {
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);
        let gelu1 = nn::Gelu::new();
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_stride(2)
            .init(device);
        let gelu2 = nn::Gelu::new();
        let blocks: Vec<_> = (0..self.n_audio_layer)
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init(device)
            })
            .collect();
        let ln_post = nn::LayerNormConfig::new(self.n_audio_state).init(device);
        let positional_embedding = Param::from_tensor(Tensor::random(
            [self.n_audio_ctx, self.n_audio_state],
            Distribution::Normal(0.0, 0.01),
            device,
        ));
        let n_mels = self.n_mels;
        let n_audio_ctx = self.n_audio_ctx;

        AudioEncoder {
            conv1,
            gelu1,
            conv2,
            gelu2,
            blocks,
            ln_post,
            positional_embedding,
            n_mels,
            n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub gelu1: nn::Gelu,
    pub conv2: Conv1d<B>,
    pub gelu2: nn::Gelu,
    pub blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    pub ln_post: nn::LayerNorm<B>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub n_mels: usize,
    pub n_audio_ctx: usize,
}

impl<B: Backend> AudioEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n_mels, n_ctx] = x.dims();

        assert!(
            n_mels == self.n_mels,
            "Audio mel spectrum size must be {}.",
            self.n_mels
        );
        assert!(
            n_ctx <= 2 * self.n_audio_ctx,
            "Audio length {} cannot exceed {}.",
            n_ctx,
            2 * self.n_audio_ctx
        );

        let x = self.gelu1.forward(self.conv1.forward(x));
        let x = self.gelu2.forward(self.conv2.forward(x));

        let x = x.swap_dims(1, 2);
        let k = x.dims()[1];
        let x = x + self
            .positional_embedding
            .val()
            .slice([0..k])
            .unsqueeze::<3>();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        self.ln_post.forward(x)
    }

    pub fn ctx_size(&self) -> usize {
        self.n_audio_ctx
    }
}

// ============================================================================
// Attention Blocks
// ============================================================================

#[derive(Config, Debug)]
pub struct ResidualEncoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualEncoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(device);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let mlp = MLPConfig::new(self.n_state).init(device);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        ResidualEncoderAttentionBlock {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadSelfAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub mlp: MLP<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), None);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        x
    }
}

#[derive(Config, Debug)]
pub struct ResidualDecoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualDecoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(device);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let cross_attn = MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head).init(device);
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let mlp = MLPConfig::new(self.n_state).init(device);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadSelfAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub cross_attn: MultiHeadCrossAttention<B>,
    pub cross_attn_ln: nn::LayerNorm<B>,
    pub mlp: MLP<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), Some(mask));
        let x = x.clone() + self.cross_attn.forward(self.cross_attn_ln.forward(x), xa);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        x
    }
}

// ============================================================================
// MLP (Feed-Forward Network)
// ============================================================================

#[derive(Config, Debug)]
pub struct MLPConfig {
    n_state: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        let lin1 = nn::LinearConfig::new(self.n_state, 4 * self.n_state).init(device);
        let gelu = nn::Gelu::new();
        let lin2 = nn::LinearConfig::new(4 * self.n_state, self.n_state).init(device);

        MLP { lin1, gelu, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub lin1: nn::Linear<B>,
    pub gelu: nn::Gelu,
    pub lin2: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.lin1.forward(x);
        let x = self.gelu.forward(x);
        self.lin2.forward(x)
    }
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

#[derive(Config, Debug)]
pub struct MultiHeadSelfAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadSelfAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);

        MultiHeadSelfAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    pub n_head: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let wv = qkv_attention(q, k, v, mask, self.n_head);

        self.out.forward(wv)
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadCrossAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadCrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadCrossAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);

        MultiHeadCrossAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention<B: Backend> {
    pub n_head: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadCrossAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.query.forward(x);
        let k = self.key.forward(xa.clone());
        let v = self.value.forward(xa);

        let wv = qkv_attention(q, k, v, None, self.n_head);

        self.out.forward(wv)
    }
}

// ============================================================================
// Attention Utilities
// ============================================================================

pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let scale = (n_state as f64 / n_head as f64).powf(-0.25);
    let n_hstate = n_state / n_head;

    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .swap_dims(1, 2)
        * scale;
    let k = k
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2)
        .transpose()
        * scale;
    let v = v
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2);

    let qk = q.matmul(k);

    // Apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    // Normalize value weightings
    let w = softmax(qk, 3);
    let o = w.matmul(v).swap_dims(1, 2).flatten(2, 3);

    o
}

/// Create causal attention mask for decoder
pub fn attn_decoder_mask<B: Backend>(seq_length: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length], device);

    for i in 0..(seq_length - 1) {
        let values =
            Tensor::<B, 2>::zeros([1, seq_length - (i + 1)], device).add_scalar(NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn_ndarray::NdArrayDevice;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_config_creation() {
        let config = WhisperConfig::tiny_en();
        assert_eq!(config.audio_encoder_config.n_audio_state, 384);
        assert_eq!(config.text_decoder_config.n_text_state, 384);
        assert_eq!(config.audio_encoder_config.n_mels, 80);
    }

    #[test]
    fn test_model_init() {
        let device = NdArrayDevice::default();
        let config = WhisperConfig::tiny_en();
        let model = config.init::<TestBackend>(&device);

        assert_eq!(model.encoder.n_mels, 80);
        assert_eq!(model.decoder.n_vocab, 51864);
    }

    #[test]
    fn test_encoder_forward() {
        let device = NdArrayDevice::default();
        let config = WhisperConfig::tiny_en();
        let model = config.init::<TestBackend>(&device);

        // Input: [batch=1, n_mels=80, time=100]
        let mel = Tensor::random([1, 80, 100], Distribution::Normal(0.0, 1.0), &device);
        let output = model.encoder.forward(mel);

        // Output: [batch=1, time/2=50, n_state=384]
        assert_eq!(output.dims()[0], 1);
        assert_eq!(output.dims()[1], 50); // Stride 2 halves the time dimension
        assert_eq!(output.dims()[2], 384);
    }

    #[test]
    fn test_decoder_forward() {
        let device = NdArrayDevice::default();
        let config = WhisperConfig::tiny_en();
        let model = config.init::<TestBackend>(&device);

        // Encoder output: [batch=1, time=50, n_state=384]
        let encoder_output = Tensor::random([1, 50, 384], Distribution::Normal(0.0, 1.0), &device);

        // Tokens: [batch=1, seq_len=5]
        let tokens = Tensor::<TestBackend, 2, Int>::zeros([1, 5], &device);

        let logits = model.decoder.forward(tokens, encoder_output);

        // Output: [batch=1, seq_len=5, vocab=51864]
        assert_eq!(logits.dims()[0], 1);
        assert_eq!(logits.dims()[1], 5);
        assert_eq!(logits.dims()[2], 51864);
    }

    #[test]
    fn test_attention_mask() {
        let device = NdArrayDevice::default();
        let mask = attn_decoder_mask::<TestBackend>(4, &device);

        assert_eq!(mask.dims(), [4, 4]);
        // Check that it's lower triangular (zeros on diagonal and below, -inf above)
    }
}
