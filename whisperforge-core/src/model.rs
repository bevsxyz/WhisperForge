use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        transformer::{TransformerDecoder, TransformerDecoderConfig, TransformerEncoder, TransformerEncoderConfig},
        Embedding, EmbeddingConfig, Linear, LinearConfig, ReLU, LayerNorm, LayerNormConfig,
    },
    tensor::{activation::softmax, backend::Backend, Bool, Int, Tensor},
};

#[derive(Config)]
pub struct WhisperConfig {
    pub n_vocab: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
    pub n_mels: usize,
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    encoder: TransformerEncoder<B>,
    positional_embedding: Tensor<B, 2>,
}

impl<B: Backend> AudioEncoder<B> {
    pub fn new(config: &WhisperConfig, device: &B::Device) -> Self {
        let conv1 = Conv1dConfig::new([config.n_mels, config.n_audio_state], 3)
            .with_stride(1)
            .with_padding(1)
            .init(device);
        let conv2 = Conv1dConfig::new([config.n_audio_state, config.n_audio_state], 3)
            .with_stride(2)
            .with_padding(1)
            .init(device);

        let encoder_config = TransformerEncoderConfig::new(
            config.n_audio_state,
            config.n_audio_head,
            config.n_audio_layer,
            config.n_audio_state * 4,
        )
        .with_norm_first(true);

        let encoder = encoder_config.init(device);

        // Positional embedding for audio (1500 tokens)
        let positional_embedding = Tensor::random(
            [config.n_audio_ctx, config.n_audio_state],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        Self {
            conv1,
            conv2,
            encoder,
            positional_embedding,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, n_mels, seq_len] = x.dims();
        
        // Apply convolutions
        let x = self.conv1.forward(x.clone());
        let x = x.relu();
        let x = self.conv2.forward(x);
        let x = x.relu();

        // Reshape for transformer: [batch, channels, seq] -> [batch, seq, channels]
        let x = x.swap_dims(1, 2);

        // Add positional embedding
        let pos_emb = self.positional_embedding.clone().slice([0..seq_len]);
        let x = x + pos_emb.unsqueeze_dim(0);

        // Apply transformer encoder
        self.encoder.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Tensor<B, 2>,
    decoder: TransformerDecoder<B>,
    ln: LayerNorm<B>,
}

impl<B: Backend> TextDecoder<B> {
    pub fn new(config: &WhisperConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.n_vocab, config.n_text_state).init(device);

        let decoder_config = TransformerDecoderConfig::new(
            config.n_text_state,
            config.n_text_head,
            config.n_text_layer,
            config.n_text_state * 4,
        )
        .with_norm_first(true)
        .with_mask(true);

        let decoder = decoder_config.init(device);

        let positional_embedding = Tensor::random(
            [config.n_text_ctx, config.n_text_state],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        let ln = LayerNormConfig::new(config.n_text_state).init(device);

        Self {
            token_embedding,
            positional_embedding,
            decoder,
            ln,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
        encoder_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = x.dims();

        // Token embeddings
        let x = self.token_embedding.forward(x.clone());

        // Add positional embedding
        let pos_emb = self.positional_embedding.clone().slice([0..seq_len]);
        let x = x + pos_emb.unsqueeze_dim(0);

        // Create causal mask
        let device = x.device();
        let mask = Tensor::ones([seq_len, seq_len], device).tril(0).unsqueeze_dim(0);

        // Apply transformer decoder
        let x = self.decoder.forward(
            x.clone(),
            encoder_output.clone(),
            Some(mask),
            encoder_mask,
        );

        // Apply layer norm
        self.ln.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct WhisperModel<B: Backend> {
    encoder: AudioEncoder<B>,
    decoder: TextDecoder<B>,
    config: WhisperConfig,
}

impl<B: Backend> WhisperModel<B> {
    pub fn new(config: WhisperConfig, device: &B::Device) -> Self {
        let encoder = AudioEncoder::new(&config, device);
        let decoder = TextDecoder::new(&config, device);

        Self {
            encoder,
            decoder,
            config,
        }
    }

    pub fn forward(&self, mel_features: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Encode audio
        let encoder_output = self.encoder.forward(mel_features);

        // Decode text
        let decoder_output = self.decoder.forward(tokens, encoder_output, None);

        decoder_output
    }

    pub fn encode(&self, mel_features: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel_features)
    }

    pub fn decode(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output, None)
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }
}

// Model configurations for different Whisper sizes
impl WhisperConfig {
    pub fn tiny() -> Self {
        Self {
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
        }
    }

    pub fn base() -> Self {
        Self {
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 512,
            n_audio_head: 8,
            n_audio_layer: 6,
            n_text_ctx: 448,
            n_text_state: 512,
            n_text_head: 8,
            n_text_layer: 6,
            n_mels: 80,
        }
    }

    pub fn small() -> Self {
        Self {
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 768,
            n_audio_head: 12,
            n_audio_layer: 12,
            n_text_ctx: 448,
            n_text_state: 768,
            n_text_head: 12,
            n_text_layer: 12,
            n_mels: 80,
        }
    }

    pub fn medium() -> Self {
        Self {
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 1024,
            n_audio_head: 16,
            n_audio_layer: 24,
            n_text_ctx: 448,
            n_text_state: 1024,
            n_text_head: 16,
            n_text_layer: 24,
            n_mels: 80,
        }
    }

    pub fn large_v2() -> Self {
        Self {
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
            n_mels: 80,
        }
    }
}