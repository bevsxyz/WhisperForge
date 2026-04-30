use anyhow::Result;
use burn::tensor::{Tensor, backend::Backend};

pub mod attn_extract;
pub mod audio;
pub mod decoding;
pub mod embed;
pub mod kv_cache;
pub mod load;
pub mod model;
pub mod transcribe;

pub use attn_extract::forward_decoder_with_cross_attn;
pub use audio::batch_mel_spectrograms;
pub use decoding::{BeamSearchDecoder, DecodingConfig, GreedyDecoder, HybridDecoder};
pub use embed::extract_speaker_embedding;
pub use kv_cache::{KvCache, forward_decoder_cached};
pub use load::{load_config, load_whisper};
pub use model::{AudioEncoderConfig, TextDecoderConfig, Whisper, WhisperConfig};
pub use transcribe::{WhisperTranscriber, transcribe_audio};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub confidence: f32,
    /// Per-token timestamps in seconds derived from cross-attention peaks.
    /// Populated by `transcribe_with_timestamps`; empty for plain `transcribe`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_timestamps: Vec<f32>,
    /// Speaker label assigned by diarization (e.g. `"SPEAKER_00"`).
    /// `None` when diarization was not requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
}

pub trait WhisperInference<B: Backend> {
    fn transcribe(&self, mel_features: Tensor<B, 3>) -> Result<TranscriptionResult>;
    fn transcribe_with_timestamps(&self, mel_features: Tensor<B, 3>)
    -> Result<TranscriptionResult>;
}

pub const SPECIAL_TOKENS: &str = "<|startoftranscript|> <|translate|> <|transcribe|> <|en|> <|zh|> <|de|> <|es|> <|ru|> <|ko|> <|fr|> <|ja|> <|pt|> <|tr|> <|pl|> <|ca|> <|nl|> <|ar|> <|sv|> <|it|> <|id|> <|hi|> <|fi|> <|vi|> <|he|> <|uk|> <|el|> <|ms|> <|cs|> <|ro|> <|da|> <|hu|> <|ta|> <|no|> <|th|> <|ur|> <|hr|> <|bg|> <|lt|> <|la|> <|mi|> <|ml|> <|cy|> <|sk|> <|te|> <|fa|> <|lv|> <|bn|> <|sr|> <|az|> <|sl|> <|kn|> <|et|> <|mk|> <|br|> <|eu|> <|is|> <|hy|> <|ne|> <|mn|> <|bs|> <|kk|> <|sq|> <|sw|> <|gl|> <|mr|> <|pa|> <|si|> <|km|> <|sn|> <|yo|> <|so|> <|af|> <|oc|> <|ka|> <|be|> <|tg|> <|sd|> <|gu|> <|am|> <|yi|> <|lo|> <|uz|> <|fo|> <|ht|> <|ps|> <|tk|> <|nn|> <|mt|> <|sa|> <|lb|> <|my|> <|bo|> <|tl|> <|mg|> <|as|> <|tt|> <|haw|> <|ln|> <|ha|> <|ba|> <|jw|> <|su|> <|yue|> <|notimestamps|>";
