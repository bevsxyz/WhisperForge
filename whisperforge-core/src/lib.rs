use anyhow::Result;
use burn::tensor::{Tensor, backend::Backend};

pub mod attn_extract;
pub mod audio;
pub mod audio_capture;
pub mod decoding;
pub mod embed;
pub mod kv_cache;
pub mod load;
pub mod model;
pub mod stream_decode;
pub mod streaming;
pub mod transcribe;
pub mod vad_silero;

#[cfg(feature = "file-io")]
pub mod audio_stream;

#[cfg(feature = "cubecl-stft")]
pub mod stft_gpu;

pub use attn_extract::forward_decoder_with_cross_attn;
pub use audio::batch_mel_spectrograms;
pub use audio::compute_mel_from_samples;
pub use audio::prepare_centered_samples_raw;
pub use audio_capture::{CaptureSource, FakeMic, MicCapture, list_input_devices};
pub use decoding::{BeamSearchDecoder, DecodingConfig, GreedyDecoder, HybridDecoder};
pub use embed::extract_speaker_embedding;
pub use kv_cache::{KvCache, forward_decoder_cached};
pub use load::{load_config_from_bytes, load_whisper_from_bytes};
pub use model::{AudioEncoderConfig, TextDecoderConfig, Whisper, WhisperConfig};
pub use stream_decode::{DecodeContext, TokenEmit, decode_window};
pub use streaming::{
    Chunker, CommitDelta, Committer, EndpointConfig, Endpointer, PromptContext, StreamWindow,
    WindowConfig,
};
pub use transcribe::{WhisperTranscriber, transcribe_audio};
pub use vad_silero::{SileroVad, ensure_silero_model};

#[cfg(feature = "file-io")]
pub use audio::load_audio_file;
#[cfg(feature = "file-io")]
pub use audio_stream::{AudioChunk, AudioChunkIterator};
#[cfg(feature = "file-io")]
pub use load::{load_config, load_whisper};

#[cfg(feature = "cubecl-stft")]
pub use stft_gpu::compute_stft_power_gpu;

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
