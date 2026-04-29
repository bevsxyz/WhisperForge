use anyhow::Result;
use burn::tensor::{backend::Backend, Tensor};

pub mod audio;
pub mod decoding;
pub mod load;
pub mod model;
pub mod transcribe;

pub use decoding::{BeamSearchDecoder, DecodingConfig, GreedyDecoder, HybridDecoder};
pub use load::{load_config, load_whisper};
pub use model::{AudioEncoderConfig, TextDecoderConfig, Whisper, WhisperConfig};
pub use transcribe::{transcribe_audio, WhisperTranscriber};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub confidence: f32,
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert!(true);
    }
}
