use std::cmp::Ordering;

use anyhow::{Context, Result};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use tokenizers::Tokenizer;

use crate::{
    attn_extract::forward_decoder_with_cross_attn,
    audio,
    decoding::{DecodingConfig, HybridDecoder},
    model::Whisper,
    TranscriptionResult, TranscriptionSegment, WhisperInference,
};

const VOCAB_SIZE: usize = 51864;
const EOT: u32 = 50257;

/// Wraps a loaded Whisper model, tokenizer, and decoding config so that
/// `WhisperInference` can be implemented without needing those at the call site.
pub struct WhisperTranscriber<B: Backend> {
    pub model: Whisper<B>,
    pub tokenizer: Tokenizer,
    pub config: DecodingConfig,
}

impl<B: Backend> WhisperTranscriber<B> {
    pub fn new(model: Whisper<B>, tokenizer: Tokenizer, config: DecodingConfig) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
    }
}

impl<B: Backend> WhisperInference<B> for WhisperTranscriber<B> {
    /// Transcribe a [1, 80, 3000] mel spectrogram.
    ///
    /// Returns a single `TranscriptionSegment` spanning 0 – 30 s with
    /// no per-token timestamps.  Use [`transcribe_with_timestamps`] to get
    /// cross-attention-derived per-token timestamps.
    fn transcribe(&self, mel_features: Tensor<B, 3>) -> Result<TranscriptionResult> {
        let device = mel_features.device();

        // Safety-trim to exactly 3000 frames; compute_mel_spectrogram already
        // pads, but guard against callers who pass longer tensors.
        let expected = 3000usize;
        let [batch, n_mels, n_frames] = mel_features.dims();
        let mel = if n_frames > expected {
            mel_features.slice([0..batch, 0..n_mels, 0..expected])
        } else {
            mel_features
        };

        let tok = |s: &str, fb: u32| self.tokenizer.token_to_id(s).unwrap_or(fb);
        let sot = tok("<|startoftranscript|>", 50258);
        let en = tok("<|en|>", 50259);
        let transcribe_tok = tok("<|transcribe|>", 50359);
        let no_timestamps = tok("<|notimestamps|>", 50363);
        let eot = tok("<|endoftext|>", EOT);
        let no_speech = tok("<|nospeech|>", 50362);

        let encoder_output = self.model.forward_encoder(mel);
        let decoder = HybridDecoder::new(self.config.clone());

        let mut context: Vec<u32> = vec![sot, en, transcribe_tok, no_timestamps];
        let mut all_logits: Vec<Vec<f32>> = Vec::new();
        let budget = self.config.max_length.saturating_sub(context.len());

        for _ in 0..budget {
            let token_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(
                    context.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, context.len()],
                ),
                &device,
            );

            let logits = self
                .model
                .forward_decoder(token_tensor, encoder_output.clone());
            let [b, seq_len, _] = logits.dims();
            let step: Vec<f32> = logits
                .slice([0..b, (seq_len - 1)..seq_len, 0..VOCAB_SIZE])
                .squeeze::<1>()
                .into_data()
                .to_vec()
                .context("extracting step logits")?;

            let unconstrained = step
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(eot);

            let greedy_next = if all_logits.is_empty() && unconstrained == eot {
                step.iter()
                    .enumerate()
                    .filter(|&(i, _)| i as u32 != eot)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(eot)
            } else {
                unconstrained
            };

            all_logits.push(step);

            if greedy_next == eot {
                break;
            }
            context.push(greedy_next);
        }

        if let Some(first) = all_logits.first_mut() {
            if (eot as usize) < first.len() {
                first[eot as usize] = f32::NEG_INFINITY;
            }
        }

        let tokens = decoder.decode_with_fallback(
            &all_logits,
            no_timestamps,
            VOCAB_SIZE,
            eot,
            no_speech,
            |ids| self.tokenizer.decode(ids, false).unwrap_or_default(),
        )?;

        let text_tokens: Vec<u32> = tokens.into_iter().filter(|&t| t < EOT).collect();
        let text = self
            .tokenizer
            .decode(&text_tokens, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let text = text.trim().to_string();

        if text.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: vec![],
                language: None,
            });
        }

        // Approximate per-segment timing: evenly distribute 30 s across words.
        let chunk_duration = 30.0f32;
        let segment = TranscriptionSegment {
            start: 0.0,
            end: chunk_duration,
            text: text.clone(),
            tokens: text_tokens,
            confidence: 1.0,
            token_timestamps: vec![],
            speaker: None,
        };

        Ok(TranscriptionResult {
            text,
            segments: vec![segment],
            language: None,
        })
    }

    /// Transcribe with per-token timestamps derived from cross-attention peaks.
    ///
    /// Each token's timestamp is `argmax(avg_layer_head_attn) * 2 * 160 / 16000`
    /// seconds (the 2× accounts for the encoder's stride-2 Conv1d).  Segment
    /// `start`/`end` are set to the first/last token timestamp rather than the
    /// full 0–30 s placeholder used by [`transcribe`].
    fn transcribe_with_timestamps(
        &self,
        mel_features: Tensor<B, 3>,
    ) -> Result<TranscriptionResult> {
        let device = mel_features.device();

        let expected = 3000usize;
        let [batch, n_mels, n_frames] = mel_features.dims();
        let mel = if n_frames > expected {
            mel_features.slice([0..batch, 0..n_mels, 0..expected])
        } else {
            mel_features
        };

        let tok = |s: &str, fb: u32| self.tokenizer.token_to_id(s).unwrap_or(fb);
        let sot = tok("<|startoftranscript|>", 50258);
        let en = tok("<|en|>", 50259);
        let transcribe_tok = tok("<|transcribe|>", 50359);
        let no_timestamps = tok("<|notimestamps|>", 50363);
        let eot = tok("<|endoftext|>", EOT);
        let no_speech = tok("<|nospeech|>", 50362);

        let encoder_output = self.model.forward_encoder(mel);
        let decoder = HybridDecoder::new(self.config.clone());

        let mut context: Vec<u32> = vec![sot, en, transcribe_tok, no_timestamps];
        let mut all_logits: Vec<Vec<f32>> = Vec::new();
        // One timestamp per generated (non-EOT) token.
        let mut token_timestamps: Vec<f32> = Vec::new();
        let budget = self.config.max_length.saturating_sub(context.len());

        // hop_length=160, sample_rate=16000, encoder stride-2 halves time dim.
        const SECONDS_PER_ENCODER_FRAME: f32 = 2.0 * 160.0 / 16000.0;

        for _ in 0..budget {
            let token_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(
                    context.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, context.len()],
                ),
                &device,
            );

            let (logits, frame_weights) =
                forward_decoder_with_cross_attn(&self.model, token_tensor, encoder_output.clone());

            let [b, seq_len, _] = logits.dims();
            let step: Vec<f32> = logits
                .slice([0..b, (seq_len - 1)..seq_len, 0..VOCAB_SIZE])
                .squeeze::<1>()
                .into_data()
                .to_vec()
                .context("extracting step logits")?;

            let best_frame = frame_weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let ts = best_frame as f32 * SECONDS_PER_ENCODER_FRAME;

            let unconstrained = step
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(eot);

            let greedy_next = if all_logits.is_empty() && unconstrained == eot {
                step.iter()
                    .enumerate()
                    .filter(|&(i, _)| i as u32 != eot)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(eot)
            } else {
                unconstrained
            };

            all_logits.push(step);

            if greedy_next == eot {
                break;
            }
            token_timestamps.push(ts);
            context.push(greedy_next);
        }

        if let Some(first) = all_logits.first_mut() {
            if (eot as usize) < first.len() {
                first[eot as usize] = f32::NEG_INFINITY;
            }
        }

        let tokens = decoder.decode_with_fallback(
            &all_logits,
            no_timestamps,
            VOCAB_SIZE,
            eot,
            no_speech,
            |ids| self.tokenizer.decode(ids, false).unwrap_or_default(),
        )?;

        let text_tokens: Vec<u32> = tokens.into_iter().filter(|&t| t < EOT).collect();
        let text = self
            .tokenizer
            .decode(&text_tokens, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let text = text.trim().to_string();

        if text.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: vec![],
                language: None,
            });
        }

        let n_tok = text_tokens.len().min(token_timestamps.len());
        let seg_start = token_timestamps.first().copied().unwrap_or(0.0);
        let seg_end = token_timestamps
            .get(n_tok.saturating_sub(1))
            .copied()
            .unwrap_or(30.0);

        let segment = TranscriptionSegment {
            start: seg_start,
            end: seg_end,
            text: text.clone(),
            tokens: text_tokens,
            confidence: 1.0,
            token_timestamps: token_timestamps[..n_tok].to_vec(),
            speaker: None,
        };

        Ok(TranscriptionResult {
            text,
            segments: vec![segment],
            language: None,
        })
    }
}

/// Convenience: transcribe a complete audio file, chunking at 30 s with 1 s overlap.
///
/// Each chunk's segment timestamps are shifted to absolute audio time before
/// assembling the final `TranscriptionResult`.
pub fn transcribe_audio<B: Backend>(
    transcriber: &WhisperTranscriber<B>,
    audio: &audio::AudioData,
    device: &B::Device,
) -> Result<TranscriptionResult> {
    let chunk_samples = 30 * audio.sample_rate as usize;
    let overlap_samples = audio.sample_rate as usize;
    let step = chunk_samples.saturating_sub(overlap_samples).max(1);

    let chunks = chunk_audio_fixed(audio, chunk_samples, overlap_samples);

    let mut all_segments: Vec<TranscriptionSegment> = Vec::new();
    let mut all_parts: Vec<String> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_start = (i * step) as f32 / audio.sample_rate as f32;
        let chunk_end = chunk_start + chunk.samples.len() as f32 / audio.sample_rate as f32;

        let mel = audio::compute_mel_spectrogram(chunk, 400, 160, 80, device)?;
        let result = transcriber.transcribe(mel)?;

        for mut seg in result.segments {
            seg.start += chunk_start;
            seg.end = seg.end.min(chunk_end) + chunk_start;
            all_segments.push(seg);
        }
        if !result.text.is_empty() {
            all_parts.push(result.text);
        }
    }

    Ok(TranscriptionResult {
        text: all_parts.join(" "),
        segments: all_segments,
        language: None,
    })
}

fn chunk_audio_fixed(
    audio: &audio::AudioData,
    chunk_samples: usize,
    overlap_samples: usize,
) -> Vec<audio::AudioData> {
    let n = audio.samples.len();
    if n <= chunk_samples {
        return vec![audio.clone()];
    }
    let step = chunk_samples.saturating_sub(overlap_samples).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + chunk_samples).min(n);
        chunks.push(audio::AudioData {
            samples: audio.samples[start..end].to_vec(),
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        });
        if end == n {
            break;
        }
        start += step;
    }
    chunks
}
