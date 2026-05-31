use std::time::Instant;

use anyhow::{Context, Result};
use burn::tensor::{Tensor, backend::Backend};
use tokenizers::Tokenizer;
use tracing::{Level, event};

use crate::kv_cache::{KvCache, forward_decoder_cached};
use crate::model::Whisper;

#[derive(Clone)]
pub struct TokenEmit {
    pub id: u32,
    /// Detokenised surface text for this single token (via `tokenizer.decode(&[id], false)`),
    /// so concatenating regular-token `text` fields yields a properly-spaced transcript.
    /// Empty string for special and timestamp tokens.
    pub text: String,
    pub logprob: f32,
    /// Seconds within the 30-s window for timestamp tokens; `None` for regular tokens.
    pub window_ts_secs: Option<f32>,
    /// `true` when `id >= eot_token` (EOT, language, task, timestamp, etc.).
    pub is_special: bool,
}

pub struct DecodeContext<'a> {
    /// Tokens from the previous committed utterance fed as a prompt prefix before `<|sot|>`.
    /// Empty slice for the first window.
    pub prompt_tokens: &'a [u32],
    pub language_token: u32,
    /// Task token (`<|transcribe|>` or `<|translate|>`). Translate is X → English only.
    pub task_token: u32,
    pub sot_token: u32,
    pub eot_token: u32,
    pub no_speech_token: u32,
    /// First timestamp token ID (50364 for all current Whisper models).
    pub timestamp_begin_token: u32,
    /// `<|notimestamps|>` token ID. Pushed as the 4th init token so the decoder produces
    /// plain text (no per-token timestamps between words). Required for greedy decode to
    /// produce coherent content on this checkpoint — greedy + timestamps-on emits mostly
    /// timestamps with little content even with `ApplyTimestampRules`-style filtering.
    pub notimestamps_token: u32,
    /// Hard cap on new tokens generated (not counting prompt or init tokens).
    pub max_new_tokens: usize,
    /// Return empty `Vec` when P(no_speech) exceeds this at step 0 (default 0.6).
    pub no_speech_threshold: f32,
}

/// Greedy KV-cached decode for one streaming window.
///
/// `encoder_out` is consumed by `KvCache::new`; clone before calling if you need it again.
///
/// `<|notimestamps|>` is pushed as the fourth init token so the decoder emits plain text
/// (no per-token timestamps between words). This matches the one-shot transcribe path and
/// gives reliable greedy output on short windows. The original plan was to leave timestamps
/// enabled so the streaming caller could anchor buffer trims at committed-token boundaries
/// — but empirically, greedy decode on tiny.en with timestamps on emits mostly timestamps
/// and little content, even with `ApplyTimestampRules`-style logit filtering. Reliable
/// timestamps in streaming would require temperature-fallback sampling (like the one-shot
/// `HybridDecoder`), which is out of scope here. The streaming caller uses a stride-based
/// trim heuristic on cap-hit instead — see `whisperforge/src/commands/stream.rs`.
pub fn decode_window<B: Backend>(
    model: &Whisper<B>,
    encoder_out: Tensor<B, 3>,
    ctx: &DecodeContext,
    tokenizer: &Tokenizer,
    device: &B::Device,
) -> Result<Vec<TokenEmit>> {
    let t0 = Instant::now();

    let mut cache = KvCache::new(model, encoder_out);

    // Seed the self-attention KV cache with any prior-context prompt tokens.
    if !ctx.prompt_tokens.is_empty() {
        event!(
            Level::DEBUG,
            prompt_len = ctx.prompt_tokens.len(),
            prompt_first_token = ctx.prompt_tokens[0],
            "feeding prompt prefix into KV cache",
        );
    }
    for &tok in ctx.prompt_tokens {
        forward_decoder_cached(model, tok, &mut cache, device)
            .with_context(|| format!("feeding prompt token {tok}"))?;
    }

    // Feed the four init tokens. `<|notimestamps|>` is included so the greedy decoder
    // produces plain text (matching the one-shot transcribe path). With timestamps enabled,
    // greedy decode on this checkpoint emits mostly timestamps with little content even
    // under `ApplyTimestampRules`-style filtering — see function docstring.
    let init = [
        ctx.sot_token,
        ctx.language_token,
        ctx.task_token,
        ctx.notimestamps_token,
    ];
    let mut logits: Vec<f32> = Vec::new();
    for (i, &tok) in init.iter().enumerate() {
        logits = forward_decoder_cached(model, tok, &mut cache, device)
            .with_context(|| format!("feeding init token at index {i}"))?;
    }

    // No-speech gate: if the model is confident there is no speech, skip this window.
    if softmax_at(&logits, ctx.no_speech_token) > ctx.no_speech_threshold {
        event!(
            Level::DEBUG,
            decode_ms = t0.elapsed().as_millis(),
            n_tokens = 0usize,
            skipped = true
        );
        return Ok(Vec::new());
    }

    // Suppress EOT at step 0 to avoid a premature stop on the very first generated token.
    if (ctx.eot_token as usize) < logits.len() {
        logits[ctx.eot_token as usize] = f32::NEG_INFINITY;
    }

    let mut emits: Vec<TokenEmit> = Vec::new();

    for _ in 0..ctx.max_new_tokens {
        let token_id = argmax(&logits);

        if token_id == ctx.eot_token {
            break;
        }

        let logprob = log_softmax_at(&logits, token_id);
        let is_special = token_id >= ctx.eot_token;
        let window_ts_secs = if token_id >= ctx.timestamp_begin_token {
            Some((token_id - ctx.timestamp_begin_token) as f32 * 0.02)
        } else {
            None
        };
        let text = if is_special {
            String::new()
        } else {
            tokenizer.decode(&[token_id], false).unwrap_or_default()
        };

        emits.push(TokenEmit {
            id: token_id,
            text,
            logprob,
            window_ts_secs,
            is_special,
        });

        logits = forward_decoder_cached(model, token_id, &mut cache, device)
            .with_context(|| format!("decode step {}", emits.len()))?;
    }

    event!(
        Level::DEBUG,
        decode_ms = t0.elapsed().as_millis(),
        n_tokens = emits.len()
    );

    // Punctuation-only decodes are a Whisper failure mode on short/uncertain windows: the
    // model emits a lone `.` or `,` then EOT. Treat as no-speech so the streaming committer
    // doesn't pick up the noise as a stable prefix. Only fires when there *is* regular text
    // — an emits list containing only specials/timestamps still passes through (the
    // committer ignores them) so we don't accidentally drop legitimate timestamp-only
    // windows.
    let regular_text: String = emits
        .iter()
        .filter(|t| !t.is_special)
        .map(|t| t.text.as_str())
        .collect();
    let trimmed = regular_text.trim();
    if !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c.is_ascii_punctuation() || c.is_whitespace())
    {
        event!(
            Level::DEBUG,
            dropped_punctuation_only = true,
            text = %trimmed
        );
        return Ok(Vec::new());
    }

    Ok(emits)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn softmax_at(logits: &[f32], token: u32) -> f32 {
    let idx = token as usize;
    if idx >= logits.len() {
        return 0.0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    ((logits[idx] - max).exp()) / exp_sum.max(f32::EPSILON)
}

fn log_softmax_at(logits: &[f32], token: u32) -> f32 {
    let idx = token as usize;
    if idx >= logits.len() {
        return f32::NEG_INFINITY;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits[idx] - log_sum
}

pub fn avg_logprob(tokens: &[TokenEmit]) -> f32 {
    let content: Vec<f32> = tokens
        .iter()
        .filter(|t| !t.is_special)
        .map(|t| t.logprob)
        .collect();
    if content.is_empty() {
        return 0.0;
    }
    content.iter().sum::<f32>() / content.len() as f32
}

/// Per-window quality thresholds, mirroring faster-whisper's `log_prob_threshold`
/// and `compression_ratio_threshold`. Used by the streaming caller to reject a decoded
/// window before it reaches the LocalAgreement committer — LA-2 only rejects *unstable*
/// output, so a *confident* hallucination loop (the `*sigh* *sigh* *sigh*` failure mode)
/// would otherwise commit. The defaults match faster-whisper.
#[derive(Clone, Copy, Debug)]
pub struct QualityGate {
    /// Reject the window when `avg_logprob` of content tokens is below this (default -1.0).
    pub log_prob_threshold: f32,
    /// Reject the window when the gzip compression ratio of its text exceeds this
    /// (default 2.4) — high ratios signal a repetition/hallucination loop.
    pub compression_ratio_threshold: f32,
}

impl Default for QualityGate {
    fn default() -> Self {
        Self {
            log_prob_threshold: -1.0,
            compression_ratio_threshold: 2.4,
        }
    }
}

/// Returns `false` when `emits` should be dropped as low-confidence or repetitive.
///
/// Windows with no content (regular) tokens always pass: the no-speech and
/// punctuation-only gates in [`decode_window`] already handle empties, and the
/// streaming committer ignores special/timestamp-only emits.
pub fn passes_quality_gate(emits: &[TokenEmit], gate: &QualityGate) -> bool {
    let text: String = emits
        .iter()
        .filter(|t| !t.is_special)
        .map(|t| t.text.as_str())
        .collect();
    if text.trim().is_empty() {
        return true;
    }
    if avg_logprob(emits) < gate.log_prob_threshold {
        return false;
    }
    if crate::decoding::compression_ratio(&text) > gate.compression_ratio_threshold {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use burn_flex::{Flex, FlexDevice};

    use crate::model::WhisperConfig;

    fn tiny_en_random() -> (Whisper<Flex<f32>>, FlexDevice) {
        let device = FlexDevice;
        let config = WhisperConfig::tiny_en();
        let model = config.init::<Flex<f32>>(&device);
        (model, device)
    }

    fn dummy_tokenizer() -> Tokenizer {
        // Backed by an empty BPE model; id_to_token returns None for every ID,
        // which decode_window handles gracefully via unwrap_or_default().
        Tokenizer::new(tokenizers::models::bpe::BPE::default())
    }

    fn ctx_no_gate<'a>() -> DecodeContext<'a> {
        DecodeContext {
            prompt_tokens: &[],
            language_token: 50259,
            task_token: 50359,
            sot_token: 50258,
            eot_token: 50257,
            no_speech_token: 50362,
            notimestamps_token: 50363,
            timestamp_begin_token: 50364,
            max_new_tokens: 8,
            // Set very high so the no-speech gate never fires with random weights.
            no_speech_threshold: 0.999,
        }
    }

    fn content_emit(text: &str, logprob: f32) -> TokenEmit {
        TokenEmit {
            id: 1,
            text: text.to_string(),
            logprob,
            window_ts_secs: None,
            is_special: false,
        }
    }

    #[test]
    fn test_quality_gate_passes_normal() {
        let gate = QualityGate::default();
        let emits = vec![
            content_emit(" the", -0.2),
            content_emit(" quick", -0.4),
            content_emit(" brown", -0.3),
            content_emit(" fox", -0.5),
        ];
        assert!(
            passes_quality_gate(&emits, &gate),
            "varied, confident text should pass"
        );
    }

    #[test]
    fn test_quality_gate_rejects_low_logprob() {
        let gate = QualityGate::default();
        // avg_logprob well below the -1.0 floor; compression ratio is irrelevant here.
        let emits = vec![content_emit(" maybe", -2.5), content_emit(" perhaps", -3.0)];
        assert!(
            !passes_quality_gate(&emits, &gate),
            "low-confidence window should be rejected"
        );
    }

    #[test]
    fn test_quality_gate_rejects_repetition() {
        let gate = QualityGate::default();
        // Confident (high logprob) but a repetition loop → high compression ratio.
        let mut emits = Vec::new();
        for _ in 0..60 {
            emits.push(content_emit(" sigh", -0.1));
        }
        assert!(
            !passes_quality_gate(&emits, &gate),
            "confident repetition loop should be rejected on compression ratio"
        );
    }

    #[test]
    fn test_quality_gate_empty_passes() {
        let gate = QualityGate::default();
        // Specials-only / empty content → always passes (handled elsewhere).
        let emits: Vec<TokenEmit> = vec![TokenEmit {
            id: 50364,
            text: String::new(),
            logprob: -5.0,
            window_ts_secs: Some(0.0),
            is_special: true,
        }];
        assert!(passes_quality_gate(&emits, &gate));
        assert!(passes_quality_gate(&[], &gate));
    }

    /// Structural test: verify `decode_window` compiles, runs, and returns Ok without panicking
    /// on a random model with a zero encoder output.
    #[test]
    fn test_decode_window_random_model() -> Result<()> {
        let (model, device) = tiny_en_random();
        let encoder_out = burn::tensor::Tensor::<Flex<f32>, 3>::zeros([1, 1500, 384], &device);
        let tokenizer = dummy_tokenizer();
        let ctx = ctx_no_gate();

        let emits = decode_window(&model, encoder_out, &ctx, &tokenizer, &device)?;
        assert!(emits.len() <= 8, "emits exceeded max_new_tokens");
        Ok(())
    }

    /// No-speech gate: with threshold = 0.0 the gate always fires (any P > 0).
    #[test]
    fn test_decode_window_no_speech_gate() -> Result<()> {
        let (model, device) = tiny_en_random();
        let encoder_out = burn::tensor::Tensor::<Flex<f32>, 3>::zeros([1, 1500, 384], &device);
        let tokenizer = dummy_tokenizer();
        let ctx = DecodeContext {
            no_speech_threshold: 0.0,
            ..ctx_no_gate()
        };

        let emits = decode_window(&model, encoder_out, &ctx, &tokenizer, &device)?;
        assert!(
            emits.is_empty(),
            "no-speech gate should have returned an empty vec"
        );
        Ok(())
    }

    /// Real-model test: decode_window on tiny_en produces non-empty output for a speech clip
    /// and the text (after filtering specials) is close to the one-shot transcribe path.
    #[test]
    #[ignore = "requires tiny_en_converted in ./models/ AND test_data/LJ001-0001_16k.wav at repo root"]
    fn test_decode_window_matches_transcribe_path() -> Result<()> {
        use crate::{
            WhisperInference, WhisperTranscriber, audio::compute_mel_from_samples,
            decoding::DecodingConfig, load::load_whisper,
        };
        use burn_flex::{Flex, FlexDevice};
        use std::path::PathBuf;

        let device = FlexDevice;
        let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("models");
        // Per-model layout: `<name>/model.{mpk,cfg}` + `<name>/tokenizer.json`.
        let model_dir = models_dir.join("tiny_en_converted");
        let model_path = model_dir.join("model");
        let model_path_str = model_path.to_str().expect("valid UTF-8 model path");

        let model = load_whisper::<Flex<f32>>(model_path_str, &device)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        // Load the first 480_000 samples (30 s at 16 kHz) from the test WAV.
        let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("test_data")
            .join("LJ001-0001_16k.wav");
        let raw =
            std::fs::read(&wav_path).with_context(|| format!("read {}", wav_path.display()))?;
        // Minimal WAV reader: skip header, parse 16-bit PCM or IEEE float.
        // Reuse the same approach as streaming.rs tests.
        let samples_30s = {
            let needed = 480_000usize;
            let mut pos = 12usize;
            let mut audio_format = 1u16;
            let mut data_start = None;
            let mut data_len = 0usize;
            while pos + 8 <= raw.len() {
                let chunk_id = &raw[pos..pos + 4];
                let size = u32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap()) as usize;
                if chunk_id == b"fmt " {
                    audio_format = u16::from_le_bytes(raw[pos + 8..pos + 10].try_into().unwrap());
                } else if chunk_id == b"data" {
                    data_start = Some(pos + 8);
                    data_len = size;
                    break;
                }
                pos += 8 + size + (size & 1);
            }
            let start = data_start.context("no 'data' chunk")?;
            let end = (start + data_len).min(raw.len());
            let all: Vec<f32> = if audio_format == 3 {
                (0..(end - start) / 4)
                    .map(|i| {
                        f32::from_le_bytes(
                            raw[start + i * 4..start + i * 4 + 4].try_into().unwrap(),
                        )
                    })
                    .collect()
            } else {
                (0..(end - start) / 2)
                    .map(|i| {
                        i16::from_le_bytes(
                            raw[start + i * 2..start + i * 2 + 2].try_into().unwrap(),
                        ) as f32
                            / 32768.0
                    })
                    .collect()
            };
            let mut padded = all;
            padded.resize(needed, 0.0);
            padded
        };

        // Reference path: compute mel and transcribe with <|notimestamps|>.
        let mel = compute_mel_from_samples::<Flex<f32>>(&samples_30s, 400, 160, 80, &device)?;
        let transcriber =
            WhisperTranscriber::new(model.clone(), tokenizer.clone(), DecodingConfig::fast());
        let ref_result = transcriber.transcribe(mel.clone())?;
        let ref_text = ref_result.text.trim().to_lowercase();

        // Stream-decode path on the same encoder output.
        let encoder_out = model.forward_encoder(mel);
        let tok = |s: &str, fb: u32| tokenizer.token_to_id(s).unwrap_or(fb);
        let ctx = DecodeContext {
            prompt_tokens: &[],
            sot_token: tok("<|startoftranscript|>", 50258),
            language_token: tok("<|en|>", 50259),
            task_token: tok("<|transcribe|>", 50359),
            eot_token: tok("<|endoftext|>", 50257),
            no_speech_token: tok("<|nospeech|>", 50362),
            notimestamps_token: tok("<|notimestamps|>", 50363),
            timestamp_begin_token: 50364,
            max_new_tokens: 128,
            no_speech_threshold: 0.6,
        };

        let emits = decode_window(&model, encoder_out, &ctx, &tokenizer, &device)?;
        assert!(
            !emits.is_empty(),
            "decode_window produced no tokens for a speech clip"
        );

        // Filter to regular text tokens only and decode.
        let text_ids: Vec<u32> = emits
            .iter()
            .filter(|e| !e.is_special)
            .map(|e| e.id)
            .collect();
        assert!(
            !text_ids.is_empty(),
            "no regular text tokens in decode_window output"
        );

        let stream_text = tokenizer
            .decode(&text_ids, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .trim()
            .to_lowercase();

        assert_eq!(
            stream_text, ref_text,
            "stream_decode text diverges from one-shot path\n  stream: {stream_text:?}\n  ref:    {ref_text:?}"
        );

        Ok(())
    }
}
