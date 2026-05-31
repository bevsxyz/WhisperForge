//! Whisper language + task token helpers.
//!
//! Whisper's decoder is seeded with `[<|sot|>, <|lang|>, <|task|>, <|notimestamps|>]`.
//! The language token selects the spoken language (transcription) or source language
//! (translation); the task token selects `transcribe` (output in the spoken language)
//! or `translate` (output in **English only** — Whisper has no other-target path).
//!
//! This module turns CLI strings (`--language hi`, `--task translate`) into the
//! corresponding token ids via the model tokenizer, and provides Whisper's
//! first-token language auto-detection.

use anyhow::{Context, Result};
use burn::tensor::{Tensor, backend::Backend};
use tokenizers::Tokenizer;

use crate::kv_cache::{KvCache, forward_decoder_cached};
use crate::model::Whisper;

/// The ~99 Whisper language codes, in token-id order (mirrors the language span of
/// [`crate::SPECIAL_TOKENS`], i.e. everything between `<|en|>` and `<|notimestamps|>`).
pub const LANGUAGE_CODES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Whisper decode task. `Translate` is **X → English only** — Whisper cannot emit any
/// other target language (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    #[default]
    Transcribe,
    Translate,
}

/// Token id for a language code (e.g. `"hi"` → `<|hi|>`), or `None` if the tokenizer
/// lacks it (the signature of an English-only `.en` model).
pub fn language_token_id(tokenizer: &Tokenizer, code: &str) -> Option<u32> {
    tokenizer.token_to_id(&format!("<|{code}|>"))
}

/// Token id for the task token (`<|transcribe|>` / `<|translate|>`).
pub fn task_token_id(tokenizer: &Tokenizer, task: Task) -> Option<u32> {
    let s = match task {
        Task::Transcribe => "<|transcribe|>",
        Task::Translate => "<|translate|>",
    };
    tokenizer.token_to_id(s)
}

/// Whisper first-token language detection.
///
/// Runs a single decoder step seeded with `<|sot|>` only, then takes the argmax of the
/// resulting logits **restricted to the language token ids** (an unrestricted argmax
/// would return a content token, not a language). Returns `(code, token_id)`.
///
/// `encoder_out` is consumed by [`KvCache::new`]; clone before calling if you need it again.
pub fn detect_language<B: Backend>(
    model: &Whisper<B>,
    encoder_out: Tensor<B, 3>,
    tokenizer: &Tokenizer,
    sot_token: u32,
    device: &B::Device,
) -> Result<(String, u32)> {
    let mut cache = KvCache::new(model, encoder_out);
    let logits = forward_decoder_cached(model, sot_token, &mut cache, device)
        .context("language-detection forward pass")?;

    let mut best: Option<(f32, u32, &str)> = None;
    for &code in LANGUAGE_CODES {
        let Some(id) = language_token_id(tokenizer, code) else {
            continue;
        };
        let Some(&logit) = logits.get(id as usize) else {
            continue;
        };
        if best.is_none_or(|(b, _, _)| logit > b) {
            best = Some((logit, id, code));
        }
    }

    let (_, id, code) = best.context(
        "no language tokens found in tokenizer — language auto-detection requires a \
         multilingual model (English-only .en models cannot detect language)",
    )?;
    Ok((code.to_string(), id))
}
