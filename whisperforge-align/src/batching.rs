use std::cmp::Ordering;

use crate::AudioSegment;
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use tokenizers::Tokenizer;
use whisperforge_core::{
    DecodingConfig, HybridDecoder, KvCache, Whisper, batch_mel_spectrograms, forward_decoder_cached,
};

const VOCAB_SIZE: usize = 51864;
const EOT: u32 = 50257;

pub struct BatchedTranscriber<B: Backend> {
    model: Whisper<B>,
    tokenizer: Tokenizer,
    config: DecodingConfig,
    device: B::Device,
    batch_size: usize,
    sample_rate: u32,
}

impl<B: Backend> BatchedTranscriber<B> {
    pub fn new(
        model: Whisper<B>,
        tokenizer: Tokenizer,
        config: DecodingConfig,
        device: B::Device,
        batch_size: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            device,
            batch_size,
            sample_rate: 16_000,
        }
    }

    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Transcribe multiple audio segments in batches.
    pub fn transcribe_batch(&self, segments: &[AudioSegment]) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(segments.len());
        for chunk in segments.chunks(self.batch_size) {
            results.extend(self.process_batch(chunk)?);
        }
        Ok(results)
    }

    /// Process one batch: mel-encode all segments together, decode each sequentially.
    fn process_batch(&self, segments: &[AudioSegment]) -> Result<Vec<String>> {
        let audio_chunks: Vec<_> = segments
            .iter()
            .map(|s| s.to_audio_data(self.sample_rate))
            .collect();

        let batch_mel = batch_mel_spectrograms::<B>(&audio_chunks, 400, 160, 80, &self.device)
            .context("batch mel spectrogram")?;
        let [n_batch, n_mels, n_frames] = batch_mel.dims();
        let batch_mel = if n_frames > 3000 {
            batch_mel.slice([0..n_batch, 0..n_mels, 0..3000])
        } else {
            batch_mel
        };
        let batch_enc = self.model.forward_encoder(batch_mel); // [N, 1500, D]
        let [_, frames, d_model] = batch_enc.dims();

        let tok = |s: &str, fb: u32| self.tokenizer.token_to_id(s).unwrap_or(fb);
        let sot = tok("<|startoftranscript|>", 50258);
        let en = tok("<|en|>", 50259);
        let transcribe_tok = tok("<|transcribe|>", 50359);
        let no_timestamps = tok("<|notimestamps|>", 50363);
        let eot = tok("<|endoftext|>", EOT);
        let no_speech = tok("<|nospeech|>", 50362);
        let init_tokens = [sot, en, transcribe_tok, no_timestamps];
        let budget = self.config.max_length.saturating_sub(init_tokens.len());

        let mut results = Vec::with_capacity(segments.len());

        for i in 0..segments.len() {
            let enc_i = batch_enc.clone().slice([i..(i + 1), 0..frames, 0..d_model]);

            let mut cache = KvCache::new(&self.model, enc_i);
            let mut step_logits = Vec::new();
            for &t in &init_tokens {
                step_logits = forward_decoder_cached(&self.model, t, &mut cache, &self.device)
                    .context("kv-cache warmup")?;
            }

            let mut all_logits: Vec<Vec<f32>> = Vec::new();
            for _ in 0..budget {
                let unconstrained = step_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(eot);

                let greedy_next = if all_logits.is_empty() && unconstrained == eot {
                    step_logits
                        .iter()
                        .enumerate()
                        .filter(|&(idx, _)| idx as u32 != eot)
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(eot)
                } else {
                    unconstrained
                };

                all_logits.push(step_logits);

                if greedy_next == eot {
                    break;
                }

                step_logits =
                    forward_decoder_cached(&self.model, greedy_next, &mut cache, &self.device)
                        .context("kv-cache decode step")?;
            }

            if let Some(first) = all_logits.first_mut() {
                if (eot as usize) < first.len() {
                    first[eot as usize] = f32::NEG_INFINITY;
                }
            }

            let decoder = HybridDecoder::new(self.config.clone());
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
            results.push(text.trim().to_string());
        }

        Ok(results)
    }

    /// Heuristic: optimal batch size for current hardware.
    pub fn optimal_batch_size() -> usize {
        16
    }
}
