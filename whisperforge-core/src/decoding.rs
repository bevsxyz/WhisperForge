/// SOTA decoding strategies for Whisper transcription
/// Implements faster-whisper's hybrid beam search + temperature fallback approach
use anyhow::{Result, anyhow};
use flate2::{Compression, write::GzEncoder};
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Write;

/// Configuration for beam search decoding
#[derive(Debug, Clone)]
pub struct DecodingConfig {
    /// Beam size for beam search (1 = greedy, 5 = default, 10+ = very accurate)
    pub beam_size: usize,
    /// Temperature fallback sequence (tried in order; retry at next temp if quality fails)
    pub temperatures: Vec<f32>,
    /// Length penalty to prevent repeating short sequences (0.0 = no penalty)
    pub length_penalty: f32,
    /// Threshold for no-speech detection based on cross-attention (0.0 to 1.0)
    pub no_speech_threshold: f32,
    /// Maximum tokens to generate
    pub max_length: usize,
    /// Language token (e.g., "en" for English)
    pub language: String,
    /// Gzip compression ratio threshold — ratio above this signals a hallucination loop (default 2.4)
    pub compression_ratio_threshold: f32,
    /// Average log-probability threshold — below this signals low-confidence output (default -1.0)
    pub log_prob_threshold: f32,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            beam_size: 5,
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            length_penalty: 1.0,
            no_speech_threshold: 0.6,
            max_length: 448, // Whisper max context
            language: "en".to_string(),
            compression_ratio_threshold: 2.4,
            log_prob_threshold: -1.0,
        }
    }
}

impl DecodingConfig {
    /// Create a fast decoding config (greedy, minimal processing)
    pub fn fast() -> Self {
        Self {
            beam_size: 1,
            temperatures: vec![0.0],
            length_penalty: 0.0,
            ..Default::default()
        }
    }

    /// Create a balanced decoding config (good quality/speed tradeoff)
    pub fn balanced() -> Self {
        Self {
            beam_size: 5,
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            length_penalty: 1.0,
            ..Default::default()
        }
    }

    /// Create an accurate decoding config (highest quality, slowest)
    pub fn accurate() -> Self {
        Self {
            beam_size: 10,
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            length_penalty: 1.0,
            ..Default::default()
        }
    }

    /// Set beam size
    pub fn with_beam_size(mut self, beam_size: usize) -> Self {
        self.beam_size = beam_size.max(1);
        self
    }

    /// Override temperatures with a single value (disables fallback sequence)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperatures = vec![temperature.max(0.0)];
        self
    }

    /// Set length penalty
    pub fn with_length_penalty(mut self, penalty: f32) -> Self {
        self.length_penalty = penalty.max(0.0);
        self
    }

    /// Set no-speech threshold
    pub fn with_no_speech_threshold(mut self, threshold: f32) -> Self {
        self.no_speech_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set language
    pub fn with_language(mut self, language: String) -> Self {
        self.language = language;
        self
    }
}

// ============================================================================
// Quality metrics
// ============================================================================

/// Gzip compression ratio of `text` — `text.len() / compressed_len`.
///
/// Values above `DecodingConfig::compression_ratio_threshold` (2.4) indicate a
/// repetitive or hallucinated output.
pub fn compression_ratio(text: &str) -> f32 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(bytes).ok();
    let compressed_len = enc.finish().unwrap_or_default().len().max(1);
    bytes.len() as f32 / compressed_len as f32
}

/// Softmax probability of `token` from raw logits.
fn softmax_at(logits: &[f32], token: u32) -> f32 {
    let idx = token as usize;
    if idx >= logits.len() {
        return 0.0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    ((logits[idx] - max).exp()) / exp_sum.max(f32::EPSILON)
}

/// Log-softmax of `token` given logits scaled by `temp` (0.0 → greedy / unscaled).
fn log_softmax_at(logits: &[f32], token: u32, temp: f32) -> f32 {
    let idx = token as usize;
    if idx >= logits.len() {
        return f32::NEG_INFINITY;
    }
    let scaled: Vec<f32> = if temp > 0.0 {
        logits.iter().map(|&l| l / temp).collect()
    } else {
        logits.to_vec()
    };
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = max + scaled.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    scaled[idx] - log_sum
}

/// Sample a token from `logits` at the given temperature.
///
/// At `temp == 0.0` this is equivalent to argmax (greedy).
fn sample_from_logits(logits: &[f32], temp: f32, rng: &mut impl rand::Rng) -> u32 {
    if temp <= 0.0 || logits.is_empty() {
        return argmax_logits(logits);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| ((l - max) / temp).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(f32::EPSILON);
    let threshold: f32 = rng.r#gen::<f32>() * sum;
    let mut cumsum = 0.0;
    for (i, &e) in exps.iter().enumerate() {
        cumsum += e;
        if cumsum >= threshold {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

fn argmax_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Represents a candidate sequence during beam search
#[derive(Debug, Clone)]
struct BeamCandidate {
    /// Token IDs in this sequence
    tokens: Vec<u32>,
    /// Cumulative log probability of this sequence
    log_prob: f32,
    /// Whether this sequence has finished (reached end-of-sequence token)
    finished: bool,
    /// Number of non-padding tokens (for length normalization)
    token_count: usize,
}

impl BeamCandidate {
    /// Create a new candidate with initial token
    fn new(token: u32) -> Self {
        Self {
            tokens: vec![token],
            log_prob: 0.0,
            finished: false,
            token_count: 1,
        }
    }

    /// Calculate normalized score for ranking (higher is better)
    fn normalized_score(&self, length_penalty: f32) -> f32 {
        if self.token_count == 0 {
            return self.log_prob;
        }
        // Length-normalized log probability (prevents bias toward short sequences)
        self.log_prob / ((self.token_count as f32).powf(length_penalty))
    }
}

/// Custom ordering for candidates in binary heap (max-heap behavior)
impl PartialEq for BeamCandidate {
    fn eq(&self, other: &Self) -> bool {
        (self.normalized_score(1.0) - other.normalized_score(1.0)).abs() < 1e-6
    }
}

impl Eq for BeamCandidate {}

impl PartialOrd for BeamCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BeamCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap (higher score = higher priority)
        other
            .normalized_score(1.0)
            .partial_cmp(&self.normalized_score(1.0))
            .unwrap_or(Ordering::Equal)
    }
}

/// Beam search decoder for Whisper
pub struct BeamSearchDecoder {
    config: DecodingConfig,
}

impl BeamSearchDecoder {
    /// Create a new beam search decoder
    pub fn new(config: DecodingConfig) -> Self {
        Self { config }
    }

    /// Decode token probabilities to a sequence using beam search
    ///
    /// # Arguments
    /// * `token_probs` - Matrix of shape (seq_len, vocab_size) with log probabilities per token
    /// * `initial_token` - Starting token (usually language token or BOS)
    /// * `vocab_size` - Size of vocabulary
    /// * `eos_token` - End-of-sequence token ID
    /// * `_pad_token` - Padding token ID (reserved for future use)
    ///
    /// # Returns
    /// Vector of token IDs representing the decoded sequence
    pub fn decode(
        &self,
        token_probs: &[Vec<f32>],
        initial_token: u32,
        vocab_size: usize,
        eos_token: u32,
        _pad_token: u32,
    ) -> Result<Vec<u32>> {
        if token_probs.is_empty() {
            return Ok(vec![initial_token]);
        }

        // Validate input
        if token_probs.iter().any(|probs| probs.len() != vocab_size) {
            return Err(anyhow!("Invalid token probabilities shape"));
        }

        // Start with initial token
        let mut candidates = BinaryHeap::new();
        candidates.push(BeamCandidate::new(initial_token));

        // Process each timestep
        for step in 0..token_probs.len().min(self.config.max_length) {
            let probs = &token_probs[step];
            let mut next_candidates = Vec::new();

            // Expand each existing candidate
            for candidate in candidates.iter().take(self.config.beam_size) {
                if candidate.finished {
                    next_candidates.push(candidate.clone());
                    continue;
                }

                // Get top-k tokens for this candidate
                let top_k = self.get_top_k_tokens(probs, self.config.beam_size);

                for (token, log_prob) in top_k {
                    let mut new_candidate = candidate.clone();
                    new_candidate.tokens.push(token);
                    new_candidate.log_prob += log_prob;
                    new_candidate.token_count += 1;

                    // Check if sequence is finished
                    if token == eos_token || step == token_probs.len() - 1 {
                        new_candidate.finished = true;
                    }

                    next_candidates.push(new_candidate);
                }
            }

            // Keep only top beam_size candidates
            next_candidates.sort_by(|a, b| {
                b.normalized_score(self.config.length_penalty)
                    .partial_cmp(&a.normalized_score(self.config.length_penalty))
                    .unwrap_or(Ordering::Equal)
            });

            candidates = next_candidates
                .into_iter()
                .take(self.config.beam_size)
                .collect::<BinaryHeap<_>>();

            // Early exit if all candidates are finished
            if candidates.iter().all(|c| c.finished) {
                break;
            }
        }

        // Return best candidate
        candidates
            .iter()
            .max_by(|a, b| {
                a.normalized_score(self.config.length_penalty)
                    .partial_cmp(&b.normalized_score(self.config.length_penalty))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|c| c.tokens.clone())
            .ok_or_else(|| anyhow!("No valid candidates found"))
    }

    /// Get top-k tokens from log probability distribution
    fn get_top_k_tokens(&self, log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut indexed: Vec<(u32, f32)> = log_probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i as u32, prob))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        indexed.into_iter().take(k).collect()
    }
}

/// Greedy decoder (baseline, fastest)
pub struct GreedyDecoder;

impl GreedyDecoder {
    /// Decode using greedy approach (always take highest probability token)
    pub fn decode(
        token_probs: &[Vec<f32>],
        initial_token: u32,
        _vocab_size: usize,
        eos_token: u32,
        _pad_token: u32,
    ) -> Result<Vec<u32>> {
        let mut tokens = vec![initial_token];

        for probs in token_probs {
            if probs.is_empty() {
                break;
            }

            let (token, _) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .unwrap_or((eos_token as usize, &f32::NEG_INFINITY));

            let token = token as u32;
            tokens.push(token);

            if token == eos_token {
                break;
            }
        }

        Ok(tokens)
    }
}

/// Multi-hypothesis decoding strategy that tries multiple approaches
pub struct HybridDecoder {
    config: DecodingConfig,
    beam_decoder: BeamSearchDecoder,
}

impl HybridDecoder {
    /// Create a new hybrid decoder
    pub fn new(config: DecodingConfig) -> Self {
        Self {
            beam_decoder: BeamSearchDecoder::new(config.clone()),
            config,
        }
    }

    /// Decode with beam-search / greedy fallback (no quality gating).
    pub fn decode(
        &self,
        token_probs: &[Vec<f32>],
        initial_token: u32,
        vocab_size: usize,
        eos_token: u32,
        pad_token: u32,
    ) -> Result<Vec<u32>> {
        match self
            .beam_decoder
            .decode(token_probs, initial_token, vocab_size, eos_token, pad_token)
        {
            Ok(tokens) if tokens.len() > 1 => Ok(tokens),
            _ => {
                GreedyDecoder::decode(token_probs, initial_token, vocab_size, eos_token, pad_token)
            }
        }
    }

    /// Decode with quality-gated temperature fallback (faster-whisper SOTA strategy).
    ///
    /// Iterates over `config.temperatures` in order. At each temperature, samples
    /// tokens from the collected logits and checks:
    /// - `no_speech_prob > no_speech_threshold` → return empty (silence)
    /// - `avg_log_prob < log_prob_threshold` → retry at next temperature
    /// - `compression_ratio > compression_ratio_threshold` → retry at next temperature
    ///
    /// Returns the first result that passes all quality gates, or the last attempt
    /// if all temperatures are exhausted.
    ///
    /// # Arguments
    /// * `token_probs` — per-step raw logits collected autoregressively from the decoder
    /// * `initial_token` — first token to prepend to the output sequence
    /// * `vocab_size` — vocabulary size (bounds check for `no_speech_token`)
    /// * `eos_token` — end-of-sequence token; stops generation when sampled
    /// * `no_speech_token` — token whose first-step softmax probability signals silence
    /// * `decode_text` — closure that converts token IDs to a UTF-8 string for compression ratio
    pub fn decode_with_fallback(
        &self,
        token_probs: &[Vec<f32>],
        initial_token: u32,
        vocab_size: usize,
        eos_token: u32,
        no_speech_token: u32,
        decode_text: impl Fn(&[u32]) -> String,
    ) -> Result<Vec<u32>> {
        if token_probs.is_empty() {
            return Ok(vec![initial_token]);
        }

        // No-speech check on first decode step (independent of temperature).
        if (no_speech_token as usize) < vocab_size {
            let ns_prob = softmax_at(&token_probs[0], no_speech_token);
            if ns_prob > self.config.no_speech_threshold {
                return Ok(vec![]);
            }
        }

        let mut best: Option<Vec<u32>> = None;

        for &temp in &self.config.temperatures {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let mut tokens = vec![initial_token];
            let mut log_probs: Vec<f32> = Vec::new();

            for step_logits in token_probs.iter().take(self.config.max_length) {
                let selected = sample_from_logits(step_logits, temp, &mut rng);
                log_probs.push(log_softmax_at(step_logits, selected, temp));
                tokens.push(selected);
                if selected == eos_token {
                    break;
                }
            }

            let avg_lp = if log_probs.is_empty() {
                0.0
            } else {
                log_probs.iter().sum::<f32>() / log_probs.len() as f32
            };

            let text = decode_text(&tokens);
            let cr = compression_ratio(&text);

            let quality_ok = avg_lp > self.config.log_prob_threshold
                && cr < self.config.compression_ratio_threshold;

            if best.is_none() {
                best = Some(tokens.clone());
            }

            if quality_ok {
                return Ok(tokens);
            }
        }

        best.ok_or_else(|| anyhow!("decode_with_fallback: no temperatures configured"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoding_config_defaults() {
        let config = DecodingConfig::default();
        assert_eq!(config.beam_size, 5);
        assert_eq!(config.temperatures, vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        assert_eq!(config.language, "en");
    }

    #[test]
    fn test_decoding_config_fast() {
        let config = DecodingConfig::fast();
        assert_eq!(config.beam_size, 1);
        assert_eq!(config.temperatures, vec![0.0]);
    }

    #[test]
    fn test_decoding_config_accurate() {
        let config = DecodingConfig::accurate();
        assert_eq!(config.beam_size, 10);
        assert_eq!(config.temperatures, vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    }

    #[test]
    fn test_with_temperature_overrides_sequence() {
        let config = DecodingConfig::default().with_temperature(0.7);
        assert_eq!(config.temperatures, vec![0.7]);
    }

    #[test]
    fn test_beam_candidate_scoring() {
        let mut c1 = BeamCandidate::new(1);
        c1.log_prob = -4.0;
        c1.token_count = 2;

        let mut c2 = BeamCandidate::new(2);
        c2.log_prob = -1.0;
        c2.token_count = 1;

        // c2 has better (higher) normalized score: -1.0 / 1^1 = -1.0 > -4.0 / 2^1 = -2.0
        assert!(c2.normalized_score(1.0) > c1.normalized_score(1.0));
    }

    #[test]
    fn test_greedy_decoder() -> Result<()> {
        // Create simple token probabilities (logits)
        let token_probs = vec![
            vec![-10.0, -5.0, -0.5, -10.0], // Token 2 is most likely
            vec![-5.0, -10.0, -0.1, -10.0], // Token 2 is most likely
            vec![-0.5, -5.0, -10.0, -10.0], // Token 0 is most likely (EOS)
        ];

        let tokens = GreedyDecoder::decode(&token_probs, 50256, 4, 0, 50257)?;

        assert_eq!(tokens.len(), 4); // Initial + 3 timesteps
        assert_eq!(tokens[0], 50256); // Initial token
        assert_eq!(tokens[1], 2); // Most likely at t=0
        assert_eq!(tokens[2], 2); // Most likely at t=1
        assert_eq!(tokens[3], 0); // EOS at t=2

        Ok(())
    }

    #[test]
    fn test_beam_search_decoder() -> Result<()> {
        let config = DecodingConfig {
            beam_size: 2,
            ..Default::default()
        };

        let decoder = BeamSearchDecoder::new(config);

        let token_probs = vec![
            vec![-5.0, -0.5, -10.0], // Token 1 is most likely
            vec![-0.1, -5.0, -10.0], // Token 0 is most likely
        ];

        let tokens = decoder.decode(&token_probs, 100, 3, 0, 99)?;

        assert!(tokens.len() >= 2);
        assert_eq!(tokens[0], 100); // Initial token

        Ok(())
    }

    #[test]
    fn test_hybrid_decoder_fallback() -> Result<()> {
        let config = DecodingConfig::default();
        let decoder = HybridDecoder::new(config);

        let token_probs = vec![vec![-0.5, -10.0, -10.0]];

        let tokens = decoder.decode(&token_probs, 100, 3, 0, 99)?;

        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], 100);

        Ok(())
    }

    #[test]
    fn test_compression_ratio_normal_text() {
        // Prose compresses to ratio < 2.4 (not a hallucination loop).
        let text = "The quick brown fox jumps over the lazy dog.";
        let cr = compression_ratio(text);
        assert!(cr < 2.4, "normal text compression ratio was {cr}");
    }

    #[test]
    fn test_compression_ratio_repetitive_text() {
        // Hallucination loops repeat the same phrase hundreds of times.
        // Gzip header overhead is ~20 bytes, so the input must be long enough
        // for the repetition signal to dominate.
        let phrase = "the quick brown fox ";
        let text = phrase.repeat(100); // 2000 chars, highly compressible
        let cr = compression_ratio(&text);
        assert!(cr > 2.4, "repetitive text compression ratio was {cr}");
    }

    #[test]
    fn test_compression_ratio_empty() {
        assert_eq!(compression_ratio(""), 0.0);
    }

    #[test]
    fn test_softmax_at_picks_max() {
        let logits = vec![-10.0, -0.1, -5.0];
        let p_max = softmax_at(&logits, 1);
        let p_min = softmax_at(&logits, 0);
        assert!(p_max > p_min, "softmax of max logit should be highest");
        let total: f32 = (0..3).map(|i| softmax_at(&logits, i)).sum();
        assert!((total - 1.0).abs() < 1e-4, "softmax probs must sum to 1");
    }

    #[test]
    fn test_decode_with_fallback_passes_quality() -> Result<()> {
        // Token 1 dominates every step → log prob near 0, repetitive text expected.
        // Use a very lax threshold so it passes immediately.
        let config = DecodingConfig {
            temperatures: vec![0.0],
            log_prob_threshold: -100.0,
            compression_ratio_threshold: 100.0,
            no_speech_threshold: 1.0,
            max_length: 5,
            ..Default::default()
        };
        let decoder = HybridDecoder::new(config);

        let token_probs = vec![vec![-0.01, -10.0, -10.0], vec![-0.01, -10.0, -10.0]];

        let tokens = decoder.decode_with_fallback(
            &token_probs,
            99,
            3,
            0, // eos
            2, // no_speech_token (low prob → not triggered)
            |ids| {
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            },
        )?;

        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], 99);
        Ok(())
    }

    #[test]
    fn test_decode_with_fallback_no_speech() -> Result<()> {
        // Make no_speech_token dominant at step 0 → should return empty Vec.
        let config = DecodingConfig {
            temperatures: vec![0.0],
            no_speech_threshold: 0.5,
            log_prob_threshold: -100.0,
            compression_ratio_threshold: 100.0,
            max_length: 5,
            ..Default::default()
        };
        let decoder = HybridDecoder::new(config);

        // Token 1 has very high logit → softmax ≈ 1.0, well above threshold 0.5.
        let token_probs = vec![vec![-10.0, 100.0, -10.0]];

        let tokens = decoder.decode_with_fallback(
            &token_probs,
            99,
            3,
            0, // eos
            1, // no_speech_token = token 1 (dominant)
            |_| String::new(),
        )?;

        assert!(
            tokens.is_empty(),
            "should return empty when no-speech detected"
        );
        Ok(())
    }
}
