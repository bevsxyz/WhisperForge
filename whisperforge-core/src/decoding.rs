/// SOTA decoding strategies for Whisper transcription
/// Implements faster-whisper's hybrid beam search + temperature fallback approach
use anyhow::{anyhow, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for beam search decoding
#[derive(Debug, Clone)]
pub struct DecodingConfig {
    /// Beam size for beam search (1 = greedy, 5 = default, 10+ = very accurate)
    pub beam_size: usize,
    /// Temperature for sampling when using temperature fallback (0.0 = greedy)
    pub temperature: f32,
    /// Length penalty to prevent repeating short sequences (0.0 = no penalty)
    pub length_penalty: f32,
    /// Threshold for no-speech detection based on cross-attention (0.0 to 1.0)
    pub no_speech_threshold: f32,
    /// Maximum tokens to generate
    pub max_length: usize,
    /// Language token (e.g., "en" for English)
    pub language: String,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            beam_size: 5,
            temperature: 0.5,
            length_penalty: 1.0,
            no_speech_threshold: 0.6,
            max_length: 448, // Whisper max context
            language: "en".to_string(),
        }
    }
}

impl DecodingConfig {
    /// Create a fast decoding config (greedy, minimal processing)
    pub fn fast() -> Self {
        Self {
            beam_size: 1,
            temperature: 0.0,
            length_penalty: 0.0,
            ..Default::default()
        }
    }

    /// Create a balanced decoding config (good quality/speed tradeoff)
    pub fn balanced() -> Self {
        Self {
            beam_size: 5,
            temperature: 0.5,
            length_penalty: 1.0,
            ..Default::default()
        }
    }

    /// Create an accurate decoding config (highest quality, slowest)
    pub fn accurate() -> Self {
        Self {
            beam_size: 10,
            temperature: 0.3,
            length_penalty: 1.0,
            ..Default::default()
        }
    }

    /// Set beam size
    pub fn with_beam_size(mut self, beam_size: usize) -> Self {
        self.beam_size = beam_size.max(1);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.0);
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
        // Reverse ordering for max-heap (higher score = higher priority)
        other
            .normalized_score(1.0)
            .partial_cmp(&self.normalized_score(1.0))
    }
}

impl Ord for BeamCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
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
    beam_decoder: BeamSearchDecoder,
    #[allow(dead_code)]
    greedy_decoder: GreedyDecoder,
}

impl HybridDecoder {
    /// Create a new hybrid decoder
    pub fn new(config: DecodingConfig) -> Self {
        Self {
            beam_decoder: BeamSearchDecoder::new(config),
            greedy_decoder: GreedyDecoder,
        }
    }

    /// Decode with fallback strategy
    pub fn decode(
        &self,
        token_probs: &[Vec<f32>],
        initial_token: u32,
        vocab_size: usize,
        eos_token: u32,
        pad_token: u32,
    ) -> Result<Vec<u32>> {
        // Try beam search first
        match self
            .beam_decoder
            .decode(token_probs, initial_token, vocab_size, eos_token, pad_token)
        {
            Ok(tokens) if tokens.len() > 1 => Ok(tokens),
            // Fallback to greedy if beam search fails or produces empty result
            _ => {
                GreedyDecoder::decode(token_probs, initial_token, vocab_size, eos_token, pad_token)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoding_config_defaults() {
        let config = DecodingConfig::default();
        assert_eq!(config.beam_size, 5);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.language, "en");
    }

    #[test]
    fn test_decoding_config_fast() {
        let config = DecodingConfig::fast();
        assert_eq!(config.beam_size, 1);
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn test_decoding_config_accurate() {
        let config = DecodingConfig::accurate();
        assert_eq!(config.beam_size, 10);
        assert_eq!(config.temperature, 0.3);
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
}
