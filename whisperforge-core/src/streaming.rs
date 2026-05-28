use std::collections::VecDeque;

use crate::stream_decode::TokenEmit;
use crate::vad_silero::SileroVad;

const SAMPLE_RATE: u32 = 16_000;
const VAD_FRAME_SIZE: usize = 512;

pub struct WindowConfig {
    /// Hard cap on the audio buffer; once reached, the chunker forces an EOU and the
    /// consumer should call [`Chunker::reset_utterance`]. Default 28.0 s (leaves headroom
    /// under Whisper's 30 s decoder context window).
    pub max_window_secs: f32,
    /// How often to re-decode the growing buffer (default 1.0 s).
    pub stride_secs: f32,
    pub vad_threshold: f32,
    /// Minimum accumulated speech before the first window of an utterance fires
    /// (debounces VAD flicker).
    pub min_speech_secs: f32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            max_window_secs: 28.0,
            stride_secs: 1.0,
            vad_threshold: 0.5,
            min_speech_secs: 0.25,
        }
    }
}

pub struct StreamWindow {
    /// Exactly `max_window_secs × 16 000` samples; zero-padded at the BACK when the
    /// growing buffer hasn't reached `max_window_secs` yet.
    pub samples: Vec<f32>,
    /// Number of real (non-padding) samples in `samples`.
    pub real_samples: usize,
    /// Wall-clock offset since stream start of the first sample in the current utterance.
    pub window_start_secs: f32,
    /// True if any VAD-positive frame falls within the current utterance buffer.
    pub had_speech: bool,
    /// Seconds since the last VAD-positive frame; for use by the endpointer.
    pub trailing_silence_secs: f32,
    /// True when the buffer hit the `max_window_secs` cap; the consumer should treat this
    /// as a forced EOU (finalize + `reset_utterance`) to avoid blowing the decoder context.
    pub forced_eou: bool,
}

pub struct Chunker {
    cfg: WindowConfig,
    vad: SileroVad,
    /// Growing audio buffer for the current utterance; never exceeds `max_window_secs * 16 kHz`.
    buf: Vec<f32>,
    pub samples_since_last_stride: usize,
    pub total_samples_seen: u64,
    pub last_speech_at_sample: Option<u64>,

    // --- private implementation details ---

    // Speech samples accumulated within the current utterance; gates the first stride.
    speech_samples_accumulated: u64,
    // Partial accumulation buffer for in-progress VAD frames.
    vad_frame_buf: Vec<f32>,
    // Once the first stride of the current utterance has fired, subsequent strides
    // fire even during silence (so the endpointer can see growing trailing silence).
    window_ever_emitted: bool,
    // VAD decisions for the current utterance's audio buffer.
    vad_decisions: VecDeque<bool>,
    // Sample index (in stream-relative terms) where the current utterance's audio[0] sits.
    utterance_start_sample: u64,
}

impl Chunker {
    pub fn new(cfg: WindowConfig, vad: SileroVad) -> Self {
        let max_samples = (cfg.max_window_secs * SAMPLE_RATE as f32) as usize;
        let max_vad_frames = max_samples.div_ceil(VAD_FRAME_SIZE);
        Self {
            cfg,
            vad,
            buf: Vec::with_capacity(max_samples),
            samples_since_last_stride: 0,
            total_samples_seen: 0,
            last_speech_at_sample: None,
            speech_samples_accumulated: 0,
            vad_frame_buf: Vec::with_capacity(VAD_FRAME_SIZE),
            window_ever_emitted: false,
            vad_decisions: VecDeque::with_capacity(max_vad_frames),
            utterance_start_sample: 0,
        }
    }

    /// Push new microphone samples; returns `Some(window)` when a stride boundary fires
    /// or the buffer hits the `max_window_secs` cap (`forced_eou == true` in that case).
    ///
    /// The buffer GROWS — it doesn't slide. Consumers must call [`reset_utterance`] after
    /// committing/finalising an utterance so the next stride starts a fresh buffer.
    pub fn push(&mut self, samples: &[f32]) -> Option<StreamWindow> {
        let stride_samples = (self.cfg.stride_secs * SAMPLE_RATE as f32) as usize;
        let max_samples = (self.cfg.max_window_secs * SAMPLE_RATE as f32) as usize;
        let min_speech_samples = (self.cfg.min_speech_secs * SAMPLE_RATE as f32) as u64;
        let max_vad_frames = max_samples.div_ceil(VAD_FRAME_SIZE);

        let mut result: Option<StreamWindow> = None;

        for &s in samples {
            if self.buf.len() < max_samples {
                self.buf.push(s);
            }
            self.total_samples_seen += 1;
            self.samples_since_last_stride += 1;

            // Accumulate into the current VAD frame.
            self.vad_frame_buf.push(s);
            if self.vad_frame_buf.len() == VAD_FRAME_SIZE {
                let frame: &[f32; VAD_FRAME_SIZE] = self
                    .vad_frame_buf
                    .as_slice()
                    .try_into()
                    .expect("vad_frame_buf is exactly VAD_FRAME_SIZE");
                let is_speech = self
                    .vad
                    .probability(frame)
                    .map(|p| p >= self.cfg.vad_threshold)
                    .unwrap_or(false);
                self.vad_frame_buf.clear();

                if is_speech {
                    self.last_speech_at_sample = Some(self.total_samples_seen);
                    self.speech_samples_accumulated += VAD_FRAME_SIZE as u64;
                }

                if self.vad_decisions.len() >= max_vad_frames {
                    self.vad_decisions.pop_front();
                }
                self.vad_decisions.push_back(is_speech);
            }

            // Force-EOU if we've hit the cap; consumer must handle and call reset_utterance.
            if self.buf.len() >= max_samples {
                let mut w = self.build_window(max_samples);
                w.forced_eou = true;
                result = Some(w);
                self.samples_since_last_stride = 0;
                continue;
            }

            // Regular stride tick.
            if self.samples_since_last_stride >= stride_samples {
                let enough_speech = self.speech_samples_accumulated >= min_speech_samples;
                if enough_speech || self.window_ever_emitted {
                    result = Some(self.build_window(max_samples));
                    self.window_ever_emitted = true;
                }
                self.samples_since_last_stride = 0;
            }
        }

        result
    }

    fn build_window(&self, max_samples: usize) -> StreamWindow {
        let real_samples = self.buf.len();
        let mut samples = Vec::with_capacity(max_samples);
        samples.extend_from_slice(&self.buf);
        samples.resize(max_samples, 0.0);

        let trailing_silence_secs = match self.last_speech_at_sample {
            Some(last) => self.total_samples_seen.saturating_sub(last) as f32 / SAMPLE_RATE as f32,
            None => {
                self.total_samples_seen
                    .saturating_sub(self.utterance_start_sample) as f32
                    / SAMPLE_RATE as f32
            }
        };

        let window_start_secs = self.utterance_start_sample as f32 / SAMPLE_RATE as f32;
        let had_speech = self.vad_decisions.iter().any(|&b| b);

        StreamWindow {
            samples,
            real_samples,
            window_start_secs,
            had_speech,
            trailing_silence_secs,
            forced_eou: false,
        }
    }

    /// Drop the current utterance buffer so the next stride starts fresh. Call after
    /// `committer.finalize_utterance()` (either natural EOU or `forced_eou`). Wall-clock
    /// counters and the last-speech timestamp persist so trailing-silence math stays valid
    /// across utterance boundaries.
    pub fn reset_utterance(&mut self) {
        self.buf.clear();
        self.samples_since_last_stride = 0;
        self.vad_decisions.clear();
        self.speech_samples_accumulated = 0;
        self.window_ever_emitted = false;
        self.utterance_start_sample = self.total_samples_seen;
    }
}

/// Outcome of one `Committer::ingest` or `finalize_utterance` call.
pub enum CommitDelta {
    /// Newly stable tokens since the previous call. May be empty.
    Committed {
        new_tokens: Vec<TokenEmit>,
        new_text: String,
    },
    /// Current tentative tail (everything after the committed prefix). Updated every round.
    Tentative {
        tokens: Vec<TokenEmit>,
        text: String,
    },
}

/// LocalAgreement-2 token committer.
///
/// Compares successive window decode outputs and commits the longest stable
/// prefix; everything past that prefix is tentative until the next round confirms it.
pub struct Committer {
    last_candidate: Vec<TokenEmit>,
    committed: Vec<TokenEmit>,
    committed_text: String,
    /// Set by `finalize_utterance`; the next `ingest` auto-resets the committer's per-
    /// utterance state. This lets the caller read `committed_tokens` / `committed_text`
    /// between finalize and the next ingest (e.g. for the `<|prevtext|>` prompt prefix)
    /// without smearing the previous utterance into the next one.
    awaiting_reset: bool,
}

impl Committer {
    pub fn new() -> Self {
        Self {
            last_candidate: Vec::new(),
            committed: Vec::new(),
            committed_text: String::new(),
            awaiting_reset: false,
        }
    }

    /// Ingest a new full-window decode result.
    ///
    /// Returns `(committed_delta, tentative_delta)`.
    pub fn ingest(&mut self, candidate: Vec<TokenEmit>) -> (CommitDelta, CommitDelta) {
        if self.awaiting_reset {
            self.last_candidate.clear();
            self.committed.clear();
            self.committed_text.clear();
            self.awaiting_reset = false;
        }
        let lcp = lcp_len(&self.last_candidate, &candidate);
        let prev_committed_len = self.committed.len();

        // Commit candidate[prev_committed_len..lcp] when lcp has advanced past what we've
        // already committed. If lcp < prev_committed_len the new decode regressed on
        // committed tokens — committed tokens are final, so we simply don't extend.
        if lcp > prev_committed_len {
            for t in candidate[prev_committed_len..lcp].iter() {
                self.committed.push(t.clone());
            }
        }

        let new_text = build_text(&self.committed[prev_committed_len..]);
        self.committed_text.push_str(&new_text);

        let new_tokens: Vec<TokenEmit> = self.committed[prev_committed_len..].to_vec();
        let tentative_tokens: Vec<TokenEmit> = candidate[lcp..].to_vec();
        let tentative_text = build_text(&tentative_tokens);

        self.last_candidate = candidate;

        (
            CommitDelta::Committed {
                new_tokens,
                new_text,
            },
            CommitDelta::Tentative {
                tokens: tentative_tokens,
                text: tentative_text,
            },
        )
    }

    /// Force-commit everything tentative. Called by the endpointer on EOU.
    ///
    /// After this call, `committed_tokens` / `committed_text` still reflect the just-
    /// finalised utterance (so the caller can build a `<|prevtext|>` prompt from it).
    /// The next `ingest` will auto-reset the committer for the new utterance. Repeated
    /// `finalize_utterance` calls without an intervening `ingest` are a no-op.
    pub fn finalize_utterance(&mut self) -> CommitDelta {
        let start = self.committed.len();
        let new_tokens: Vec<TokenEmit> = if start < self.last_candidate.len() {
            self.last_candidate[start..].to_vec()
        } else {
            Vec::new()
        };
        let new_text = build_text(&new_tokens);
        self.committed_text.push_str(&new_text);
        self.committed.extend(new_tokens.iter().cloned());
        self.last_candidate.clear();
        self.awaiting_reset = true;
        CommitDelta::Committed {
            new_tokens,
            new_text,
        }
    }

    pub fn committed_tokens(&self) -> &[TokenEmit] {
        &self.committed
    }

    pub fn committed_text(&self) -> &str {
        &self.committed_text
    }
}

impl Default for Committer {
    fn default() -> Self {
        Self::new()
    }
}

/// Join text pieces for non-special tokens.
fn build_text(tokens: &[TokenEmit]) -> String {
    tokens
        .iter()
        .filter(|t| !t.is_special)
        .map(|t| t.text.as_str())
        .collect()
}

/// Length of the longest common non-timestamp-token prefix between `a` and `b`,
/// expressed as an exclusive end index in `b`.
///
/// Timestamp tokens (`window_ts_secs.is_some()`) are skipped in both sequences
/// so that differing timestamp values between overlapping windows don't break the match.
fn lcp_len(a: &[TokenEmit], b: &[TokenEmit]) -> usize {
    let a_content: Vec<(usize, u32)> = a
        .iter()
        .enumerate()
        .filter(|(_, t)| t.window_ts_secs.is_none())
        .map(|(i, t)| (i, t.id))
        .collect();
    let b_content: Vec<(usize, u32)> = b
        .iter()
        .enumerate()
        .filter(|(_, t)| t.window_ts_secs.is_none())
        .map(|(i, t)| (i, t.id))
        .collect();

    let lcp_n = a_content
        .iter()
        .zip(b_content.iter())
        .take_while(|((_, aid), (_, bid))| aid == bid)
        .count();

    if lcp_n == 0 {
        return 0;
    }
    b_content[lcp_n - 1].0 + 1
}

pub struct EndpointConfig {
    /// Hard EOU: minimum trailing silence (seconds) after which an utterance is ended (default 2.0).
    pub silence_secs: f32,
    /// Soft EOU: minimum trailing silence after a terminal punctuation mark (default 0.8).
    pub punct_silence_secs: f32,
    /// Suppress EOU if the utterance has been running for less than this many seconds (default 0.5).
    pub min_utterance_secs: f32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            silence_secs: 2.0,
            punct_silence_secs: 0.8,
            min_utterance_secs: 0.5,
        }
    }
}

/// Hybrid silence + punctuation end-of-utterance detector.
///
/// Call `step` after each chunker + committer round. When it returns `true`, fire the EOU
/// event, then call `reset` before the next utterance.
pub struct Endpointer {
    cfg: EndpointConfig,
    /// Byte-length of committed text when this utterance baseline was captured.
    text_len_at_reset: usize,
    /// Prevents re-firing before the caller calls `reset`.
    fired: bool,
    /// Set by `reset`; the next `step` call captures the current committed-text length
    /// as the new baseline (avoids requiring committed text as a parameter to `reset`).
    needs_baseline_update: bool,
    /// Wall-clock second when first new committed text appeared after the last reset.
    utterance_start_secs: Option<f32>,
}

impl Endpointer {
    pub fn new(cfg: EndpointConfig) -> Self {
        Self {
            cfg,
            text_len_at_reset: 0,
            fired: false,
            needs_baseline_update: false,
            utterance_start_secs: None,
        }
    }

    /// Called after each chunker tick + committer round. Returns `true` if EOU should fire.
    pub fn step(&mut self, window: &StreamWindow, latest_committed_text: &str) -> bool {
        if self.fired {
            return false;
        }

        if self.needs_baseline_update {
            self.text_len_at_reset = latest_committed_text.len();
            self.needs_baseline_update = false;
        }

        let has_new_committed = latest_committed_text.len() > self.text_len_at_reset;

        if has_new_committed && self.utterance_start_secs.is_none() {
            self.utterance_start_secs = Some(window.window_start_secs);
        }

        let current_secs =
            window.window_start_secs + window.real_samples as f32 / SAMPLE_RATE as f32;
        let utterance_secs = self
            .utterance_start_secs
            .map(|start| current_secs - start)
            .unwrap_or(0.0);
        if utterance_secs < self.cfg.min_utterance_secs {
            return false;
        }

        if has_new_committed && window.trailing_silence_secs >= self.cfg.silence_secs {
            self.fired = true;
            return true;
        }

        if has_new_committed {
            let new_text = &latest_committed_text[self.text_len_at_reset..];
            if matches!(new_text.chars().last(), Some('.') | Some('!') | Some('?'))
                && window.trailing_silence_secs >= self.cfg.punct_silence_secs
            {
                self.fired = true;
                return true;
            }
        }

        false
    }

    /// Reset state after an EOU has been handled. The next `step` call will capture the
    /// current committed-text length as the baseline for the new utterance.
    pub fn reset(&mut self) {
        self.fired = false;
        self.utterance_start_secs = None;
        self.needs_baseline_update = true;
    }
}

/// Manages the cross-utterance prompt prefix fed to the streaming decoder.
///
/// Call `update_after_eou` immediately after an EOU fires, passing the full committed
/// token list from `Committer::committed_tokens`. Then pass `prompt_tokens()` into
/// `DecodeContext::prompt_tokens` for every window until the next EOU.
pub struct PromptContext {
    /// Maximum number of committed regular-token IDs to carry forward (default 60).
    max_prompt_tokens: usize,
    /// The `<|prevtext|>` token ID, or `None` for models that lack it (e.g. tiny.en).
    prevtext_token_id: Option<u32>,
    /// Prompt built after the last EOU; empty before the first EOU.
    current_prompt: Vec<u32>,
}

impl PromptContext {
    /// `prevtext_token_id`: result of `tokenizer.token_to_id("<|prevtext|>")`.
    /// Pass `None` for English-only models that lack `<|prevtext|>` in their vocabulary.
    pub fn new(max_prompt_tokens: usize, prevtext_token_id: Option<u32>) -> Self {
        Self {
            max_prompt_tokens,
            prevtext_token_id,
            current_prompt: Vec::new(),
        }
    }

    /// Update the stored prompt after an EOU.
    ///
    /// Extracts the last `max_prompt_tokens` regular-token IDs (filter: `!is_special`),
    /// prepends `<|prevtext|>` when available and the tail is non-empty, and caps the total
    /// so that `prompt.len() + sot(1) + lang(1) + transcribe(1) + max_new_tokens ≤ 448`.
    pub fn update_after_eou(&mut self, committed_tokens: &[TokenEmit], max_new_tokens: usize) {
        // Whisper's decoder context window is 448 tokens.
        // 3 slots are reserved for the sot + language + transcribe init tokens.
        let max_allowed = 448usize.saturating_sub(3 + max_new_tokens);
        // Reserve one additional slot for the <|prevtext|> prefix if it will be emitted.
        let text_slots = if self.prevtext_token_id.is_some() {
            max_allowed.saturating_sub(1)
        } else {
            max_allowed
        };
        let cap = self.max_prompt_tokens.min(text_slots);

        // Collect regular (non-special) token IDs — same predicate as build_text.
        let regular_ids: Vec<u32> = committed_tokens
            .iter()
            .filter(|t| !t.is_special)
            .map(|t| t.id)
            .collect();

        let start = regular_ids.len().saturating_sub(cap);
        let tail = &regular_ids[start..];

        self.current_prompt.clear();
        // Only emit <|prevtext|> when there are actual text tokens to accompany it.
        if let Some(prevtext) = self.prevtext_token_id {
            if !tail.is_empty() {
                self.current_prompt.push(prevtext);
            }
        }
        self.current_prompt.extend_from_slice(tail);
    }

    /// Returns the prompt tokens to pass into `DecodeContext::prompt_tokens`.
    /// Empty before the first EOU or after `reset`.
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.current_prompt
    }

    /// Clear the stored prompt (call on stream restart or explicit reset).
    pub fn reset(&mut self) {
        self.current_prompt.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad_silero::ensure_silero_model;
    use anyhow::Result;
    use std::path::PathBuf;

    fn models_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("models")
    }

    fn open_vad() -> Result<SileroVad> {
        let path = ensure_silero_model(&models_dir())?;
        SileroVad::open(&path)
    }

    /// Read all f32 samples from a 16-bit PCM or IEEE float32 WAV, returning raw f32.
    fn read_wav_samples(path: &std::path::Path) -> Result<Vec<f32>> {
        use anyhow::Context;
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let mut pos = 12usize;
        let mut data_start = None;
        let mut data_len = 0usize;
        let mut audio_format = 1u16; // PCM default
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            if id == b"fmt " {
                audio_format = u16::from_le_bytes(bytes[pos + 8..pos + 10].try_into().unwrap());
            } else if id == b"data" {
                data_start = Some(pos + 8);
                data_len = size;
                break;
            }
            pos += 8 + size + (size & 1);
        }
        let start = data_start.context("no 'data' chunk in WAV")?;
        let end = (start + data_len).min(bytes.len());
        if audio_format == 3 {
            // IEEE float32
            let n = (end - start) / 4;
            let mut samples = Vec::with_capacity(n);
            for i in 0..n {
                let b: [u8; 4] = bytes[start + i * 4..start + i * 4 + 4].try_into().unwrap();
                samples.push(f32::from_le_bytes(b));
            }
            Ok(samples)
        } else {
            // 16-bit PCM
            let n = (end - start) / 2;
            let mut samples = Vec::with_capacity(n);
            for i in 0..n {
                let b: [u8; 2] = bytes[start + i * 2..start + i * 2 + 2].try_into().unwrap();
                samples.push(i16::from_le_bytes(b) as f32 / 32768.0);
            }
            Ok(samples)
        }
    }

    #[test]
    #[ignore = "requires silero_vad.onnx in ./models AND test_data/LJ001-0001_16k.wav at repo root"]
    fn test_chunker_speech_windows_cluster() -> Result<()> {
        let vad = open_vad()?;
        let cfg = WindowConfig {
            max_window_secs: 28.0,
            stride_secs: 1.0,
            vad_threshold: 0.5,
            min_speech_secs: 0.25,
        };
        let mut chunker = Chunker::new(cfg, vad);

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        let speech_wav = repo_root.join("test_data").join("LJ001-0001_16k.wav");
        assert!(
            speech_wav.exists(),
            "speech WAV not found: {}",
            speech_wav.display()
        );
        let speech_samples = read_wav_samples(&speech_wav)?;
        // Use up to 2 s of speech.
        let speech_len = (2 * SAMPLE_RATE as usize).min(speech_samples.len());
        let speech_clip = &speech_samples[..speech_len];

        // 4 s silence + 2 s speech + 4 s silence ≈ 10 s total.
        let silence_4s = vec![0.0f32; 4 * SAMPLE_RATE as usize];
        let silence_4s_end = vec![0.0f32; 4 * SAMPLE_RATE as usize];

        let chunk_size = 512usize;
        let mut windows: Vec<StreamWindow> = Vec::new();

        let all_samples: Vec<f32> = silence_4s
            .iter()
            .chain(speech_clip.iter())
            .chain(silence_4s_end.iter())
            .copied()
            .collect();

        for chunk in all_samples.chunks(chunk_size) {
            if let Some(w) = chunker.push(chunk) {
                windows.push(w);
            }
        }

        // At least one window with speech should have fired (the growing buffer keeps the
        // speech samples from the middle 2 s of audio for all subsequent windows).
        assert!(!windows.is_empty(), "no windows emitted");
        let speech_windows: Vec<&StreamWindow> = windows.iter().filter(|w| w.had_speech).collect();
        assert!(!speech_windows.is_empty(), "no speech windows detected");

        // window_start_secs anchors to where the *current utterance buffer* started — which
        // here is 0 (we never reset). Just sanity-check it's a reasonable wall-clock value.
        for w in &speech_windows {
            assert!(
                w.window_start_secs >= 0.0 && w.window_start_secs <= 11.0,
                "speech window start {:.2} s outside expected 0–11 s range",
                w.window_start_secs
            );
        }

        Ok(())
    }

    fn make_token(id: u32) -> TokenEmit {
        TokenEmit {
            id,
            text: format!("tok{id}"),
            logprob: 0.0,
            window_ts_secs: None,
            is_special: false,
        }
    }

    fn make_ts_token(id: u32, ts: f32) -> TokenEmit {
        TokenEmit {
            id,
            text: String::new(),
            logprob: 0.0,
            window_ts_secs: Some(ts),
            is_special: true,
        }
    }

    fn committed_ids(delta: &CommitDelta) -> Vec<u32> {
        match delta {
            CommitDelta::Committed { new_tokens, .. } => new_tokens.iter().map(|t| t.id).collect(),
            CommitDelta::Tentative { .. } => vec![],
        }
    }

    fn tentative_ids(delta: &CommitDelta) -> Vec<u32> {
        match delta {
            CommitDelta::Tentative { tokens, .. } => tokens.iter().map(|t| t.id).collect(),
            CommitDelta::Committed { .. } => vec![],
        }
    }

    /// LocalAgreement-2: three-round scenario from the C7 acceptance criteria.
    #[test]
    fn test_committer_local_agreement_2() -> Result<()> {
        let mut c = Committer::new();

        // Round 1: [tok1, tok2, tok3] — nothing can be committed yet (no prior candidate).
        let (comm, tent) = c.ingest(vec![make_token(1), make_token(2), make_token(3)]);
        assert!(
            committed_ids(&comm).is_empty(),
            "round 1: nothing committed"
        );
        assert_eq!(
            tentative_ids(&tent),
            vec![1, 2, 3],
            "round 1: all tokens tentative"
        );

        // Round 2: [tok1, tok2, tok4, tok5] — LCP with round-1 is [tok1, tok2].
        let (comm, tent) = c.ingest(vec![
            make_token(1),
            make_token(2),
            make_token(4),
            make_token(5),
        ]);
        assert_eq!(
            committed_ids(&comm),
            vec![1, 2],
            "round 2: [tok1, tok2] committed"
        );
        assert_eq!(
            tentative_ids(&tent),
            vec![4, 5],
            "round 2: tentative = [tok4, tok5]"
        );
        assert_eq!(c.committed_text(), "tok1tok2");

        // Round 3: [tok1, tok2, tok4, tok6] — LCP with round-2 is [tok1, tok2, tok4].
        let (comm, tent) = c.ingest(vec![
            make_token(1),
            make_token(2),
            make_token(4),
            make_token(6),
        ]);
        assert_eq!(
            committed_ids(&comm),
            vec![4],
            "round 3: [tok4] newly committed"
        );
        assert_eq!(tentative_ids(&tent), vec![6], "round 3: tentative = [tok6]");
        assert_eq!(c.committed_text(), "tok1tok2tok4");

        // finalize_utterance: force-commit the remaining [tok6].
        let final_delta = c.finalize_utterance();
        assert_eq!(
            committed_ids(&final_delta),
            vec![6],
            "finalize: [tok6] committed"
        );
        assert_eq!(c.committed_text(), "tok1tok2tok4tok6");
        assert_eq!(
            c.committed_tokens()
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 6],
            "committed_tokens must remain readable after finalize for PromptContext"
        );

        // Repeated finalize without an intervening ingest must not panic and must yield empty.
        let noop = c.finalize_utterance();
        assert!(
            committed_ids(&noop).is_empty(),
            "second finalize must be a no-op"
        );

        // Next ingest auto-resets the committer; the new utterance must start fresh and
        // commit normally rather than refusing because cumulative `committed.len()` is high.
        c.ingest(vec![make_token(10), make_token(20), make_token(30)]);
        let (comm, _) = c.ingest(vec![make_token(10), make_token(20), make_token(40)]);
        assert_eq!(
            committed_ids(&comm),
            vec![10, 20],
            "new utterance must commit [tok10, tok20] after auto-reset"
        );

        Ok(())
    }

    /// Timestamp tokens between words must not break LCP matching.
    #[test]
    fn test_committer_lcp_ignores_timestamps() -> Result<()> {
        let mut c = Committer::new();
        // Round 1: [ts:0.0, tok1, ts:0.2, tok2]
        c.ingest(vec![
            make_ts_token(50364, 0.0),
            make_token(1),
            make_ts_token(50366, 0.2),
            make_token(2),
        ]);
        // Round 2: same content tokens but different timestamp values
        let (comm, _tent) = c.ingest(vec![
            make_ts_token(50365, 0.02),
            make_token(1),
            make_ts_token(50367, 0.22),
            make_token(2),
        ]);
        assert_eq!(
            committed_ids(&comm),
            vec![50365, 1, 50367, 2],
            "content tokens tok1/tok2 match despite different timestamp IDs; both windows committed"
        );
        Ok(())
    }

    fn make_window(trailing_silence_secs: f32, window_start_secs: f32) -> StreamWindow {
        let real_samples = 5 * SAMPLE_RATE as usize;
        StreamWindow {
            samples: vec![0.0f32; real_samples],
            real_samples,
            window_start_secs,
            had_speech: trailing_silence_secs < 5.0,
            trailing_silence_secs,
            forced_eou: false,
        }
    }

    #[test]
    fn test_endpointer_no_eou_silence_no_text() -> Result<()> {
        let cfg = EndpointConfig {
            silence_secs: 2.0,
            punct_silence_secs: 0.8,
            min_utterance_secs: 0.0,
        };
        let mut ep = Endpointer::new(cfg);
        let window = make_window(2.5, 0.0);
        assert!(!ep.step(&window, ""), "no EOU when committed text is empty");
        Ok(())
    }

    #[test]
    fn test_endpointer_hard_eou_fires_once() -> Result<()> {
        let cfg = EndpointConfig {
            silence_secs: 2.0,
            punct_silence_secs: 0.8,
            min_utterance_secs: 0.0,
        };
        let mut ep = Endpointer::new(cfg);

        // Speech in progress — not enough silence yet.
        assert!(!ep.step(&make_window(0.3, 0.0), "hello world"));
        // Long trailing silence — hard EOU fires.
        assert!(
            ep.step(&make_window(2.5, 1.0), "hello world"),
            "hard EOU should fire"
        );
        // Without reset, must not re-fire.
        assert!(
            !ep.step(&make_window(3.0, 2.0), "hello world"),
            "must not re-fire before reset"
        );

        Ok(())
    }

    #[test]
    fn test_endpointer_soft_eou_punctuation() -> Result<()> {
        let cfg = EndpointConfig {
            silence_secs: 2.0,
            punct_silence_secs: 0.8,
            min_utterance_secs: 0.0,
        };
        let mut ep = Endpointer::new(cfg);
        // Terminal period + 0.9 s silence → soft EOU.
        assert!(
            ep.step(&make_window(0.9, 0.0), "Hello."),
            "soft EOU should fire after '.' + 0.9 s silence"
        );
        Ok(())
    }

    // --- PromptContext tests (no model files required) ---

    fn make_committed_tokens(ids: &[u32]) -> Vec<TokenEmit> {
        ids.iter().map(|&id| make_token(id)).collect()
    }

    #[test]
    fn test_prompt_context_basic() -> Result<()> {
        let tokens = make_committed_tokens(&[1, 2, 3]);
        let mut ctx = PromptContext::new(60, Some(99_999));
        ctx.update_after_eou(&tokens, 128);
        let p = ctx.prompt_tokens();
        assert_eq!(p[0], 99_999, "must start with <|prevtext|>");
        assert_eq!(&p[1..], &[1, 2, 3]);
        Ok(())
    }

    #[test]
    fn test_prompt_context_no_prevtext() -> Result<()> {
        let tokens = make_committed_tokens(&[1, 2, 3]);
        let mut ctx = PromptContext::new(60, None);
        ctx.update_after_eou(&tokens, 128);
        assert_eq!(ctx.prompt_tokens(), &[1, 2, 3]);
        Ok(())
    }

    #[test]
    fn test_prompt_context_filters_specials() -> Result<()> {
        let tokens = vec![
            make_ts_token(50364, 0.0),
            make_token(1),
            make_ts_token(50366, 0.2),
            make_token(2),
        ];
        let mut ctx = PromptContext::new(60, None);
        ctx.update_after_eou(&tokens, 128);
        assert_eq!(
            ctx.prompt_tokens(),
            &[1, 2],
            "special/timestamp tokens must be excluded from prompt"
        );
        Ok(())
    }

    #[test]
    fn test_prompt_context_caps_max_prompt_tokens() -> Result<()> {
        // 101 regular tokens; cap = 60 → last 60 IDs (41 through 100).
        let ids: Vec<u32> = (0u32..101).collect();
        let tokens = make_committed_tokens(&ids);
        let mut ctx = PromptContext::new(60, None);
        ctx.update_after_eou(&tokens, 128);
        assert_eq!(ctx.prompt_tokens().len(), 60);
        assert_eq!(
            ctx.prompt_tokens()[0],
            41,
            "should be the 42nd token (ID 41)"
        );
        assert_eq!(ctx.prompt_tokens()[59], 100);
        Ok(())
    }

    #[test]
    fn test_prompt_context_caps_context_limit() -> Result<()> {
        // max_new_tokens=400 → max_allowed = 448 - 3 - 400 = 45.
        // prevtext takes 1 slot → text_slots = 44; cap = min(60, 44) = 44.
        // Total = 1 (prevtext) + 44 = 45.
        let ids: Vec<u32> = (0u32..100).collect();
        let tokens = make_committed_tokens(&ids);
        let mut ctx = PromptContext::new(60, Some(99_999));
        ctx.update_after_eou(&tokens, 400);
        assert_eq!(
            ctx.prompt_tokens().len(),
            45,
            "prompt must fit: prevtext(1) + 44 text = 45 ≤ 448-3-400"
        );
        assert_eq!(
            ctx.prompt_tokens()[0],
            99_999,
            "first token must be <|prevtext|>"
        );
        Ok(())
    }

    #[test]
    fn test_prompt_context_no_prevtext_on_empty_commit() -> Result<()> {
        // Only a timestamp token committed → regular tail is empty → no prevtext emitted.
        let tokens = vec![make_ts_token(50364, 0.0)];
        let mut ctx = PromptContext::new(60, Some(99_999));
        ctx.update_after_eou(&tokens, 128);
        assert!(
            ctx.prompt_tokens().is_empty(),
            "<|prevtext|> must not be emitted when no regular tokens were committed"
        );
        Ok(())
    }

    #[test]
    fn test_prompt_context_reset() -> Result<()> {
        let tokens = make_committed_tokens(&[1, 2]);
        let mut ctx = PromptContext::new(60, None);
        ctx.update_after_eou(&tokens, 128);
        assert!(!ctx.prompt_tokens().is_empty());
        ctx.reset();
        assert!(ctx.prompt_tokens().is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires silero_vad.onnx in ./models"]
    fn test_chunker_first_window_at_stride_boundary() -> Result<()> {
        // min_speech_secs = 0 means the speech gate is always satisfied; the first
        // window must fire after exactly stride_secs regardless of VAD output.
        let vad = open_vad()?;
        let cfg = WindowConfig {
            max_window_secs: 3.0,
            stride_secs: 1.0,
            vad_threshold: 0.5,
            min_speech_secs: 0.0,
        };
        let mut chunker = Chunker::new(cfg, vad);

        let one_second = vec![0.0f32; SAMPLE_RATE as usize];

        // Push all at once.
        let window = chunker.push(&one_second);

        let w = window.expect("first window should have fired after 1 s (stride_secs = 1.0)");

        assert_eq!(
            w.real_samples, SAMPLE_RATE as usize,
            "real_samples should equal exactly 1 s of input"
        );
        assert_eq!(
            w.samples.len(),
            3 * SAMPLE_RATE as usize,
            "samples vec must be max_window_secs × 16000 long"
        );
        assert!(
            (w.window_start_secs - 0.0).abs() < 1e-4,
            "window_start_secs should be 0.0"
        );
        assert!(!w.forced_eou, "first 1-s window should not be a forced EOU");

        Ok(())
    }

    /// Growing-buffer EOU: pushing > max_window_secs of audio must produce a forced_eou window.
    #[test]
    #[ignore = "requires silero_vad.onnx in ./models"]
    fn test_chunker_forced_eou_at_cap() -> Result<()> {
        let vad = open_vad()?;
        let cfg = WindowConfig {
            max_window_secs: 2.0,
            stride_secs: 1.0,
            vad_threshold: 0.5,
            min_speech_secs: 0.0,
        };
        let mut chunker = Chunker::new(cfg, vad);

        // Push 3 s of silence in one call; the chunker should cap the buffer at 2 s of
        // real audio and emit a forced_eou window. (The last window returned wins per
        // Chunker::push semantics.)
        let three_seconds = vec![0.0f32; 3 * SAMPLE_RATE as usize];
        let window = chunker
            .push(&three_seconds)
            .expect("at least one window should have fired");
        assert!(window.forced_eou, "forced_eou must be set at the cap");
        assert_eq!(window.real_samples, 2 * SAMPLE_RATE as usize);

        // After reset_utterance, the buffer drops to 0 and the next stride starts fresh.
        chunker.reset_utterance();
        let one_second = vec![0.0f32; SAMPLE_RATE as usize];
        let w2 = chunker
            .push(&one_second)
            .expect("first stride of new utterance should fire");
        assert!(!w2.forced_eou);
        assert_eq!(w2.real_samples, SAMPLE_RATE as usize);
        // window_start_secs anchors to the new utterance's audio[0] = 3 s of wall-clock.
        assert!(
            (w2.window_start_secs - 3.0).abs() < 1e-3,
            "new utterance start should be ~3 s, got {}",
            w2.window_start_secs
        );

        Ok(())
    }
}
