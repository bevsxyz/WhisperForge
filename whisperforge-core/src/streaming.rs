use std::collections::VecDeque;

use crate::stream_decode::TokenEmit;
use crate::vad_silero::SileroVad;

const SAMPLE_RATE: u32 = 16_000;
const VAD_FRAME_SIZE: usize = 512;

pub struct WindowConfig {
    pub window_secs: f32,
    pub stride_secs: f32,
    pub vad_threshold: f32,
    /// Minimum accumulated speech before the first window fires (debounces VAD flicker).
    pub min_speech_secs: f32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_secs: 5.0,
            stride_secs: 1.0,
            vad_threshold: 0.5,
            min_speech_secs: 0.25,
        }
    }
}

pub struct StreamWindow {
    /// Exactly `window_secs × 16 000` samples; zero-padded at the front when the
    /// buffer is not yet full, or at the back when draining post-speech.
    pub samples: Vec<f32>,
    /// Number of real (non-padding) samples in `samples`.
    pub real_samples: usize,
    /// Wall-clock offset since stream start: `(total_seen − real_samples) / 16000`.
    pub window_start_secs: f32,
    /// True if any VAD-positive frame falls within the current window buffer.
    pub had_speech: bool,
    /// Seconds since the last VAD-positive frame; for use by the endpointer.
    pub trailing_silence_secs: f32,
}

pub struct Chunker {
    cfg: WindowConfig,
    vad: SileroVad,
    buf: VecDeque<f32>,
    pub samples_since_last_stride: usize,
    pub total_samples_seen: u64,
    pub last_speech_at_sample: Option<u64>,

    // --- private implementation details ---

    // Total speech samples seen in the session; gates the first window emission.
    speech_samples_accumulated: u64,
    // Partial accumulation buffer for in-progress VAD frames.
    vad_frame_buf: Vec<f32>,
    // Once we have emitted at least one window, subsequent strides fire even during silence.
    window_ever_emitted: bool,
    // Rolling log of is_speech per completed VAD frame, bounded to cover the window.
    vad_decisions: VecDeque<bool>,
}

impl Chunker {
    pub fn new(cfg: WindowConfig, vad: SileroVad) -> Self {
        let window_samples = (cfg.window_secs * SAMPLE_RATE as f32) as usize;
        let max_vad_frames = window_samples.div_ceil(VAD_FRAME_SIZE);
        Self {
            cfg,
            vad,
            buf: VecDeque::with_capacity(window_samples),
            samples_since_last_stride: 0,
            total_samples_seen: 0,
            last_speech_at_sample: None,
            speech_samples_accumulated: 0,
            vad_frame_buf: Vec::with_capacity(VAD_FRAME_SIZE),
            window_ever_emitted: false,
            vad_decisions: VecDeque::with_capacity(max_vad_frames),
        }
    }

    /// Push new microphone samples; returns `Some(window)` when a stride boundary fires.
    ///
    /// In the common case (push size ≪ stride interval) at most one window fires per call.
    /// If multiple stride boundaries happen to fall within one call, only the last one
    /// is returned; the intermediate window state is still correctly advanced.
    pub fn push(&mut self, samples: &[f32]) -> Option<StreamWindow> {
        let stride_samples = (self.cfg.stride_secs * SAMPLE_RATE as f32) as usize;
        let window_samples = (self.cfg.window_secs * SAMPLE_RATE as f32) as usize;
        let min_speech_samples = (self.cfg.min_speech_secs * SAMPLE_RATE as f32) as u64;
        let max_vad_frames = window_samples.div_ceil(VAD_FRAME_SIZE);

        let mut result: Option<StreamWindow> = None;

        for &s in samples {
            // Maintain sliding window buffer (bounded to window_secs × 16 kHz samples).
            if self.buf.len() >= window_samples {
                self.buf.pop_front();
            }
            self.buf.push_back(s);
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

                // Keep a rolling window of decisions aligned with the audio buffer.
                if self.vad_decisions.len() >= max_vad_frames {
                    self.vad_decisions.pop_front();
                }
                self.vad_decisions.push_back(is_speech);
            }

            // Check for stride tick.
            if self.samples_since_last_stride >= stride_samples {
                let enough_speech = self.speech_samples_accumulated >= min_speech_samples;
                if enough_speech || self.window_ever_emitted {
                    result = Some(self.build_window(window_samples));
                    self.window_ever_emitted = true;
                }
                self.samples_since_last_stride = 0;
            }
        }

        result
    }

    fn build_window(&self, window_samples: usize) -> StreamWindow {
        let real_samples = self.buf.len();
        let buf_slice: Vec<f32> = self.buf.iter().copied().collect();

        // Zero-pad the front when the buffer isn't full yet.
        let samples = if buf_slice.len() < window_samples {
            let pad_len = window_samples - buf_slice.len();
            let mut v = vec![0.0f32; pad_len];
            v.extend_from_slice(&buf_slice);
            v
        } else {
            buf_slice
        };

        let trailing_silence_secs = match self.last_speech_at_sample {
            Some(last) => self.total_samples_seen.saturating_sub(last) as f32 / SAMPLE_RATE as f32,
            None => self.total_samples_seen as f32 / SAMPLE_RATE as f32,
        };

        let window_start_secs =
            self.total_samples_seen.saturating_sub(real_samples as u64) as f32 / SAMPLE_RATE as f32;

        let had_speech = self.vad_decisions.iter().any(|&b| b);

        StreamWindow {
            samples,
            real_samples,
            window_start_secs,
            had_speech,
            trailing_silence_secs,
        }
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
}

impl Committer {
    pub fn new() -> Self {
        Self {
            last_candidate: Vec::new(),
            committed: Vec::new(),
            committed_text: String::new(),
        }
    }

    /// Ingest a new full-window decode result.
    ///
    /// Returns `(committed_delta, tentative_delta)`.
    pub fn ingest(&mut self, candidate: Vec<TokenEmit>) -> (CommitDelta, CommitDelta) {
        let lcp = lcp_len(&self.last_candidate, &candidate);
        let new_start = self.committed.len();
        let new_end = lcp.max(new_start);

        // Extend the committed buffer with any newly stable tokens.
        for t in candidate[new_start..new_end].iter() {
            self.committed.push(t.clone());
        }
        let new_text = build_text(&self.committed[new_start..]);
        self.committed_text.push_str(&new_text);

        // Collect the return slice before moving candidate.
        let new_tokens: Vec<TokenEmit> = self.committed[new_start..].to_vec();
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
    /// Clears `last_candidate` so the next utterance starts with a clean slate.
    pub fn finalize_utterance(&mut self) -> CommitDelta {
        let start = self.committed.len();
        let new_tokens: Vec<TokenEmit> = self.last_candidate[start..].to_vec();
        let new_text = build_text(&new_tokens);
        self.committed_text.push_str(&new_text);
        self.committed.extend(new_tokens.iter().cloned());
        self.last_candidate.clear();
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
            window_secs: 5.0,
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

        // At least one window with speech should have fired.
        assert!(!windows.is_empty(), "no windows emitted");
        let speech_windows: Vec<&StreamWindow> = windows.iter().filter(|w| w.had_speech).collect();
        assert!(!speech_windows.is_empty(), "no speech windows detected");

        // Speech windows should start in the 3–7 s range (generous margin for VAD latency).
        for w in &speech_windows {
            assert!(
                w.window_start_secs >= 2.0 && w.window_start_secs <= 8.0,
                "speech window start {:.2} s outside expected 2–8 s band",
                w.window_start_secs
            );
        }

        // Windows entirely within the first 3 s should be silence-only.
        let early_speech: Vec<&StreamWindow> = windows
            .iter()
            .filter(|w| w.window_start_secs < 3.0 && w.had_speech)
            .collect();
        assert!(
            early_speech.is_empty(),
            "{} speech window(s) detected before 3 s",
            early_speech.len()
        );

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
            vec![1, 2, 4, 6]
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

    #[test]
    #[ignore = "requires silero_vad.onnx in ./models"]
    fn test_chunker_first_window_at_stride_boundary() -> Result<()> {
        // min_speech_secs = 0 means the speech gate is always satisfied; the first
        // window must fire after exactly stride_secs regardless of VAD output.
        let vad = open_vad()?;
        let cfg = WindowConfig {
            window_secs: 3.0,
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
            "samples vec must be window_secs × 16000 long"
        );
        assert!(
            (w.window_start_secs - 0.0).abs() < 1e-4,
            "window_start_secs should be 0.0"
        );

        Ok(())
    }
}
