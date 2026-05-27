use std::collections::VecDeque;

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
