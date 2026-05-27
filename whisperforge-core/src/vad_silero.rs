use anyhow::{Context, Result};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use ort::{session::Session, value::TensorRef};
use std::path::{Path, PathBuf};

const STATE_ELEMS: usize = 2 * 1 * 128;
const SILERO_REPO: &str = "onnx-community/silero-vad";
const SILERO_HF_FILE: &str = "onnx/model.onnx";
const SILERO_LOCAL_FILE: &str = "silero_vad.onnx";

/// Voice activity detector backed by the Silero ONNX model (v5).
///
/// Maintains per-stream LSTM state across frames; call [`reset_state`] at
/// stream open and after every end-of-utterance event.
pub struct SileroVad {
    session: Session,
    state: Vec<f32>, // [2, 1, 128] LSTM state, stored flat
    sample_rate: i64,
}

impl SileroVad {
    /// Load a Silero ONNX model from `model_path`.
    /// Use [`ensure_silero_model`] to obtain the path.
    pub fn open(model_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .with_intra_threads(1)
            .context("Failed to set intra-op threads")?
            .commit_from_file(model_path)
            .with_context(|| {
                format!(
                    "Failed to load Silero VAD model from {}",
                    model_path.display()
                )
            })?;
        Ok(Self {
            session,
            state: vec![0.0f32; STATE_ELEMS],
            sample_rate: 16000,
        })
    }

    /// Reset LSTM state. Call at stream open and after every EOU event.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Score exactly 512 samples (32 ms @ 16 kHz). Returns P(speech) ∈ [0, 1].
    pub fn probability(&mut self, frame: &[f32; 512]) -> Result<f32> {
        let sr_buf = [self.sample_rate];

        let input_t = TensorRef::from_array_view(([1_usize, 512], frame.as_slice()))
            .context("input tensor")?;
        let state_t = TensorRef::from_array_view(([2_usize, 1, 128], self.state.as_slice()))
            .context("state tensor")?;
        let sr_t =
            TensorRef::from_array_view(([1_usize], sr_buf.as_slice())).context("sr tensor")?;

        // Scoped block so `outputs` (which holds &mut self.session) drops before
        // we update self.state.
        let (prob, new_state) = {
            let outputs = self
                .session
                .run(ort::inputs![
                    "input" => input_t,
                    "state" => state_t,
                    "sr"    => sr_t
                ])
                .context("VAD inference")?;

            let (_, prob_data) = outputs["output"]
                .try_extract_tensor::<f32>()
                .context("extract output")?;
            let prob = prob_data[0];

            let (_, state_data) = outputs["stateN"]
                .try_extract_tensor::<f32>()
                .context("extract stateN")?;
            let new_state = state_data.to_vec();

            (prob, new_state)
        };

        self.state = new_state;
        Ok(prob)
    }
}

/// Download `silero_vad.onnx` into `models_dir` if not already present.
///
/// Fetches from `onnx-community/silero-vad` on HuggingFace Hub (ONNX v5 model,
/// `onnx/model.onnx`). Idempotent: returns immediately if the file already exists.
pub fn ensure_silero_model(models_dir: &Path) -> Result<PathBuf> {
    let target = models_dir.join(SILERO_LOCAL_FILE);
    if target.exists() {
        return Ok(target);
    }

    std::fs::create_dir_all(models_dir)
        .with_context(|| format!("create models dir {}", models_dir.display()))?;

    let api = ApiBuilder::from_env()
        .build()
        .context("HuggingFace API init")?;
    let cached = api
        .repo(Repo::new(SILERO_REPO.to_string(), RepoType::Model))
        .get(SILERO_HF_FILE)
        .with_context(|| format!("download {SILERO_HF_FILE} from {SILERO_REPO}"))?;

    std::fs::copy(&cached, &target).with_context(|| format!("copy to {}", target.display()))?;

    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn models_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("models")
    }

    fn open_vad() -> Result<SileroVad> {
        let model_path = ensure_silero_model(&models_dir())?;
        SileroVad::open(&model_path)
    }

    fn mean_prob(vad: &mut SileroVad, samples: &[f32]) -> Result<f32> {
        let frames = samples.len() / 512;
        let mut sum = 0.0f32;
        for i in 0..frames {
            let frame: &[f32; 512] = samples[i * 512..(i + 1) * 512]
                .try_into()
                .expect("slice length");
            sum += vad.probability(frame)?;
        }
        Ok(if frames > 0 { sum / frames as f32 } else { 0.0 })
    }

    #[test]
    #[ignore = "requires silero_vad.onnx in ./models — download via ensure_silero_model or wforge convert"]
    fn test_vad_silence() -> Result<()> {
        let mut vad = open_vad()?;
        let silence = vec![0.0f32; 16000];
        let p = mean_prob(&mut vad, &silence)?;
        assert!(p < 0.1, "silence should score < 0.1, got {p:.4}");
        Ok(())
    }

    #[test]
    #[ignore = "requires silero_vad.onnx in ./models — download via ensure_silero_model or wforge convert"]
    fn test_vad_sine_not_speech() -> Result<()> {
        let mut vad = open_vad()?;
        let sine: Vec<f32> = (0..16000)
            .map(|i| (TAU * 1000.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let p = mean_prob(&mut vad, &sine)?;
        assert!(p < 0.3, "1 kHz sine should score < 0.3, got {p:.4}");
        Ok(())
    }

    /// Read all samples from a RIFF WAV file that uses IEEE float32 PCM (format tag 3).
    /// Returns the raw f32 samples without any resampling.
    fn read_f32_wav(path: &std::path::Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        // Find the 'data' chunk (skip RIFF/fmt/fact headers).
        let mut pos = 12usize;
        let mut data_start = None;
        let mut data_len = 0usize;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            if id == b"data" {
                data_start = Some(pos + 8);
                data_len = size;
                break;
            }
            pos += 8 + size + (size & 1); // chunks are word-aligned
        }
        let start = data_start.context("no 'data' chunk in WAV")?;
        let end = (start + data_len).min(bytes.len());
        let n = (end - start) / 4;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let b: [u8; 4] = bytes[start + i * 4..start + i * 4 + 4].try_into().unwrap();
            samples.push(f32::from_le_bytes(b));
        }
        Ok(samples)
    }

    #[test]
    #[ignore = "requires silero_vad.onnx in ./models AND test_data/LJ001-0001_16k.wav at repo root"]
    fn test_vad_speech() -> Result<()> {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        // LJ001-0001_16k.wav: IEEE float32, mono, 16 kHz — read raw bytes to avoid
        // the copy_to_vec_interleaved multi-packet overwrite bug in audio.rs::load_audio_file.
        let wav = repo_root.join("test_data").join("LJ001-0001_16k.wav");
        assert!(wav.exists(), "speech WAV not found: {}", wav.display());

        let samples = read_f32_wav(&wav)?;
        assert!(!samples.is_empty(), "no samples in {}", wav.display());

        let mut vad = open_vad()?;
        let p = mean_prob(&mut vad, &samples)?;
        assert!(p > 0.7, "speech should score > 0.7, got {p:.4}");
        Ok(())
    }
}
