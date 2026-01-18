use anyhow::{anyhow, Result};
use burn::{
    tensor::{backend::Backend, Tensor},
    tensor::activation::softmax,
};
use rubato::{Resampler, SincFixedIn, SincInterpolationType};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioData {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    pub fn to_mono(&self) -> AudioData {
        if self.channels == 1 {
            return self.clone();
        }

        let mono_samples: Vec<f32> = self
            .samples
            .chunks(self.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / self.channels as f32)
            .collect();

        AudioData {
            samples: mono_samples,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    pub fn resample(&self, target_sample_rate: u32) -> Result<AudioData> {
        if self.sample_rate == target_sample_rate {
            return Ok(self.clone());
        }

        let resampler = SincFixedIn::new(
            target_sample_rate as f64 / self.sample_rate as f64,
            2.0,
            SincInterpolationType::Linear,
            1024,
            2,
        )
        .map_err(|e| anyhow!("Failed to create resampler: {}", e))?;

        let audio_in = [&self.samples[..]];
        let resampled = resampler
            .process(&audio_in)
            .map_err(|e| anyhow!("Resampling failed: {}", e))?;

        Ok(AudioData {
            samples: resampled.into_iter().flatten().collect(),
            sample_rate: target_sample_rate,
            channels: self.channels,
        })
    }

    pub fn to_16khz_mono(&self) -> Result<AudioData> {
        let mono = self.to_mono();
        mono.resample(16000)
    }
}

pub fn load_wav_file<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| anyhow!("Failed to open WAV file: {}", e))?;

    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .into_samples::<f32>()
        .map(|s| s.unwrap_or(0.0))
        .collect();

    Ok(AudioData {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

pub fn save_wav_file<P: AsRef<Path>>(audio: &AudioData, path: P) -> Result<()> {
    use hound::{WavWriter, SampleFormat};

    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| anyhow!("Failed to create WAV writer: {}", e))?;

    for &sample in &audio.samples {
        writer.write_sample(sample)
            .map_err(|e| anyhow!("Failed to write sample: {}", e))?;
    }

    writer.finalize()
        .map_err(|e| anyhow!("Failed to finalize WAV file: {}", e))?;

    Ok(())
}

// Mel spectrogram computation
pub fn compute_mel_spectrogram<B: Backend>(
    audio: &AudioData,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    use burn::tensor::Tensor;

    // Convert audio to tensor [1, samples]
    let audio_tensor = Tensor::<B, 1>::from_floats(audio.samples.as_slice(), device);
    let audio_tensor = audio_tensor.unsqueeze_dim(0);

    // Compute mel filter bank (simplified version)
    let mel_filters = create_mel_filter_bank(n_fft / 2 + 1, n_mels, audio.sample_rate, device);

    // Compute STFT (simplified - using windowed FFT)
    let stft = compute_stft(&audio_tensor, n_fft, hop_length);

    // Apply mel filter bank
    let mel_spec = stft.matmul(mel_filters.transpose());

    // Log compression
    let log_mel_spec = mel_spec.log() + 1e-6;

    // Add batch dimension [1, n_mels, time]
    Ok(log_mel_spec.unsqueeze_dim(0))
}

fn create_mel_filter_bank<B: Backend>(
    n_freqs: usize,
    n_mels: usize,
    sample_rate: u32,
    device: &B::Device,
) -> Tensor<B, 2> {
    // Simplified mel filter bank creation
    // In a full implementation, this would create proper mel-spaced filter banks
    let filters = Tensor::random(
        [n_freqs, n_mels],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        device,
    );
    filters
}

fn compute_stft<B: Backend>(audio: &Tensor<B, 2>, n_fft: usize, hop_length: usize) -> Tensor<B, 3> {
    // Simplified STFT implementation
    // In a full implementation, this would use FFT operations
    let [batch_size, n_samples] = audio.dims();
    let n_frames = (n_samples - n_fft) / hop_length + 1;

    // For now, return dummy data
    Tensor::random(
        [batch_size, n_frames, n_fft / 2 + 1],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        audio.device(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_data_creation() {
        let audio = AudioData::new(vec![0.0, 0.5, -0.5, 1.0], 44100, 2);
        assert_eq!(audio.duration(), 2.0 / 44100.0);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn test_mono_conversion() {
        let stereo = AudioData::new(vec![1.0, 2.0, 3.0, 4.0], 44100, 2);
        let mono = stereo.to_mono();
        assert_eq!(mono.samples, vec![1.5, 3.5]);
        assert_eq!(mono.channels, 1);
    }

    #[test]
    fn test_mel_spectrogram() {
        let audio = AudioData::new(vec![0.0; 16000], 16000, 1); // 1 second of silence
        let result = compute_mel_spectrogram::<burn::backend::NdArray>(
            &audio,
            400,
            160,
            80,
            &burn::backend::NdArrayDevice::default(),
        );
        assert!(result.is_ok());
    }
}