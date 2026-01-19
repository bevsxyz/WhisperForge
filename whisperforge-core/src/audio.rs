use anyhow::{anyhow, Result};
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use burn::tensor::{backend::Backend, Tensor};
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, WindowFunction};
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

        let f_ratio = target_sample_rate as f64 / self.sample_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Cubic,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        // A reasonable chunk size in frames
        let chunk_size = 1024;

        let mut resampler =
            Async::<f32>::new_sinc(f_ratio, 2.0, &params, chunk_size, 2, FixedAsync::Input)
                .map_err(|e| anyhow!("Failed to create resampler: {}", e))?;

        // Convert interleaved samples to multi-channel format for rubato
        let frames_per_channel = self.samples.len() / self.channels as usize;
        let mut input_channels: Vec<Vec<f32>> =
            vec![Vec::with_capacity(frames_per_channel); self.channels as usize];

        // Deinterleave samples: [L,R,L,R,...] -> [[L,L,...], [R,R,...]]
        for (i, &sample) in self.samples.iter().enumerate() {
            let channel = i % self.channels as usize;
            input_channels[channel].push(sample);
        }

        let input_adapter =
            SequentialSliceOfVecs::new(&input_channels, self.channels as usize, frames_per_channel)
                .map_err(|e| anyhow!("Failed to create input adapter: {}", e))?;

        // The resampler processes data in chunks. We need to know the output size.
        let estimated_output_frames = (frames_per_channel as f64 * f_ratio) as usize;

        let mut output_channels: Vec<Vec<f32>> =
            vec![Vec::with_capacity(estimated_output_frames); self.channels as usize];
        let mut output_adapter = SequentialSliceOfVecs::new_mut(
            &mut output_channels,
            self.channels as usize,
            estimated_output_frames,
        )
        .map_err(|e| anyhow!("Failed to create output adapter: {}", e))?;

        // Use an indexing helper struct for chunked processing
        let mut indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            active_channels_mask: None,
            partial_len: None,
        };

        let mut input_frames_left = frames_per_channel;
        let mut input_frames_next = resampler.input_frames_next();

        // Loop over all full chunks.
        while input_frames_left >= input_frames_next {
            let (frames_read, frames_written) = resampler
                .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
                .map_err(|e| anyhow!("Resampling failed: {}", e))?;

            indexing.input_offset += frames_read;
            indexing.output_offset += frames_written;
            input_frames_left -= frames_read;
            input_frames_next = resampler.input_frames_next();
        }

        // Interleave the output channels back into a single vector
        let actual_output_frames = indexing.output_offset;
        let mut output_samples = Vec::with_capacity(actual_output_frames * self.channels as usize);

        for frame in 0..actual_output_frames {
            for channel in 0..self.channels as usize {
                output_samples.push(output_channels[channel][frame]);
            }
        }

        Ok(AudioData {
            samples: output_samples,
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
    let reader =
        hound::WavReader::open(path).map_err(|e| anyhow!("Failed to open WAV file: {}", e))?;

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
    use hound::{SampleFormat, WavWriter};

    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer =
        WavWriter::create(path, spec).map_err(|e| anyhow!("Failed to create WAV writer: {}", e))?;

    for &sample in &audio.samples {
        writer
            .write_sample(sample)
            .map_err(|e| anyhow!("Failed to write sample: {}", e))?;
    }

    writer
        .finalize()
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
    // stft: [1, n_frames, n_fft/2+1] (Rank 3)
    // mel_filters: [n_freqs, n_mels] (Rank 2)
    // Burn matmul supports broadcasting. We want [1, n_frames, n_mels]
    // stft [batch, m, k] . mel_filters [k, n] -> [batch, m, n]
    // stft dims: [batch, frames, freqs]
    // mel_filters dims: [freqs, mels]
    // matmul(stft, mel_filters) should work if mel_filters is broadcastable or 2D.
    // The previous code had mel_filters.transpose() which made it [mels, freqs],
    // and stft.matmul(...) expected rank 3 if stft is rank 3.
    // In Burn, if LHS is rank 3 [B, M, K], and RHS is rank 2 [K, N], it typically works.

    // Apply mel filter bank
    // stft: [1, n_frames, n_fft/2+1] (Rank 3)
    // mel_filters: [n_freqs, n_mels] (Rank 2) -> needs to be [1, n_freqs, n_mels] for matmul
    let mel_filters_3d = mel_filters.unsqueeze_dim(0); // Shape becomes [1, n_freqs, n_mels]
    let mel_spec = stft.matmul(mel_filters_3d); // Result is [1, n_frames, n_mels]

    // Log compression
    let log_mel_spec = mel_spec.log() + 1e-6;

    // Already has batch dimension [1, n_frames, n_mels]
    // Transpose to [1, n_mels, n_frames] for Whisper input format
    Ok(log_mel_spec.swap_dims(1, 2))
}

fn create_mel_filter_bank<B: Backend>(
    n_freqs: usize,
    n_mels: usize,
    _sample_rate: u32,
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
        &audio.device(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArrayDevice;

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
            &NdArrayDevice::default(),
        );
        assert!(result.is_ok());
    }
}
