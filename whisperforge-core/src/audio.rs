use anyhow::{Result, anyhow};
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use burn::tensor::{Tensor, backend::Backend};
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, WindowFunction};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::path::Path;
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

// Whisper audio parameters
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_N_MELS: usize = 80;
pub const WHISPER_CHUNK_LENGTH: usize = 30; // seconds

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

        let mut resampler = Async::<f32>::new_sinc(
            f_ratio,
            2.0,
            &params,
            chunk_size,
            self.channels as usize,
            FixedAsync::Input,
        )
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
            vec![vec![0.0f32; estimated_output_frames]; self.channels as usize];
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
            for ch in &output_channels {
                output_samples.push(ch[frame]);
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

/// Load an audio file into raw f32 samples.
///
/// Supports WAV, MP3, FLAC, OGG/Vorbis, AAC/M4A, and MKV/WebM audio via
/// symphonia. The file format is detected from the extension first; symphonia
/// falls back to content-based probing when the extension is absent or unknown.
///
/// Returns interleaved f32 samples in `[-1.0, 1.0]` at the file's native
/// sample rate and channel count. Call [`AudioData::to_16khz_mono`] to
/// normalise before passing to the mel spectrogram pipeline.
pub fn load_audio_file<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow!("Failed to open audio file '{}': {}", path.display(), e))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow!("Unsupported audio format '{}': {}", path.display(), e))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No audio tracks found in '{}'", path.display()))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("Unknown sample rate in '{}'", path.display()))?;
    let channels = track
        .codec_params
        .channels
        .ok_or_else(|| anyhow!("Unknown channel count in '{}'", path.display()))?
        .count() as u16;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow!("Failed to create decoder for '{}': {}", path.display(), e))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                sample_buf = None;
                continue;
            }
            Err(e) => {
                return Err(anyhow!("Error reading '{}': {}", path.display(), e));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_)) => continue,
            Err(e) => return Err(anyhow!("Decode error in '{}': {}", path.display(), e)),
        };

        let buf = sample_buf.get_or_insert_with(|| {
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec())
        });
        buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buf.samples());
    }

    Ok(AudioData {
        samples,
        sample_rate,
        channels,
    })
}

/// Compute mel spectrogram for Whisper model input.
///
/// Matches OpenAI Whisper's Python preprocessing exactly:
/// 1. Resample/convert to 16 kHz mono.
/// 2. Pad (or trim) audio to exactly 30 seconds in **sample** space before the STFT.
/// 3. Apply `center=True` reflection padding (`n_fft/2` samples each side) so each
///    STFT frame is centred on its sample — matches `torch.stft` default.
/// 4. Use power spectrum and Slaney-normalised mel filters.
///
/// Always returns a tensor of shape `[1, n_mels, 3000]` (30 s at 100 fps).
pub fn compute_mel_spectrogram<B: Backend>(
    audio: &AudioData,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    // Ensure mono audio at correct sample rate
    let audio = if audio.channels != 1 || audio.sample_rate != WHISPER_SAMPLE_RATE {
        audio.to_16khz_mono()?
    } else {
        audio.clone()
    };

    // Pad or trim to exactly 30 s in sample space so silence carries the correct
    // log-mel value (~-1.0) rather than 0.0.
    let target_samples = 30 * WHISPER_SAMPLE_RATE as usize; // 480000
    let mut padded_samples = audio.samples.clone();
    if padded_samples.len() > target_samples {
        padded_samples.truncate(target_samples);
    } else {
        padded_samples.resize(target_samples, 0.0);
    }

    // center=True STFT (Python torch.stft default): reflect-pad n_fft/2 samples on
    // each side so each frame is centred on its sample rather than starting there.
    // For 480000 samples + 400 pad → 3001 STFT frames; we drop the last to match
    // Python's `magnitudes[..., :-1]` → exactly 3000 frames.
    let pad_len = n_fft / 2;
    let n = padded_samples.len();
    let mut centered_samples = Vec::with_capacity(n + 2 * pad_len);
    // Reflect-pad start: samples[pad_len], samples[pad_len-1], ..., samples[1]
    for i in (1..=pad_len).rev() {
        centered_samples.push(padded_samples[i]);
    }
    centered_samples.extend_from_slice(&padded_samples);
    // Reflect-pad end: samples[n-2], samples[n-3], ..., samples[n-1-pad_len]
    for i in 0..pad_len {
        centered_samples.push(padded_samples[n - 2 - i]);
    }

    // Compute STFT power spectrogram
    let magnitudes = compute_stft_magnitudes(&centered_samples, n_fft, hop_length);

    // Create mel filter bank
    let mel_filters = create_mel_filter_bank(n_fft, n_mels, audio.sample_rate as f32);

    // Apply mel filters: [n_mels, n_freqs] @ [n_freqs, n_frames] = [n_mels, n_frames]
    // Drop last STFT frame (Python does magnitudes[..., :-1] after center=True STFT).
    let n_frames = magnitudes[0].len().saturating_sub(1);
    let n_freqs = n_fft / 2 + 1;

    let mut mel_spec = vec![vec![0.0f32; n_frames]; n_mels];

    for mel in 0..n_mels {
        for frame in 0..n_frames {
            let mut sum = 0.0;
            for freq in 0..n_freqs {
                sum += mel_filters[mel][freq] * magnitudes[freq][frame];
            }
            mel_spec[mel][frame] = sum;
        }
    }

    // Log compression with clamping (Whisper uses log10 with specific scaling)
    let log_spec = log_mel_spectrogram(&mel_spec);

    let n_out = n_frames; // callers trim/pad to 3000 after receiving this
    let flat: Vec<f32> = log_spec
        .iter()
        .flat_map(|row| row[..n_out].iter().copied())
        .collect();
    let tensor = Tensor::<B, 1>::from_floats(flat.as_slice(), device);
    let tensor = tensor.reshape([1, n_mels, n_out]);

    Ok(tensor)
}

/// Stack mel spectrograms for multiple audio chunks into a single batch tensor.
///
/// Returns `[N, n_mels, 3000]` where N = `chunks.len()`. Each chunk is padded /
/// trimmed to exactly 3000 frames by `compute_mel_spectrogram`, so all slices are
/// the same shape and can be concatenated without further alignment.
///
/// Feeding the batch to the encoder in one call amortises GPU kernel launch
/// overhead and keeps the GPU compute unit saturated across chunks.
pub fn batch_mel_spectrograms<B: Backend>(
    chunks: &[AudioData],
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    anyhow::ensure!(!chunks.is_empty(), "batch_mel_spectrograms: no chunks");
    let mels: Vec<Tensor<B, 3>> = chunks
        .iter()
        .map(|c| compute_mel_spectrogram(c, n_fft, hop_length, n_mels, device))
        .collect::<Result<_>>()?;
    Ok(Tensor::cat(mels, 0))
}

/// Compute STFT and return magnitude spectrogram.
/// Returns [n_freqs][n_frames] where n_freqs = n_fft/2 + 1
#[allow(clippy::needless_range_loop)]
fn compute_stft_magnitudes(samples: &[f32], n_fft: usize, hop_length: usize) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    if n_frames == 0 {
        return vec![vec![0.0]; n_freqs];
    }

    // Create Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();

    // Setup FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Output: [n_freqs][n_frames]
    let mut magnitudes = vec![vec![0.0f32; n_frames]; n_freqs];

    // Process each frame
    let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Apply window and copy to buffer
        for i in 0..n_fft {
            let sample = if start + i < samples.len() {
                samples[start + i]
            } else {
                0.0
            };
            buffer[i] = Complex::new(sample * window[i], 0.0);
        }

        // Compute FFT in-place
        fft.process(&mut buffer);

        // Extract power spectrum (magnitude squared) for positive frequencies.
        // Whisper uses |STFT|^2, not |STFT|.
        for freq in 0..n_freqs {
            magnitudes[freq][frame_idx] = buffer[freq].norm_sqr();
        }
    }

    magnitudes
}

/// Create mel filter bank matrix matching OpenAI Whisper / librosa defaults.
///
/// Uses the Slaney mel scale (linear below 1 kHz, log above) with Slaney
/// normalization (`2 / (upper_hz - lower_hz)`) and FFT frequencies
/// `k * sample_rate / n_fft`. This replicates `librosa.filters.mel` exactly.
///
/// Returns `[n_mels][n_freqs]` where `n_freqs = n_fft / 2 + 1`.
fn create_mel_filter_bank(n_fft: usize, n_mels: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let fmax = sample_rate / 2.0;

    // Slaney mel scale: linear below 1 kHz, log above.
    let f_sp: f32 = 200.0 / 3.0;
    let min_log_hz: f32 = 1000.0;
    let min_log_mel: f32 = min_log_hz / f_sp;
    let logstep: f32 = 6.4f32.ln() / 27.0;

    let hz_to_mel = |f: f32| -> f32 {
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let mel_to_hz = |m: f32| -> f32 {
        if m >= min_log_mel {
            min_log_hz * ((m - min_log_mel) * logstep).exp()
        } else {
            f_sp * m
        }
    };

    // n_mels + 2 equally spaced points in mel space
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);
    let hz_pts: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    // FFT center frequencies: k * SR / N_FFT  (matches np.fft.rfftfreq)
    let fftfreqs: Vec<f32> = (0..n_freqs)
        .map(|k| k as f32 * sample_rate / n_fft as f32)
        .collect();

    let mut filters = vec![vec![0.0f32; n_freqs]; n_mels];
    for (i, filt) in filters.iter_mut().enumerate() {
        let lower = hz_pts[i];
        let center = hz_pts[i + 1];
        let upper = hz_pts[i + 2];
        // Slaney normalization: scale so that the filter sums to 2/(upper-lower)
        let enorm = 2.0 / (upper - lower).max(1e-8);
        for (k, &freq) in fftfreqs.iter().enumerate() {
            let rising = if center > lower {
                ((freq - lower) / (center - lower)).max(0.0)
            } else {
                0.0
            };
            let falling = if upper > center {
                ((upper - freq) / (upper - center)).max(0.0)
            } else {
                0.0
            };
            filt[k] = rising.min(falling) * enorm;
        }
    }

    filters
}

/// Apply log compression to mel spectrogram (Whisper-style).
/// Uses log10 with specific clamping and scaling.
fn log_mel_spectrogram(mel_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n_mels = mel_spec.len();
    if n_mels == 0 {
        return vec![];
    }
    let n_frames = mel_spec[0].len();

    // Find max value for normalization
    let mut max_val = 1e-10_f32;
    for mel in mel_spec {
        for &val in mel {
            if val > max_val {
                max_val = val;
            }
        }
    }

    let mut log_spec = vec![vec![0.0f32; n_frames]; n_mels];

    for (log_row, mel_row) in log_spec.iter_mut().zip(mel_spec.iter()) {
        for (log_val, &mel_val) in log_row.iter_mut().zip(mel_row.iter()) {
            *log_val = mel_val.max(1e-10).log10();
        }
    }

    // Apply Whisper-style normalization
    let log_max = log_spec
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(-f32::INFINITY, f32::max);

    // Clamp to max - 8.0 and scale
    for row in &mut log_spec {
        for val in row {
            *val = ((*val).max(log_max - 8.0) + 4.0) / 4.0;
        }
    }

    log_spec
}

/// Pad or trim audio to exactly 30 seconds (Whisper chunk length).
pub fn pad_or_trim_audio(audio: &AudioData, length_samples: usize) -> AudioData {
    let mut samples = audio.samples.clone();

    if samples.len() > length_samples {
        samples.truncate(length_samples);
    } else if samples.len() < length_samples {
        samples.resize(length_samples, 0.0);
    }

    AudioData {
        samples,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
    }
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
    fn test_hann_window() {
        // Verify Hann window properties
        let n = 400;
        let window: Vec<f32> = (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
            .collect();

        // Window should start and end near 0
        assert!(window[0] < 0.01);
        assert!(window[n - 1] < 0.01);

        // Window should peak at center
        let center = n / 2;
        assert!(window[center] > 0.99);
    }

    #[test]
    fn test_mel_filter_bank() {
        let filters = create_mel_filter_bank(400, 80, 16000.0);

        // Should have 80 mel bins
        assert_eq!(filters.len(), 80);

        // Each filter should have n_fft/2+1 = 201 frequency bins
        assert_eq!(filters[0].len(), 201);

        // Filters should be non-negative
        for filter in &filters {
            for &val in filter {
                assert!(val >= 0.0, "Filter values should be non-negative");
            }
        }

        // Slaney-normalized filters don't sum to 1; they use enorm = 2/(upper_hz-lower_hz).
        // Just verify each filter has at least one non-zero bin and a reasonable peak value.
        for filter in &filters {
            let max_val = filter.iter().cloned().fold(0.0f32, f32::max);
            // Peak should be positive (filter covers some FFT bins) and bounded
            assert!(
                max_val > 0.0,
                "Filter should have at least one non-zero bin"
            );
            assert!(
                max_val < 1.0,
                "Filter peak should be less than 1.0 (Slaney norm)"
            );
        }
    }

    #[test]
    fn test_stft_magnitudes() {
        // Test with a simple sine wave
        let sample_rate = 16000.0;
        let freq = 440.0; // A4 note
        let duration = 0.1; // 100ms
        let n_samples = (sample_rate * duration) as usize;

        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let magnitudes = compute_stft_magnitudes(&samples, 400, 160);

        // Should have n_fft/2+1 frequency bins
        assert_eq!(magnitudes.len(), 201);

        // Should have multiple frames
        assert!(!magnitudes[0].is_empty(), "Should have at least one frame");

        // Magnitudes should be non-negative
        for freq_bin in &magnitudes {
            for &mag in freq_bin {
                assert!(mag >= 0.0, "Magnitudes should be non-negative");
            }
        }
    }

    #[test]
    fn test_mel_spectrogram() {
        let audio = AudioData::new(vec![0.0; 16000], 16000, 1); // 1 second of silence
        let result = compute_mel_spectrogram::<burn::backend::NdArray>(
            &audio,
            WHISPER_N_FFT,
            WHISPER_HOP_LENGTH,
            WHISPER_N_MELS,
            &NdArrayDevice::default(),
        );

        assert!(result.is_ok());
        let tensor = result.unwrap();
        let dims = tensor.dims();

        // Should be [1, n_mels, n_frames]
        assert_eq!(dims[0], 1, "Batch size should be 1");
        assert_eq!(
            dims[1], WHISPER_N_MELS,
            "Should have {} mel bins",
            WHISPER_N_MELS
        );

        // compute_mel_spectrogram always pads to 30 s (480000 samples) with center=True
        // STFT reflection padding → always returns exactly 3000 mel frames.
        assert_eq!(dims[2], 3000, "Should always return 3000 mel frames");
    }

    #[test]
    fn test_mel_spectrogram_with_sine() {
        // Test with a 440 Hz sine wave
        let sample_rate = 16000;
        let freq = 440.0;
        let duration = 1.0;
        let n_samples = (sample_rate as f32 * duration) as usize;

        let samples: Vec<f32> = (0..n_samples)
            .map(|i| 0.5 * (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let audio = AudioData::new(samples, sample_rate, 1);
        let result = compute_mel_spectrogram::<burn::backend::NdArray>(
            &audio,
            WHISPER_N_FFT,
            WHISPER_HOP_LENGTH,
            WHISPER_N_MELS,
            &NdArrayDevice::default(),
        );

        assert!(result.is_ok());
        let tensor = result.unwrap();

        // Values should be in reasonable range after normalization
        let data = tensor.to_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        for &val in &values {
            assert!(val.is_finite(), "All values should be finite");
            // After Whisper normalization, values should be roughly in [-1, 1] range
            assert!(
                (-2.0..=2.0).contains(&val),
                "Values should be in reasonable range, got {}",
                val
            );
        }
    }

    #[test]
    fn test_pad_or_trim() {
        let audio = AudioData::new(vec![1.0, 2.0, 3.0], 16000, 1);

        // Test trimming
        let trimmed = pad_or_trim_audio(&audio, 2);
        assert_eq!(trimmed.samples, vec![1.0, 2.0]);

        // Test padding
        let padded = pad_or_trim_audio(&audio, 5);
        assert_eq!(padded.samples, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }
}
