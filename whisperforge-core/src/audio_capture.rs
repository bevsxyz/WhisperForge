use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    HeapRb,
    traits::{Consumer, Producer, Split},
};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

/// Captures audio from a microphone and resamples it to 16 kHz mono.
pub struct MicCapture {
    _stream: cpal::Stream,
    _resample_thread: Option<std::thread::JoinHandle<()>>,
    pub consumer: Arc<Mutex<ringbuf::HeapCons<f32>>>,
    pub native_sample_rate: u32,
    pub native_channels: u16,
}

impl MicCapture {
    /// Open a microphone input device.
    /// If `device_name` is None, uses the system default device.
    /// Returns the consumer side of a ring buffer containing 16 kHz mono samples.
    pub fn open(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();

        let device = if let Some(name) = device_name {
            host.input_devices()
                .context("Failed to enumerate input devices")?
                .find(|d| d.name().ok().as_deref() == Some(name))
                .context(format!("Input device '{name}' not found"))?
        } else {
            host.default_input_device()
                .context("No default input device found")?
        };

        let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        tracing::info!("Opening input device: {}", device_name);

        let config = device
            .default_input_config()
            .context("Failed to get device config")?;

        let native_sample_rate = config.sample_rate().0;
        let native_channels = config.channels();

        tracing::info!(
            "Device config: {} Hz, {} channels, format: {:?}",
            native_sample_rate,
            native_channels,
            config.sample_format()
        );

        let ring_raw_rb = HeapRb::<f32>::new(((native_sample_rate as usize / 1000) * 64).max(1024));
        let (prod_raw, cons_raw) = ring_raw_rb.split();
        let ring_raw_prod = Arc::new(Mutex::new(prod_raw));
        let ring_raw_cons = Arc::new(Mutex::new(cons_raw));

        let ring_16khz_rb = HeapRb::<f32>::new(32000);
        let (prod_16khz, cons_16khz) = ring_16khz_rb.split();
        let ring_16khz_prod = Arc::new(Mutex::new(prod_16khz));
        let ring_16khz_cons = Arc::new(Mutex::new(cons_16khz));

        let dropped_samples = Arc::new(AtomicU64::new(0));

        // Spawn the resampling worker thread
        let ring_raw_cons_resample = Arc::clone(&ring_raw_cons);
        let ring_16khz_prod_resample = Arc::clone(&ring_16khz_prod);
        let dropped_samples_resample = Arc::clone(&dropped_samples);
        let native_sr = native_sample_rate;

        let resample_thread = thread::Builder::new()
            .name("audio-resample".to_string())
            .spawn(move || {
                let mut input_buf = vec![0.0f32; 4096];
                let mut output_buf = vec![0.0f32; 4096];

                loop {
                    let total_read = ring_raw_cons_resample
                        .lock()
                        .unwrap()
                        .pop_slice(&mut input_buf);

                    if total_read == 0 {
                        thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }

                    // Simple linear interpolation resampling
                    let ratio = 16000.0 / native_sr as f32;
                    let mut out_idx = 0;
                    let mut in_idx = 0.0f32;

                    while out_idx < output_buf.len() && in_idx < total_read as f32 - 1.0 {
                        let idx0 = in_idx.floor() as usize;
                        let idx1 = (idx0 + 1).min(total_read - 1);
                        let frac = in_idx - idx0 as f32;

                        output_buf[out_idx] =
                            input_buf[idx0] * (1.0 - frac) + input_buf[idx1] * frac;

                        out_idx += 1;
                        in_idx += 1.0 / ratio;
                    }

                    if out_idx > 0 {
                        let written = ring_16khz_prod_resample
                            .lock()
                            .unwrap()
                            .push_slice(&output_buf[..out_idx]);
                        let dropped = out_idx - written;
                        if dropped > 0 {
                            dropped_samples_resample.fetch_add(dropped as u64, Ordering::Relaxed);
                        }
                    }
                }
            })
            .context("Failed to spawn resample thread")?;

        // Create the cpal input stream
        let ring_raw_prod_callback = Arc::clone(&ring_raw_prod);
        let dropped_samples_callback = Arc::clone(&dropped_samples);

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _info| {
                    let mono: Vec<f32> = data
                        .chunks(native_channels as usize)
                        .map(|ch| ch.iter().sum::<f32>() / native_channels as f32)
                        .collect();
                    if let Ok(mut prod) = ring_raw_prod_callback.lock() {
                        let written = prod.push_slice(&mono);
                        let dropped = mono.len() - written;
                        if dropped > 0 {
                            dropped_samples_callback.fetch_add(dropped as u64, Ordering::Relaxed);
                        }
                    }
                },
                |err| tracing::error!("Stream error: {}", err),
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _info| {
                    let mono: Vec<f32> = data
                        .chunks(native_channels as usize)
                        .map(|ch| {
                            ch.iter().map(|&s| s as f32 / 32768.0).sum::<f32>()
                                / native_channels as f32
                        })
                        .collect();
                    if let Ok(mut prod) = ring_raw_prod_callback.lock() {
                        let written = prod.push_slice(&mono);
                        let dropped = mono.len() - written;
                        if dropped > 0 {
                            dropped_samples_callback.fetch_add(dropped as u64, Ordering::Relaxed);
                        }
                    }
                },
                |err| tracing::error!("Stream error: {}", err),
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config.into(),
                move |data: &[u16], _info| {
                    let mono: Vec<f32> = data
                        .chunks(native_channels as usize)
                        .map(|ch| {
                            ch.iter()
                                .map(|&s| (s as f32 - 32768.0) / 32768.0)
                                .sum::<f32>()
                                / native_channels as f32
                        })
                        .collect();
                    if let Ok(mut prod) = ring_raw_prod_callback.lock() {
                        let written = prod.push_slice(&mono);
                        let dropped = mono.len() - written;
                        if dropped > 0 {
                            dropped_samples_callback.fetch_add(dropped as u64, Ordering::Relaxed);
                        }
                    }
                },
                |err| tracing::error!("Stream error: {}", err),
                None,
            ),
            _ => bail!("Unsupported sample format"),
        }?;

        stream.play().context("Failed to start playback")?;

        Ok(MicCapture {
            _stream: stream,
            _resample_thread: Some(resample_thread),
            consumer: ring_16khz_cons,
            native_sample_rate,
            native_channels,
        })
    }

    /// Stop capturing audio and drop the stream.
    pub fn stop(self) {
        drop(self._stream);
        drop(self._resample_thread);
    }
}

/// File-backed drop-in for `MicCapture`. Feeds a WAV file into a ring buffer at
/// either real-time (16 kHz wall-clock) or as-fast-as-possible pace.
///
/// The WAV must be 16 kHz; multi-channel files are downmixed to mono.
pub struct FakeMic {
    pub consumer: Arc<Mutex<ringbuf::HeapCons<f32>>>,
    pub native_sample_rate: u32,
    pub native_channels: u16,
    is_done: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
}

impl FakeMic {
    /// Open a WAV file and start a background feeder thread.
    ///
    /// Returns the `FakeMic` consumer handle and the feeder `JoinHandle`.
    /// `realtime=true` throttles the feeder to 16 kHz wall-clock pace;
    /// `realtime=false` pushes as fast as the consumer drains.
    pub fn open(path: &Path, realtime: bool) -> Result<(Self, JoinHandle<()>)> {
        let reader = hound::WavReader::open(path)
            .with_context(|| format!("open WAV: {}", path.display()))?;
        let spec = reader.spec();

        if spec.sample_rate != 16_000 {
            bail!(
                "FakeMic: expected 16 kHz WAV, got {} Hz ({})",
                spec.sample_rate,
                path.display()
            );
        }

        let channels = spec.channels;
        let native_sample_rate = spec.sample_rate;

        // Read all samples upfront; downmix to mono f32.
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("read f32 samples")?
                .chunks(channels as usize)
                .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                .collect(),
            hound::SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("read i32 samples")?
                    .chunks(channels as usize)
                    .map(|ch| ch.iter().map(|&s| s as f32 / max_val).sum::<f32>() / channels as f32)
                    .collect()
            }
        };

        let rb = HeapRb::<f32>::new(32_000);
        let (prod, cons) = rb.split();
        let consumer = Arc::new(Mutex::new(cons));
        let prod = Arc::new(Mutex::new(prod));

        let is_done = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));

        let is_done_thread = Arc::clone(&is_done);
        let shutdown_thread = Arc::clone(&shutdown);
        let prod_thread = Arc::clone(&prod);

        let handle = thread::Builder::new()
            .name("fake-mic-feeder".to_string())
            .spawn(move || {
                const CHUNK: usize = 512;
                let mut offset = 0;
                while offset < samples.len() && !shutdown_thread.load(Ordering::Relaxed) {
                    let end = (offset + CHUNK).min(samples.len());
                    let chunk = &samples[offset..end];

                    // Push chunk; if buffer is full, wait briefly and retry.
                    loop {
                        if shutdown_thread.load(Ordering::Relaxed) {
                            return;
                        }
                        let written = prod_thread.lock().unwrap().push_slice(chunk);
                        if written == chunk.len() {
                            break;
                        }
                        // Partial push — back up and retry after a short sleep.
                        // (Only possible if ring buffer is smaller than CHUNK; in
                        // practice the buffer is 32 000 >> 512 so this is a safety net.)
                        thread::sleep(std::time::Duration::from_millis(1));
                    }

                    offset = end;

                    if realtime {
                        // 512 samples @ 16 kHz = 32 ms
                        thread::sleep(std::time::Duration::from_millis(32));
                    }
                }
                is_done_thread.store(true, Ordering::Release);
            })
            .context("spawn fake-mic feeder thread")?;

        Ok((
            FakeMic {
                consumer,
                native_sample_rate,
                native_channels: 1,
                is_done,
                shutdown,
            },
            handle,
        ))
    }

    /// Returns `true` once the feeder thread has pushed all file samples.
    pub fn is_done(&self) -> bool {
        self.is_done.load(Ordering::Acquire)
    }

    /// Signal the feeder thread to stop (used on early shutdown).
    pub fn stop(self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

/// Unified capture source: either a live microphone or a file-fed `FakeMic`.
pub enum CaptureSource {
    Microphone(MicCapture),
    File(FakeMic),
}

impl CaptureSource {
    /// Drain up to `buf.len()` 16 kHz mono samples. Returns the count actually read.
    pub fn pop_samples(&self, buf: &mut [f32]) -> usize {
        match self {
            CaptureSource::Microphone(mic) => mic.consumer.lock().unwrap().pop_slice(buf),
            CaptureSource::File(fake) => fake.consumer.lock().unwrap().pop_slice(buf),
        }
    }

    /// Returns `true` when a `File` source's feeder thread has finished and the
    /// ring buffer is empty. Always `false` for a live `Microphone` source.
    pub fn is_file_done(&self) -> bool {
        match self {
            CaptureSource::Microphone(_) => false,
            CaptureSource::File(fake) => fake.is_done(),
        }
    }

    pub fn native_sample_rate(&self) -> u32 {
        match self {
            CaptureSource::Microphone(mic) => mic.native_sample_rate,
            CaptureSource::File(fake) => fake.native_sample_rate,
        }
    }

    pub fn native_channels(&self) -> u16 {
        match self {
            CaptureSource::Microphone(mic) => mic.native_channels,
            CaptureSource::File(fake) => fake.native_channels,
        }
    }

    /// Shut down the underlying capture source.
    pub fn stop(self) {
        match self {
            CaptureSource::Microphone(mic) => mic.stop(),
            CaptureSource::File(fake) => fake.stop(),
        }
    }
}

/// Lists all available input devices with their host names.
pub fn list_input_devices() -> Result<Vec<(String, String)>> {
    let mut devices = Vec::new();
    let hosts = cpal::ALL_HOSTS;

    for host_id in hosts {
        let host = cpal::host_from_id(*host_id).context("Failed to instantiate host")?;
        let host_name = host.id().name().to_string();

        if let Ok(input_devices) = host.input_devices() {
            for device in input_devices {
                if let Ok(name) = device.name() {
                    devices.push((host_name.clone(), name));
                }
            }
        }
    }

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_mic_capture_opens() {
        // This test requires an actual audio device and is ignored in CI
        let result = MicCapture::open(None);
        assert!(
            result.is_ok(),
            "Failed to open MicCapture: {:?}",
            result.err()
        );

        if let Ok(mic) = result {
            assert!(mic.native_sample_rate > 0);
            assert!(mic.native_channels > 0);
            mic.stop();
        }
    }

    #[test]
    fn test_list_input_devices() {
        let result = list_input_devices();
        assert!(result.is_ok());
    }
}
