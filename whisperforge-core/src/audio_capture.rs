use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    HeapRb,
    traits::{Consumer, Producer, Split},
};
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

const RESAMPLER_CHUNK: usize = 1024;

/// Captures audio from a microphone and resamples it to 16 kHz mono.
pub struct MicCapture {
    _stream: cpal::Stream,
    _resample_thread: Option<std::thread::JoinHandle<()>>,
    pub consumer: Arc<Mutex<ringbuf::HeapCons<f32>>>,
    pub native_sample_rate: u32,
    pub native_channels: u16,
    /// Cumulative samples dropped because a downstream ring filled up. Non-zero
    /// values indicate the consumer (typically the decoder) is not keeping up
    /// with real-time audio arrival.
    pub dropped_samples: Arc<AtomicU64>,
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

        // Native-rate ring sized to a couple of seconds so callback bursts never block the
        // audio thread. cpal callbacks typically deliver 10–40 ms per fire, so this is generous.
        let ring_raw_rb = HeapRb::<f32>::new(((native_sample_rate as usize) * 2).max(4096));
        let (prod_raw, cons_raw) = ring_raw_rb.split();

        let ring_16khz_rb = HeapRb::<f32>::new(32_000);
        let (prod_16khz, cons_16khz) = ring_16khz_rb.split();
        let ring_16khz_cons = Arc::new(Mutex::new(cons_16khz));

        let dropped_samples = Arc::new(AtomicU64::new(0));

        // Spawn the resampling worker thread — moves `cons_raw` and `prod_16khz` in
        // directly; ringbuf SPSC needs no synchronization.
        let dropped_samples_resample = Arc::clone(&dropped_samples);
        let native_sr = native_sample_rate;

        let resample_thread = thread::Builder::new()
            .name("audio-resample".to_string())
            .spawn(move || {
                run_resample(native_sr, cons_raw, prod_16khz, dropped_samples_resample);
            })
            .context("Failed to spawn resample thread")?;

        // Build the cpal input stream — `prod_raw` is moved into the matching arm's
        // closure (cpal picks exactly one format, so the producer has exactly one owner)
        // so no Arc<Mutex<>> is needed around the audio callback's producer.
        let dropped_samples_callback = Arc::clone(&dropped_samples);

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let mut prod = prod_raw;
                let dropped_cb = dropped_samples_callback;
                device.build_input_stream(
                    &config.into(),
                    move |data: &[f32], _info| {
                        let mono: Vec<f32> = data
                            .chunks_exact(native_channels as usize)
                            .map(|ch| ch.iter().sum::<f32>() / native_channels as f32)
                            .collect();
                        let written = prod.push_slice(&mono);
                        let dropped = mono.len() - written;
                        if dropped > 0 {
                            dropped_cb.fetch_add(dropped as u64, Ordering::Relaxed);
                        }
                    },
                    |err| tracing::error!("Stream error: {}", err),
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let mut prod = prod_raw;
                let dropped_cb = dropped_samples_callback;
                device.build_input_stream(
                    &config.into(),
                    move |data: &[i16], _info| {
                        let mono: Vec<f32> = data
                            .chunks_exact(native_channels as usize)
                            .map(|ch| {
                                ch.iter().map(|&s| s as f32 / 32_768.0).sum::<f32>()
                                    / native_channels as f32
                            })
                            .collect();
                        let written = prod.push_slice(&mono);
                        let dropped = mono.len() - written;
                        if dropped > 0 {
                            dropped_cb.fetch_add(dropped as u64, Ordering::Relaxed);
                        }
                    },
                    |err| tracing::error!("Stream error: {}", err),
                    None,
                )
            }
            cpal::SampleFormat::U16 => {
                let mut prod = prod_raw;
                let dropped_cb = dropped_samples_callback;
                device.build_input_stream(
                    &config.into(),
                    move |data: &[u16], _info| {
                        let mono: Vec<f32> = data
                            .chunks_exact(native_channels as usize)
                            .map(|ch| {
                                ch.iter()
                                    .map(|&s| (s as f32 - 32_768.0) / 32_768.0)
                                    .sum::<f32>()
                                    / native_channels as f32
                            })
                            .collect();
                        let written = prod.push_slice(&mono);
                        let dropped = mono.len() - written;
                        if dropped > 0 {
                            dropped_cb.fetch_add(dropped as u64, Ordering::Relaxed);
                        }
                    },
                    |err| tracing::error!("Stream error: {}", err),
                    None,
                )
            }
            _ => bail!("Unsupported sample format"),
        }?;

        stream.play().context("Failed to start playback")?;

        Ok(MicCapture {
            _stream: stream,
            _resample_thread: Some(resample_thread),
            consumer: ring_16khz_cons,
            native_sample_rate,
            native_channels,
            dropped_samples,
        })
    }

    /// Stop capturing audio and drop the stream.
    pub fn stop(self) {
        drop(self._stream);
        drop(self._resample_thread);
    }
}

/// Worker body: drain `cons_raw` at native rate, resample to 16 kHz mono with rubato,
/// push into `prod_16khz`. Loops forever until the upstream ring is dropped (shutdown).
fn run_resample(
    native_sr: u32,
    mut cons_raw: ringbuf::HeapCons<f32>,
    mut prod_16khz: ringbuf::HeapProd<f32>,
    dropped_samples: Arc<AtomicU64>,
) {
    // Fast path: device is already 16 kHz, just forward without resampling.
    if native_sr == 16_000 {
        let mut buf = vec![0.0f32; 4096];
        loop {
            let n = cons_raw.pop_slice(&mut buf);
            if n == 0 {
                thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            let written = prod_16khz.push_slice(&buf[..n]);
            let dropped = n - written;
            if dropped > 0 {
                dropped_samples.fetch_add(dropped as u64, Ordering::Relaxed);
            }
        }
    }

    // FFT resampler: fixed input chunk, mono, 1.1× ratio headroom isn't required for synchronous.
    let mut resampler = match Fft::<f32>::new(
        native_sr as usize,
        16_000,
        RESAMPLER_CHUNK,
        2,
        1,
        FixedSync::Input,
    ) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("rubato resampler construction failed ({native_sr} Hz → 16 kHz): {e}");
            return;
        }
    };

    let max_in = resampler.input_frames_max();
    let max_out = resampler.output_frames_max();
    let mut input_buf = vec![0.0f32; max_in];
    let mut output_buf = vec![0.0f32; max_out];

    loop {
        let chunk_in = resampler.input_frames_next();
        let mut filled = 0;
        while filled < chunk_in {
            let n = cons_raw.pop_slice(&mut input_buf[filled..chunk_in]);
            if n == 0 {
                thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            filled += n;
        }

        let in_adapter = match SequentialSlice::new(&input_buf[..chunk_in], 1, chunk_in) {
            Ok(a) => a,
            Err(e) => {
                tracing::error!("rubato input adapter: {e:?}");
                return;
            }
        };
        let mut out_adapter = match SequentialSlice::new_mut(&mut output_buf[..max_out], 1, max_out)
        {
            Ok(a) => a,
            Err(e) => {
                tracing::error!("rubato output adapter: {e:?}");
                return;
            }
        };
        let out_n = match resampler.process_into_buffer(&in_adapter, &mut out_adapter, None) {
            Ok((_in_used, out_n)) => out_n,
            Err(e) => {
                tracing::error!("rubato resample failed: {e:?}");
                return;
            }
        };

        if out_n > 0 {
            let written = prod_16khz.push_slice(&output_buf[..out_n]);
            let dropped = out_n - written;
            if dropped > 0 {
                dropped_samples.fetch_add(dropped as u64, Ordering::Relaxed);
            }
        }
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

    /// Cumulative samples dropped due to a full downstream ring. Always 0 for file sources.
    /// A growing value means the consumer (decoder) can't keep up with real-time audio.
    pub fn dropped_samples(&self) -> u64 {
        match self {
            CaptureSource::Microphone(mic) => mic.dropped_samples.load(Ordering::Relaxed),
            CaptureSource::File(_) => 0,
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
