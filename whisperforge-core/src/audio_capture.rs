use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    HeapRb,
    traits::{Consumer, Producer, Split},
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

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
