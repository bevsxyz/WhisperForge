use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};
use ringbuf::traits::Consumer as _ConsumerTrait;
use whisperforge_core::MicCapture;

use super::list_models::{MODELS_DIR_ENV, resolve_models_dir};
use crate::device::{DeviceChoice, resolve};

#[derive(Parser, Debug)]
pub struct StreamArgs {
    /// Model name to use for streaming transcription (required unless --list-input-devices)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Directory to load model `.mpk`/`.cfg` files from. Defaults to `$WF_MODELS_DIR` or `./models/`.
    #[arg(long, env = MODELS_DIR_ENV)]
    pub models_dir: Option<PathBuf>,

    /// Backend selection: auto (default), cpu, wgpu, or cuda (feature-gated).
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    pub device: DeviceChoice,

    /// Optional input device name (default = system default)
    #[arg(long)]
    pub input_device: Option<String>,

    /// List available input devices and exit
    #[arg(long)]
    pub list_input_devices: bool,

    /// Window size in seconds for processing (default 5.0)
    #[arg(long, default_value = "5.0")]
    pub window_secs: f32,

    /// Stride (hop) size in seconds between windows (default 1.0)
    #[arg(long, default_value = "1.0")]
    pub stride_secs: f32,

    /// VAD detection threshold (0.0–1.0, default 0.5)
    #[arg(long, default_value = "0.5")]
    pub vad_threshold: f32,

    /// Hard end-of-utterance silence duration in seconds (default 2.0)
    #[arg(long, default_value = "2.0")]
    pub silence_secs: f32,

    /// Soft EOU (after punctuation) silence duration in seconds (default 0.8)
    #[arg(long, default_value = "0.8")]
    pub punct_silence_secs: f32,

    /// Number of previous tokens to carry over as context (default 60)
    #[arg(long, default_value = "60")]
    pub prompt_tokens: usize,

    /// Output streaming results as NDJSON to stdout
    #[arg(long)]
    pub json: bool,

    /// Record input audio to a WAV file at the specified path
    #[arg(long)]
    pub record_to: Option<PathBuf>,

    /// Append committed transcript lines to a file at the specified path
    #[arg(long)]
    pub transcript_to: Option<PathBuf>,

    /// Disable ANSI color output
    #[arg(long)]
    pub no_color: bool,
}

pub fn run(args: StreamArgs) -> Result<()> {
    if args.list_input_devices {
        list_input_devices()?;
        return Ok(());
    }

    let _model = args.model.context("--model is required")?;
    let _models_dir = resolve_models_dir(args.models_dir.as_deref());
    let _device = resolve(args.device).context("Failed to resolve device")?;

    // Open microphone capture
    let mic =
        MicCapture::open(args.input_device.as_deref()).context("Failed to open microphone")?;

    println!(
        "Recording for 5 seconds... ({}Hz, {} channels native)",
        mic.native_sample_rate, mic.native_channels
    );

    let target_duration = Duration::from_secs(5);
    let start = std::time::Instant::now();
    let mut per_second_buf = Vec::new();
    let mut total_samples = 0u64;

    let mut drain_buf = vec![0.0f32; 4096];
    while start.elapsed() < target_duration {
        let n = mic.consumer.lock().unwrap().pop_slice(&mut drain_buf);

        if n > 0 {
            total_samples += n as u64;
            per_second_buf.extend_from_slice(&drain_buf[..n]);

            // Log every second
            if per_second_buf.len() >= 16000 {
                let rms = compute_rms(&per_second_buf[..16000]);
                println!("samples={} rms={:.6}", total_samples, rms);
                per_second_buf.drain(..16000);
            }
        } else {
            thread::sleep(Duration::from_millis(10));
        }
    }

    // Log final partial second if any
    if !per_second_buf.is_empty() {
        let rms = compute_rms(&per_second_buf);
        println!("samples={} rms={:.6}", total_samples, rms);
    }

    mic.stop();
    Ok(())
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn list_input_devices() -> Result<()> {
    let hosts = cpal::ALL_HOSTS;

    for host_id in hosts {
        let host = cpal::host_from_id(*host_id).context("Failed to instantiate host")?;
        println!("Host: {}", host.id().name());

        let devices = host
            .input_devices()
            .context("Failed to enumerate input devices")?;

        for device in devices {
            let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
            println!("  - {}", name);
        }
    }

    Ok(())
}
