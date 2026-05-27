use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn_flex::{Flex, FlexDevice};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};
use ringbuf::traits::Consumer as _;
use tokenizers::Tokenizer;
use whisperforge_core::{
    Chunker, CommitDelta, Committer, EndpointConfig, Endpointer, MicCapture, PromptContext,
    SileroVad, Whisper, WindowConfig, compute_mel_from_samples, decode_window, ensure_silero_model,
    stream_decode::DecodeContext,
};

use super::list_models::{MODELS_DIR_ENV, model_base_path, resolve_models_dir};
use crate::device::{DeviceChoice, ResolvedDevice, resolve};
use crate::stream_ui::{FileTranscriptSink, JsonSink, MultiSink, StreamSink, TerminalSink};

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

    let resolved = resolve(args.device).context("Failed to resolve device")?;

    match resolved {
        ResolvedDevice::Cpu => {
            eprintln!("Backend: Flex (CPU)");
            run_stream::<Flex<f32>>(args, FlexDevice)
        }
        #[cfg(feature = "gpu")]
        ResolvedDevice::Wgpu => {
            use burn::backend::wgpu::WgpuDevice;
            let device = WgpuDevice::default();
            #[cfg(feature = "cubecl-stft")]
            {
                use burn_wgpu::CubeBackend;
                eprintln!("Backend: WGPU (GPU, CubeCL STFT)");
                run_stream::<CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>>(args, device)
            }
            #[cfg(not(feature = "cubecl-stft"))]
            {
                use burn::backend::Wgpu;
                eprintln!("Backend: WGPU (GPU)");
                run_stream::<Wgpu>(args, device)
            }
        }
        #[cfg(feature = "cuda")]
        ResolvedDevice::Cuda => {
            use burn_cuda::{Cuda, CudaDevice};
            type B = Cuda<f32, i32>;
            eprintln!("Backend: CUDA (CubeCL)");
            run_stream::<B>(args, CudaDevice::default())
        }
    }
}

fn run_stream<B: Backend>(args: StreamArgs, device: B::Device) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .with_writer(std::io::stderr)
        .try_init()
        .ok();

    let model_name = args.model.as_deref().context("--model is required")?;
    let models_dir = resolve_models_dir(args.models_dir.as_deref());
    let base = model_base_path(&models_dir, model_name);
    let mpk_path = base.with_extension("mpk");
    if !mpk_path.exists() {
        anyhow::bail!(
            "model '{model_name}' not found in {}. Run `wforge list-models` to see available models.",
            models_dir.display(),
        );
    }

    eprintln!("Loading model: {}", base.display());
    let model: Whisper<B> =
        whisperforge_core::load::load_whisper(base.to_str().context("invalid path")?, &device)?;

    let tokenizer_path = models_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let tok = |s: &str, fb: u32| tokenizer.token_to_id(s).unwrap_or(fb);
    let sot_token = tok("<|startoftranscript|>", 50258);
    let language_token = tok("<|en|>", 50259);
    let transcribe_token = tok("<|transcribe|>", 50359);
    let eot_token = tok("<|endoftext|>", 50257);
    let no_speech_token = tok("<|nospeech|>", 50362);
    let timestamp_begin_token = 50364u32;
    let prevtext_token_id = tokenizer.token_to_id("<|prevtext|>");

    eprintln!("Ensuring Silero VAD model…");
    let vad_path = ensure_silero_model(&models_dir)?;
    let vad = SileroVad::open(&vad_path)?;

    let window_cfg = WindowConfig {
        window_secs: args.window_secs,
        stride_secs: args.stride_secs,
        vad_threshold: args.vad_threshold,
        min_speech_secs: 0.25,
    };
    let mut chunker = Chunker::new(window_cfg, vad);
    let mut committer = Committer::new();
    let mut endpointer = Endpointer::new(EndpointConfig {
        silence_secs: args.silence_secs,
        punct_silence_secs: args.punct_silence_secs,
        min_utterance_secs: 0.5,
    });
    let mut prompt_ctx = PromptContext::new(args.prompt_tokens, prevtext_token_id);

    eprintln!("Opening microphone…");
    let mic = MicCapture::open(args.input_device.as_deref()).context("open microphone")?;
    eprintln!(
        "Streaming at {}Hz / {} ch native → 16 kHz mono. Press Ctrl-C to stop.",
        mic.native_sample_rate, mic.native_channels
    );

    let mut sinks: Vec<Box<dyn StreamSink>> = Vec::new();
    if args.json {
        sinks.push(Box::new(JsonSink::new(io::stdout())));
    } else {
        sinks.push(Box::new(TerminalSink::new(!args.no_color)));
    }
    if let Some(path) = &args.transcript_to {
        sinks.push(Box::new(FileTranscriptSink::open(path)?));
    }
    let mut sink: Box<dyn StreamSink> = if sinks.len() == 1 {
        sinks.remove(0)
    } else {
        Box::new(MultiSink::new(sinks))
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime for ctrl-c");
            rt.block_on(async {
                tokio::signal::ctrl_c().await.ok();
            });
            shutdown.store(true, Ordering::SeqCst);
        });
    }

    let mut wav_writer: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> = args
        .record_to
        .as_ref()
        .map(|path| {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            hound::WavWriter::create(path, spec)
        })
        .transpose()
        .context("open WAV writer for --record-to")?;

    const WINDOW_SAMPLES: usize = 480_000; // 30 s at 16 kHz — Whisper's expected input length

    let mut drain_buf = vec![0.0f32; 4096];
    let mut utterance_committed = String::new();
    let mut utterance_start_secs = 0.0f32;
    let mut utterance_started = false;

    while !shutdown.load(Ordering::SeqCst) {
        let n = mic.consumer.lock().unwrap().pop_slice(&mut drain_buf);
        if n == 0 {
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        if let Some(writer) = wav_writer.as_mut() {
            for &sample in &drain_buf[..n] {
                writer.write_sample(sample).context("write WAV sample")?;
            }
        }

        let window = match chunker.push(&drain_buf[..n]) {
            Some(w) => w,
            None => continue,
        };

        let emits = if window.had_speech {
            let mut padded = window.samples.clone();
            padded.resize(WINDOW_SAMPLES, 0.0);

            let mel = compute_mel_from_samples::<B>(&padded, 400, 160, 80, &device)
                .context("compute mel")?;
            let encoder_out = model.forward_encoder(mel);

            let ctx = DecodeContext {
                prompt_tokens: prompt_ctx.prompt_tokens(),
                language_token,
                transcribe_token,
                sot_token,
                eot_token,
                no_speech_token,
                timestamp_begin_token,
                max_new_tokens: 128,
                no_speech_threshold: 0.6,
            };

            decode_window::<B>(&model, encoder_out, &ctx, &tokenizer, &device)
                .context("decode window")?
        } else {
            Vec::new()
        };

        let (commit_delta, tentative_delta) = committer.ingest(emits);

        if let CommitDelta::Committed { ref new_text, .. } = commit_delta {
            if !new_text.is_empty() {
                if !utterance_started {
                    utterance_start_secs = window.window_start_secs;
                    utterance_started = true;
                }
                utterance_committed.push_str(new_text);
                sink.on_commit(new_text, window.window_start_secs)?;
            }
        }

        let tentative_text = match &tentative_delta {
            CommitDelta::Tentative { text, .. } => text.clone(),
            CommitDelta::Committed { .. } => String::new(),
        };

        sink.on_partial(&utterance_committed, &tentative_text)?;

        if endpointer.step(&window, committer.committed_text()) {
            let final_delta = committer.finalize_utterance();
            if let CommitDelta::Committed { ref new_text, .. } = final_delta {
                if !new_text.is_empty() {
                    utterance_committed.push_str(new_text);
                }
            }

            if !utterance_committed.is_empty() {
                let end_secs = window.window_start_secs + window.real_samples as f32 / 16_000.0;
                sink.on_endpoint(&utterance_committed, utterance_start_secs, end_secs)?;
            }

            prompt_ctx.update_after_eou(committer.committed_tokens(), 128);
            endpointer.reset();
            utterance_committed.clear();
            utterance_started = false;
        }
    }

    // Finalize any pending partial utterance on shutdown.
    let final_delta = committer.finalize_utterance();
    if let CommitDelta::Committed { ref new_text, .. } = final_delta {
        if !new_text.is_empty() {
            utterance_committed.push_str(new_text);
        }
    }
    if !utterance_committed.is_empty() {
        let end_secs = chunker.total_samples_seen as f32 / 16_000.0;
        sink.on_endpoint(&utterance_committed, utterance_start_secs, end_secs)?;
    }

    if let Some(writer) = wav_writer.take() {
        writer.finalize().context("finalize WAV writer")?;
    }
    mic.stop();
    sink.close()?;
    Ok(())
}

fn list_input_devices() -> Result<()> {
    for host_id in cpal::ALL_HOSTS {
        let host = cpal::host_from_id(*host_id).context("instantiate host")?;
        println!("Host: {}", host.id().name());
        let devices = host.input_devices().context("enumerate input devices")?;
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
            println!("  - {name}");
        }
    }
    Ok(())
}
