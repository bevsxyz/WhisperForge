/// Streaming latency benchmark binary.
///
/// Run with:
///   cargo run --release -p whisperforge --bin stream_bench -- \
///     --audio test_data/LJ001-0001_16k.wav [--device auto] [--models-dir ./models]
///
/// Prints one JSON line with p50/p99 latencies (ms) for each pipeline stage.
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use ringbuf::traits::Consumer as _;

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn_flex::{Flex, FlexDevice};
use clap::Parser;
use tokenizers::Tokenizer;
use whisperforge_core::{
    Chunker, CommitDelta, Committer, EndpointConfig, Endpointer, FakeMic, PromptContext,
    QualityGate, SileroVad, Whisper, WindowConfig, compute_mel_from_samples, decode_window,
    ensure_silero_model, passes_quality_gate, stream_decode::DecodeContext,
};

#[derive(Parser, Debug)]
#[command(name = "stream_bench", about = "Streaming pipeline latency benchmark")]
struct Args {
    /// Path to the input WAV file (16 kHz mono)
    #[arg(long)]
    audio: PathBuf,

    /// Backend: auto (default), cpu, wgpu, cuda
    #[arg(long, default_value = "auto")]
    device: String,

    /// Directory containing model files (default: ./models)
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx]
}

fn run_bench<B: Backend>(audio: &Path, models_dir: &Path, device: B::Device) -> Result<()> {
    let model_dir = models_dir.join("tiny_en_converted");
    let base = model_dir.join("model");
    let mpk = base.with_extension("mpk");
    if !mpk.exists() {
        anyhow::bail!(
            "model not found at {}. Run `wforge convert` first.",
            mpk.display()
        );
    }

    eprintln!("Loading model: {}", base.display());
    let model: Whisper<B> =
        whisperforge_core::load::load_whisper(base.to_str().context("invalid path")?, &device)?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let tok = |s: &str, fb: u32| tokenizer.token_to_id(s).unwrap_or(fb);
    let sot_token = tok("<|startoftranscript|>", 50258);
    let language_token = tok("<|en|>", 50259);
    let transcribe_token = tok("<|transcribe|>", 50359);
    let eot_token = tok("<|endoftext|>", 50257);
    let no_speech_token = tok("<|nospeech|>", 50362);
    let notimestamps_token = tok("<|notimestamps|>", 50363);
    let timestamp_begin_token = 50364u32;
    let prevtext_token_id = tokenizer.token_to_id("<|prevtext|>");

    eprintln!("Loading Silero VAD…");
    let vad_path = ensure_silero_model(models_dir)?;
    let vad = SileroVad::open(&vad_path)?;

    let window_cfg = WindowConfig {
        max_window_secs: 28.0,
        stride_secs: 1.0,
        vad_threshold: 0.5,
        min_speech_secs: 0.25,
    };
    let mut chunker = Chunker::new(window_cfg, vad);
    let mut committer = Committer::new();
    let mut endpointer = Endpointer::new(EndpointConfig {
        silence_secs: 2.0,
        punct_silence_secs: 0.8,
        min_utterance_secs: 0.5,
    });
    let mut prompt_ctx = PromptContext::new(60, prevtext_token_id);

    eprintln!("Opening file source (no-realtime)…");
    let (fake_mic, _feeder_handle) = FakeMic::open(audio, false)?;

    const WINDOW_SAMPLES: usize = 480_000;
    let mut drain_buf = vec![0.0f32; 4096];

    let mut encoder_ms: Vec<u64> = Vec::new();
    let mut decode_ms: Vec<u64> = Vec::new();
    let mut total_window_ms: Vec<u64> = Vec::new();

    eprintln!("Running pipeline…");
    loop {
        let n = fake_mic.consumer.lock().unwrap().pop_slice(&mut drain_buf);
        if n == 0 {
            if fake_mic.is_done() {
                break;
            }
            thread::sleep(Duration::from_millis(1));
            continue;
        }

        let window = match chunker.push(&drain_buf[..n]) {
            Some(w) => w,
            None => continue,
        };

        if !window.had_speech {
            committer.ingest(vec![]);
            endpointer.step(&window, committer.committed_text());
            continue;
        }

        let window_start = Instant::now();

        let mut padded = window.samples.clone();
        padded.resize(WINDOW_SAMPLES, 0.0);
        let mel =
            compute_mel_from_samples::<B>(&padded, 400, 160, 80, &device).context("compute mel")?;

        let enc_start = Instant::now();
        let encoder_out = model.forward_encoder(mel);
        encoder_ms.push(enc_start.elapsed().as_millis() as u64);

        let ctx = DecodeContext {
            prompt_tokens: prompt_ctx.prompt_tokens(),
            language_token,
            task_token: transcribe_token,
            sot_token,
            eot_token,
            no_speech_token,
            notimestamps_token,
            timestamp_begin_token,
            max_new_tokens: 128,
            no_speech_threshold: 0.6,
        };

        let dec_start = Instant::now();
        let emits = decode_window::<B>(&model, encoder_out, &ctx, &tokenizer, &device)
            .context("decode window")?;
        decode_ms.push(dec_start.elapsed().as_millis() as u64);

        // Apply the same confidence gate as the live loop so total_window_ms reflects
        // the real pipeline (the gate is a cheap post-decode check, included in the
        // total-window timing below, not in decode_ms).
        let emits = if !emits.is_empty() && !passes_quality_gate(&emits, &QualityGate::default()) {
            Vec::new()
        } else {
            emits
        };

        let (_commit_delta, _) = committer.ingest(emits);

        if endpointer.step(&window, committer.committed_text()) || window.forced_eou {
            let final_delta = committer.finalize_utterance();
            if let CommitDelta::Committed { .. } = final_delta {
                prompt_ctx.update_after_eou(committer.committed_tokens(), 128);
            }
            endpointer.reset();
            chunker.reset_utterance();
        }

        total_window_ms.push(window_start.elapsed().as_millis() as u64);
    }

    // Finalize.
    committer.finalize_utterance();

    encoder_ms.sort_unstable();
    decode_ms.sort_unstable();
    total_window_ms.sort_unstable();

    let result = serde_json::json!({
        "encoder_ms_p50": percentile(&encoder_ms, 50.0),
        "encoder_ms_p99": percentile(&encoder_ms, 99.0),
        "decode_ms_p50": percentile(&decode_ms, 50.0),
        "decode_ms_p99": percentile(&decode_ms, 99.0),
        "total_window_ms_p50": percentile(&total_window_ms, 50.0),
        "total_window_ms_p99": percentile(&total_window_ms, 99.0),
        "windows_measured": total_window_ms.len(),
    });

    println!("{result}");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // For stream_bench we default to CPU (Flex) for reproducibility.
    // --device wgpu/cuda are accepted but require the corresponding features.
    match args.device.as_str() {
        "cpu" | "auto" => {
            eprintln!("Backend: Flex (CPU)");
            run_bench::<Flex<f32>>(&args.audio, &args.models_dir, FlexDevice)
        }
        #[cfg(feature = "gpu")]
        "wgpu" => {
            use burn::backend::Wgpu;
            use burn::backend::wgpu::WgpuDevice;
            eprintln!("Backend: WGPU");
            run_bench::<Wgpu>(&args.audio, &args.models_dir, WgpuDevice::default())
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use burn_cuda::{Cuda, CudaDevice};
            eprintln!("Backend: CUDA");
            run_bench::<Cuda<f32, i32>>(&args.audio, &args.models_dir, CudaDevice::default())
        }
        other => anyhow::bail!("unknown --device {other}; supported: auto, cpu, wgpu, cuda"),
    }
}
