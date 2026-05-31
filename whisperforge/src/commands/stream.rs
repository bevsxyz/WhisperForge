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
use tokenizers::Tokenizer;
use whisperforge_core::{
    CaptureSource, Chunker, CommitDelta, Committer, EndpointConfig, Endpointer, FakeMic,
    MicCapture, PromptContext, QualityGate, SileroVad, Whisper, WindowConfig, avg_logprob,
    compute_mel_from_samples, decode_window, detect_language, ensure_silero_model,
    language_token_id, list_input_devices as core_list_input_devices, passes_quality_gate,
    stream_decode::DecodeContext, task_token_id,
};

use super::list_models::{MODELS_DIR_ENV, model_base_path, resolve_models_dir};
use super::transcribe::TaskArg;
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

    /// Spoken-language code (e.g. `en`, `hi`, `es`) or `auto` to detect on the first
    /// speech window. Requires a multilingual model; `.en` models support `en` only.
    #[arg(short, long, default_value = "en")]
    pub language: String,

    /// Decode task: `transcribe` (output in the spoken language) or `translate`
    /// (X → English only — Whisper cannot output any other target language).
    #[arg(long, value_enum, default_value_t = TaskArg::Transcribe)]
    pub task: TaskArg,

    /// Optional input device name (default = system default)
    #[arg(long)]
    pub input_device: Option<String>,

    /// List available input devices and exit
    #[arg(long)]
    pub list_input_devices: bool,

    /// Hard cap on the growing decode buffer in seconds (default 5.0). When the buffer
    /// hits this cap the chunker forces an EOU and the utterance splits into a new
    /// endpoint event; long utterances therefore transcribe as multiple consecutive
    /// endpoints rather than one continuous committed line. 5.0 s bounds per-window
    /// decode at ~25 tokens × ~100 ms/token ≈ 2.5 s on tiny.en CPU, comfortably under
    /// `--stride-secs`. Raise this only if your hardware sustains shorter per-window
    /// decode at larger windows (otherwise the cpal ring drops samples, logged to stderr).
    #[arg(long, default_value = "5.0")]
    pub max_window_secs: f32,

    /// Stride (hop) size in seconds between windows (default 1.0). Each stride triggers a
    /// full re-encode + re-decode of the growing buffer; with the 5 s default cap this
    /// costs ~2.5 s worst-case on tiny.en + CPU, so 1.0 keeps committed-text latency low
    /// while staying ahead of real-time. Raise to 2.0 if your hardware can't keep up.
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

    /// Number of previously-committed tokens to feed as <|prevtext|> on each window
    /// (default 0 — disabled). Non-zero values can lock the decoder into hallucination
    /// loops on quiet audio (the model keeps repeating the last phrase). Re-enable with
    /// e.g. `--prompt-tokens 60` if you need topical continuity across utterances.
    #[arg(long, default_value = "0")]
    pub prompt_tokens: usize,

    /// Reject a decoded window whose average token log-probability falls below this
    /// (default -1.0, matching faster-whisper's `log_prob_threshold`). Low-confidence
    /// windows are dropped before the committer sees them. Lower (more negative) keeps
    /// more output; raise toward 0 to be stricter.
    #[arg(long, default_value = "-1.0")]
    pub logprob_threshold: f32,

    /// Reject a decoded window whose gzip compression ratio exceeds this (default 2.4,
    /// matching faster-whisper's `compression_ratio_threshold`). High ratios signal a
    /// repetition/hallucination loop (the `*sigh* *sigh*` failure mode).
    #[arg(long, default_value = "2.4")]
    pub compression_ratio_threshold: f32,

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

    /// Read audio from a WAV file instead of the microphone (16 kHz mono expected)
    #[arg(long)]
    pub from_file: Option<PathBuf>,

    /// With --from-file: push samples as fast as possible (no real-time throttle)
    #[arg(long)]
    pub no_realtime: bool,
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

    // Task token (transcribe vs translate-to-English).
    let task_token = task_token_id(&tokenizer, args.task.into())
        .context("model tokenizer has no task token (<|transcribe|>/<|translate|>)")?;

    // Language: resolve an explicit code now, or detect on the first speech window for `auto`.
    // `language_token` stays mutable so auto-detect can lock it in mid-session.
    let auto_detect_language = args.language == "auto";
    let mut language_detected = !auto_detect_language;
    let mut language_token = if auto_detect_language {
        tok("<|en|>", 50259) // provisional until the first speech window detects
    } else {
        language_token_id(&tokenizer, &args.language).with_context(|| {
            format!(
                "model tokenizer has no <|{}|> token — you're likely using an English-only (.en) \
                 model. Convert a multilingual model, e.g. `wforge convert --model-id \
                 openai/whisper-small`.",
                args.language
            )
        })?
    };

    let eot_token = tok("<|endoftext|>", 50257);
    let no_speech_token = tok("<|nospeech|>", 50362);
    let notimestamps_token = tok("<|notimestamps|>", 50363);
    let timestamp_begin_token = 50364u32;
    let prevtext_token_id = tokenizer.token_to_id("<|prevtext|>");

    eprintln!("Ensuring Silero VAD model…");
    let vad_path = ensure_silero_model(&models_dir)?;
    let vad = SileroVad::open(&vad_path)?;

    let window_cfg = WindowConfig {
        max_window_secs: args.max_window_secs,
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
    let quality_gate = QualityGate {
        log_prob_threshold: args.logprob_threshold,
        compression_ratio_threshold: args.compression_ratio_threshold,
    };

    let source = if let Some(ref path) = args.from_file {
        eprintln!("Reading from file: {}", path.display());
        let (fake_mic, _feeder_handle) =
            FakeMic::open(path, !args.no_realtime).context("open file source")?;
        CaptureSource::File(fake_mic)
    } else {
        eprintln!("Opening microphone…");
        CaptureSource::Microphone(
            MicCapture::open(args.input_device.as_deref()).context("open microphone")?,
        )
    };
    eprintln!(
        "Streaming at {}Hz / {} ch native → 16 kHz mono. Press Ctrl-C to stop.",
        source.native_sample_rate(),
        source.native_channels()
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
    const SILENCE_FRAME: [f32; 512] = [0.0f32; 512]; // 32 ms of silence at 16 kHz

    let mut drain_buf = vec![0.0f32; 4096];
    let mut utterance_committed = String::new();
    let mut utterance_start_secs = 0.0f32;
    let mut utterance_started = false;
    // Wall-clock seconds of synthesised silence since the last real audio sample. Reset on
    // any non-empty pop_samples. Used to (a) keep the chunker striding so the endpointer can
    // fire naturally after speech stops, and (b) cleanly shut down file-fed runs.
    let mut silence_pushed_secs = 0.0f32;
    // Drop telemetry: log to stderr whenever the cumulative dropped count grows.
    // A growing count means the decoder isn't draining the 16 kHz ring fast enough
    // and audio is being silently lost — corrupts VAD state and transcripts.
    let mut last_dropped = 0u64;
    let mut last_drop_log = std::time::Instant::now();

    while !shutdown.load(Ordering::SeqCst) {
        let n = source.pop_samples(&mut drain_buf);

        // Surface dropped-sample counter once per second when it changes.
        if last_drop_log.elapsed() >= Duration::from_secs(1) {
            let total = source.dropped_samples();
            if total > last_dropped {
                let delta = total - last_dropped;
                eprintln!(
                    "[audio] dropped {} samples last second ({:.2} s of audio); total dropped: {} ({:.2} s). Decoder is not keeping up with real-time input.",
                    delta,
                    delta as f32 / 16_000.0,
                    total,
                    total as f32 / 16_000.0
                );
                last_dropped = total;
            }
            last_drop_log = std::time::Instant::now();
        }

        let window = if n > 0 {
            silence_pushed_secs = 0.0;

            if let Some(writer) = wav_writer.as_mut() {
                for &sample in &drain_buf[..n] {
                    writer.write_sample(sample).context("write WAV sample")?;
                }
            }

            match chunker.push(&drain_buf[..n]) {
                Some(w) => w,
                None => continue,
            }
        } else {
            // No new audio: synthesise a 32 ms silence frame so the chunker keeps striding
            // and `trailing_silence_secs` grows. The endpointer can then fire naturally
            // after `args.silence_secs` of post-speech silence, even when a file source
            // has reached EOF.
            //
            // Shutdown gate: when the source is finished AND nothing is pending in the
            // current utterance buffer AND we've already drained ~silence_secs of trailing
            // silence past the last EOU, stop the loop.
            if source.is_file_done()
                && utterance_committed.is_empty()
                && !utterance_started
                && silence_pushed_secs > args.silence_secs + 0.5
            {
                shutdown.store(true, Ordering::SeqCst);
                continue;
            }
            silence_pushed_secs += SILENCE_FRAME.len() as f32 / 16_000.0;
            thread::sleep(Duration::from_millis(32));
            match chunker.push(&SILENCE_FRAME) {
                Some(w) => w,
                None => continue,
            }
        };

        let emits = if window.had_speech {
            let mut padded = window.samples.clone();
            padded.resize(WINDOW_SAMPLES, 0.0);

            let t_mel_start = std::time::Instant::now();
            let mel = compute_mel_from_samples::<B>(&padded, 400, 160, 80, &device)
                .context("compute mel")?;
            let mel_ms = t_mel_start.elapsed().as_millis();

            let t_enc_start = std::time::Instant::now();
            let encoder_out = model.forward_encoder(mel);
            let enc_ms = t_enc_start.elapsed().as_millis();

            // `--language auto`: detect on the first speech window, then lock for the session.
            if !language_detected {
                let (code, id) =
                    detect_language(&model, encoder_out.clone(), &tokenizer, sot_token, &device)
                        .context("language auto-detection")?;
                eprintln!("Detected language: {code}");
                language_token = id;
                language_detected = true;
            }

            let ctx = DecodeContext {
                prompt_tokens: prompt_ctx.prompt_tokens(),
                language_token,
                task_token,
                sot_token,
                eot_token,
                no_speech_token,
                notimestamps_token,
                timestamp_begin_token,
                // Cap autoregressive decode work per window. At ~100 ms/token on tiny.en CPU,
                // 32 bounds worst-case window cost at ~3.2 s — comfortable headroom over the
                // 5 s growing buffer's natural ~25-token ceiling. Whisper loop-failure modes
                // self-terminate in ~3 s.
                max_new_tokens: 32,
                no_speech_threshold: 0.6,
            };

            let t_dec_start = std::time::Instant::now();
            let out = decode_window::<B>(&model, encoder_out, &ctx, &tokenizer, &device)
                .context("decode window")?;
            let dec_ms = t_dec_start.elapsed().as_millis();

            tracing::debug!(
                window_secs = window.samples.len() as f32 / 16_000.0,
                real_secs = window.real_samples as f32 / 16_000.0,
                mel_ms,
                enc_ms,
                dec_ms,
                total_ms = mel_ms + enc_ms + dec_ms,
                "per-window latency"
            );

            out
        } else {
            Vec::new()
        };

        let content_logprobs: Vec<f32> = emits
            .iter()
            .filter(|t| !t.is_special)
            .map(|t| t.logprob)
            .collect();
        let (window_avg_lp, window_min_lp) = if content_logprobs.is_empty() {
            (f32::NAN, f32::NAN)
        } else {
            (
                avg_logprob(&emits),
                content_logprobs
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min),
            )
        };
        tracing::debug!(
            n_tokens = emits.len(),
            n_content = content_logprobs.len(),
            window_avg_lp,
            window_min_lp,
            cap_hit = window.cap_hit,
            had_speech = window.had_speech,
            "window decode metrics"
        );
        sink.on_decode_metrics(
            window_avg_lp,
            window_min_lp,
            content_logprobs.len(),
            window.cap_hit,
        )?;

        // Confidence gate (faster-whisper parity): drop low-confidence or repetition-loop
        // windows before the committer sees them. LocalAgreement-2 only rejects *unstable*
        // output, so a confident hallucination loop would otherwise commit. Dropping reduces
        // the window to an empty ingest — already a normal per-frame occurrence on silence.
        let emits = if !emits.is_empty() && !passes_quality_gate(&emits, &quality_gate) {
            tracing::debug!(window_avg_lp, "window rejected by quality gate");
            Vec::new()
        } else {
            emits
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

        let endpoint_fires = endpointer.step(&window, committer.committed_text());
        if endpoint_fires || window.forced_eou {
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
            chunker.reset_utterance();
            utterance_committed.clear();
            utterance_started = false;
        } else if window.cap_hit {
            // Stride-based buffer trim: drop the oldest ~max_window/2 samples and clear
            // the committer's last_candidate so the next stride establishes a fresh
            // baseline. This keeps the utterance going as one continuous commit stream
            // without an endpoint event at the cap.
            //
            // We trim only when the committer has finalized at least some content this
            // utterance (committed_tokens non-empty) — otherwise the trim could discard
            // audio the decoder hasn't yet produced stable output for. If the trim path
            // is unavailable, leave cap_pending_handler set so the next cap escalates
            // to forced_eou (the safety-net branch above).
            //
            // The trim point is intentionally NOT timestamp-anchored: with `<|notimestamps|>`
            // in the streaming decode init, the model doesn't emit segment timestamps in
            // greedy mode. A timestamp-anchored design was prototyped but greedy decode
            // on tiny.en with timestamps on emits mostly timestamps and little content.
            // Half-buffer trim is the pragmatic alternative — accepts one "wasted" stride
            // post-trim (LCP starts fresh) in exchange for long-form continuity.
            // Trim 1.5 s — empirically the sweet spot. Larger trim (e.g. half-buffer)
            // shrinks the post-trim overlap, the non-causal encoder produces different
            // tokens for short vs. long buffers of the same audio, and LCP regresses at
            // the seam (no commits between caps). 1.5 s leaves 3.5 s of overlap which is
            // enough for stable LCP commits.
            let target_samples = (1.5_f32 * 16_000.0) as usize;
            let has_commits = !committer.committed_tokens().is_empty();
            let trimmed = if has_commits {
                chunker.trim_oldest(target_samples)
            } else {
                0
            };

            if trimmed > 0 {
                let trim_delta = committer.on_trim();
                if let CommitDelta::Committed { ref new_text, .. } = trim_delta {
                    if !new_text.is_empty() {
                        utterance_committed.push_str(new_text);
                        sink.on_commit(new_text, window.window_start_secs)?;
                    }
                }
                tracing::debug!(trimmed_samples = trimmed, "cap-hit trim applied");
            } else {
                tracing::debug!(
                    has_commits,
                    "cap-hit trim skipped (no prior commits); next cap will force EOU"
                );
            }
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
    source.stop();
    sink.close()?;
    Ok(())
}

fn list_input_devices() -> Result<()> {
    let devices = core_list_input_devices()?;
    let mut current_host: Option<&str> = None;
    for (host, name) in &devices {
        if current_host != Some(host.as_str()) {
            println!("Host: {host}");
            current_host = Some(host.as_str());
        }
        println!("  - {name}");
    }
    Ok(())
}
