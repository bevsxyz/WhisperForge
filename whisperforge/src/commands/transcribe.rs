use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn_flex::Flex;
use burn_flex::FlexDevice;
use clap::{Parser, ValueEnum};
use std::cmp::Ordering;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use whisperforge_align::{AudioSegmenter, SrtEntry, SrtWriter};
use whisperforge_core::{
    AudioChunkIterator, DecodingConfig, HybridDecoder, KvCache, Task, TranscriptionResult,
    TranscriptionSegment, Whisper, audio, audio::AudioData, batch_mel_spectrograms,
    detect_language, extract_speaker_embedding, forward_decoder_cached, language_token_id,
    load_whisper, task_token_id,
};
use whisperforge_diarize::SpeakerDiarizer;

use super::list::{model_base_path, model_tokenizer_path, resolve_models_dir};
use crate::device::{DeviceChoice, ResolvedDevice, resolve};

/// CLI surface for the Whisper decode task. `Translate` is X → English only —
/// Whisper has no other-target output (see `whisperforge_core::language`).
#[derive(ValueEnum, Clone, Copy, Debug, Default)]
pub enum TaskArg {
    #[default]
    Transcribe,
    Translate,
}

impl From<TaskArg> for Task {
    fn from(t: TaskArg) -> Self {
        match t {
            TaskArg::Transcribe => Task::Transcribe,
            TaskArg::Translate => Task::Translate,
        }
    }
}

/// Output rendering for the transcript.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum Format {
    #[default]
    Text,
    /// SubRip subtitles (`HH:MM:SS,mmm`) with per-segment timestamps.
    Srt,
    /// WebVTT subtitles (`HH:MM:SS.mmm`) with per-segment timestamps.
    Vtt,
    Json,
}

/// Decoding quality/speed preset. Maps to `DecodingConfig::{fast,balanced,accurate}`.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum Preset {
    Fast,
    #[default]
    Balanced,
    Accurate,
}

impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Preset::Fast => "fast",
            Preset::Balanced => "balanced",
            Preset::Accurate => "accurate",
        };
        f.write_str(s)
    }
}

#[derive(Parser, Debug)]
pub struct TranscribeArgs {
    /// Path to the audio file to transcribe (wav/mp3/flac/ogg/m4a).
    #[arg(value_name = "AUDIO")]
    pub audio_file: Option<String>,

    /// Model name (see `wforge list models`). If omitted on a TTY you'll be prompted;
    /// defaults to `tiny.en` when present.
    #[arg(short, long)]
    pub model: Option<String>,

    /// Spoken-language code (e.g. `en`, `hi`, `es`) or `auto` for first-token detection.
    /// Requires a multilingual model; English-only (.en) models support `en` only.
    ///
    /// Experimental "translate-into-X": forcing a language code that differs from the
    /// spoken audio (with `--task transcribe`) coerces Whisper into emitting that
    /// language. This was never a training objective — best-effort only, works on
    /// `large`/`large-v1`, mostly broken on `large-v2`+, and tends to drift back to the
    /// spoken language on long audio.
    #[arg(short, long, default_value = "en")]
    pub language: String,

    /// Decode task: `transcribe` (output in the spoken language) or `translate`
    /// (X → English only — Whisper cannot output any other target language).
    #[arg(long, value_enum, default_value_t = TaskArg::Transcribe)]
    pub task: TaskArg,

    #[arg(short, long)]
    pub output: Option<String>,

    /// Output format
    #[arg(short = 'f', long, value_enum, default_value_t = Format::Text)]
    pub format: Format,

    /// Decoding preset
    #[arg(long, value_enum, default_value_t = Preset::Balanced)]
    pub preset: Preset,

    /// Backend selection: auto (default), cpu, wgpu, or cuda (feature-gated).
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    pub device: DeviceChoice,

    /// Beam size (overrides preset)
    #[arg(long, help_heading = "Advanced decoding")]
    pub beam_size: Option<usize>,

    /// Temperature for sampling (overrides preset)
    #[arg(long, help_heading = "Advanced decoding")]
    pub temperature: Option<f32>,

    /// Length penalty (overrides preset)
    #[arg(long, help_heading = "Advanced decoding")]
    pub length_penalty: Option<f32>,

    /// No-speech detection threshold
    #[arg(long, help_heading = "Advanced decoding")]
    pub no_speech_threshold: Option<f32>,

    /// Enable voice activity detection (VAD) filtering
    #[arg(long, help_heading = "Voice activity detection")]
    pub vad: bool,

    /// VAD detection threshold (0.0-1.0, higher = stricter)
    #[arg(long, help_heading = "Voice activity detection")]
    pub vad_threshold: Option<f32>,

    /// Enable speaker diarization (assigns SPEAKER_NN labels to segments)
    #[arg(long, help_heading = "Diarization")]
    pub diarize: bool,

    /// Cosine similarity threshold for speaker clustering (0.0–1.0, default 0.7)
    #[arg(long, default_value = "0.7", help_heading = "Diarization")]
    pub diarize_threshold: f32,

    /// Encoder forward-pass batch size. Larger = faster but more VRAM/RAM.
    /// Default: 1 on WGPU (54 MB, safe on any GPU), 4 on `--device cpu` (216 MB, safe on ≥1 GB RAM).
    #[arg(long, help_heading = "Tuning")]
    pub encoder_batch_size: Option<usize>,

    /// Debug inference with different encoder inputs
    #[arg(long, hide = true)]
    pub debug_inference: bool,
}

/// Decode one audio chunk given a pre-computed encoder output `[1, 1500, D]`.
///
/// Separating mel+encode from decode lets callers batch-encode multiple chunks
/// in one GPU call before decoding each sequentially (see `run_backend`).
fn transcribe_chunk<B: Backend>(
    encoder_output: burn::tensor::Tensor<B, 3>,
    model: &Whisper<B>,
    tokenizer: &Tokenizer,
    config: &DecodingConfig,
    device: &B::Device,
) -> Result<(String, Vec<u32>)> {
    let tok = |s: &str, fb: u32| tokenizer.token_to_id(s).unwrap_or(fb);
    let sot = tok("<|startoftranscript|>", 50258);
    let no_timestamps = tok("<|notimestamps|>", 50363);
    let eot = tok("<|endoftext|>", 50257);
    let no_speech = tok("<|nospeech|>", 50362);

    // Resolve language + task tokens from the decode config (no longer hardcoded to English).
    let lang_tok = language_token_id(tokenizer, &config.language).with_context(|| {
        format!(
            "model tokenizer has no <|{}|> token — you're likely using an English-only (.en) \
             model. Get a multilingual model, e.g. `wforge pull small`.",
            config.language
        )
    })?;
    let task_tok = task_token_id(tokenizer, config.task)
        .context("model tokenizer has no task token (<|transcribe|>/<|translate|>)")?;

    let decoder = HybridDecoder::new(config.clone());
    let vocab_size = 51864usize;
    let init_tokens = [sot, lang_tok, task_tok, no_timestamps];
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    let budget = config.max_length.saturating_sub(init_tokens.len());

    // Precompute encoder cross-attention K,V once; warm up self-attention cache
    // with the initial prompt tokens. O(n) per subsequent step instead of O(n²).
    let mut cache = KvCache::new(model, encoder_output);
    let mut step_logits = Vec::new();
    for &tok in &init_tokens {
        step_logits =
            forward_decoder_cached(model, tok, &mut cache, device).context("kv-cache warmup")?;
    }

    for _ in 0..budget {
        let unconstrained = step_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(eot);

        let greedy_next = if all_logits.is_empty() && unconstrained == eot {
            step_logits
                .iter()
                .enumerate()
                .filter(|&(i, _)| i as u32 != eot)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(eot)
        } else {
            unconstrained
        };

        all_logits.push(step_logits);

        if greedy_next == eot {
            break;
        }

        if greedy_next < 50257 {
            if let Ok(word) = tokenizer.decode(&[greedy_next], false) {
                print!("{word}");
                std::io::stdout().flush().ok();
            }
        }

        step_logits = forward_decoder_cached(model, greedy_next, &mut cache, device)
            .context("kv-cache decode step")?;
    }

    if let Some(first) = all_logits.first_mut() {
        if (eot as usize) < first.len() {
            first[eot as usize] = f32::NEG_INFINITY;
        }
    }

    let tokens = decoder.decode_with_fallback(
        &all_logits,
        no_timestamps,
        vocab_size,
        eot,
        no_speech,
        |ids| tokenizer.decode(ids, false).unwrap_or_default(),
    )?;

    let text_tokens: Vec<u32> = tokens.into_iter().filter(|&t| t < 50257).collect();
    let text = tokenizer
        .decode(&text_tokens, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok((text.trim().to_string(), text_tokens))
}

/// Format a `TranscriptionResult` as an SRT subtitle string.
/// When a segment has a speaker label, the text is prefixed `[SPEAKER_NN]: `.
fn result_to_srt(result: &TranscriptionResult) -> String {
    let mut writer = SrtWriter::new();
    for (i, seg) in result.segments.iter().enumerate() {
        let text = match &seg.speaker {
            Some(spk) => format!("[{spk}]: {}", seg.text),
            None => seg.text.clone(),
        };
        writer.add_entry(SrtEntry::new(i + 1, seg.start as f64, seg.end as f64, text));
    }
    writer.to_string()
}

/// Format a `TranscriptionResult` as a WebVTT string (per-segment timestamps).
/// WebVTT differs from SRT only in the header and a `.` (not `,`) before milliseconds.
fn result_to_vtt(result: &TranscriptionResult) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in &result.segments {
        let text = match &seg.speaker {
            Some(spk) => format!("[{spk}]: {}", seg.text),
            None => seg.text.clone(),
        };
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            vtt_time(seg.start as f64),
            vtt_time(seg.end as f64),
            text.trim()
        ));
    }
    out
}

/// Format `seconds` as a WebVTT timestamp `HH:MM:SS.mmm`.
fn vtt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

fn run_backend<B: Backend>(
    args: TranscribeArgs,
    models_dir: Option<PathBuf>,
    device: B::Device,
    is_cpu_backend: bool,
    mel_fn: impl Fn(&[AudioData], &B::Device) -> Result<burn::tensor::Tensor<B, 3>>,
) -> Result<()> {
    let audio_file = args.audio_file.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Audio file is required: wforge transcribe <AUDIO> -m <model>")
    })?;

    let models_dir = resolve_models_dir(models_dir.as_deref());
    let model_name = crate::interactive::resolve_model(&models_dir, args.model.as_deref())?;
    let base = model_base_path(&models_dir, &model_name);

    println!("Loading model: {}", base.display());
    let model: Whisper<B> = load_whisper(
        base.to_str().context("models-dir is not valid UTF-8")?,
        &device,
    )?;
    println!("Model loaded successfully!");

    let tokenizer_path = model_tokenizer_path(&models_dir, &model_name);
    println!("Loading tokenizer from: {}", tokenizer_path.display());
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let mut decoding_config = match args.preset {
        Preset::Fast => DecodingConfig::fast(),
        Preset::Balanced => DecodingConfig::balanced(),
        Preset::Accurate => DecodingConfig::accurate(),
    };
    if let Some(beam_size) = args.beam_size {
        decoding_config = decoding_config.with_beam_size(beam_size);
    }
    if let Some(temperature) = args.temperature {
        decoding_config = decoding_config.with_temperature(temperature);
    }
    if let Some(length_penalty) = args.length_penalty {
        decoding_config = decoding_config.with_length_penalty(length_penalty);
    }
    if let Some(no_speech_threshold) = args.no_speech_threshold {
        decoding_config = decoding_config.with_no_speech_threshold(no_speech_threshold);
    }
    decoding_config = decoding_config
        .with_language(args.language.clone())
        .with_task(args.task.into());

    println!(
        "Decoding config: preset={}, beam_size={}, temperatures={:?}, length_penalty={}",
        args.preset,
        decoding_config.beam_size,
        decoding_config.temperatures,
        decoding_config.length_penalty
    );

    println!("Streaming audio from: {}", audio_file);

    // VAD path: requires full audio, so fall back to eager loading for now
    if args.vad {
        let audio_data = audio::load_audio_file(audio_file)?;
        let processed_audio = audio_data.to_16khz_mono()?;
        println!(
            "Audio loaded: {:.2}s, {}Hz (via VAD)",
            processed_audio.duration(),
            processed_audio.sample_rate
        );

        let vad_threshold = args.vad_threshold.unwrap_or(0.5);
        let segmenter = AudioSegmenter::new(processed_audio.sample_rate)
            .with_vad_threshold(vad_threshold)
            .with_max_segment_length(30.0);
        let segments = segmenter.segment(&processed_audio)?;
        if segments.is_empty() {
            println!("VAD: no voice activity detected — skipping transcription");
            return Ok(());
        }
        println!(
            "VAD: detected {} voice segment(s) (threshold={:.2})",
            segments.len(),
            vad_threshold
        );
        for (i, seg) in segments.iter().enumerate() {
            println!(
                "  [{i}] {:.2}s – {:.2}s ({:.2}s)",
                seg.start_time,
                seg.end_time,
                seg.end_time - seg.start_time,
            );
        }

        let mut segments_out: Vec<TranscriptionSegment> = Vec::new();
        let mut parts: Vec<(String, Vec<f32>)> = Vec::new();

        for (idx, seg) in segments.iter().enumerate() {
            let audio_chunk = seg.to_audio_data(processed_audio.sample_rate);
            let mel = mel_fn(&[audio_chunk], &device).context("mel spectrogram")?;
            let [_, n_mels, n_frames] = mel.dims();
            let mel = if n_frames > 3000 {
                mel.slice([0..1, 0..n_mels, 0..3000])
            } else {
                mel
            };
            let enc = model.forward_encoder(mel);

            // Detect language once (on the first encoder output) when `--language auto`.
            if decoding_config.language == "auto" {
                let sot = tokenizer
                    .token_to_id("<|startoftranscript|>")
                    .unwrap_or(50258);
                let (code, _) = detect_language(&model, enc.clone(), &tokenizer, sot, &device)
                    .context("language auto-detection")?;
                println!("Detected language: {code}");
                decoding_config = decoding_config.with_language(code);
            }

            println!(
                "\n[segment {}/{}  {:.1}s – {:.1}s]",
                idx + 1,
                segments.len(),
                seg.start_time,
                seg.end_time
            );
            print!(">>> ");
            std::io::stdout().flush().ok();

            let speaker_emb = if args.diarize {
                extract_speaker_embedding(enc.clone()).unwrap_or_default()
            } else {
                vec![]
            };

            let (text, tokens) =
                transcribe_chunk(enc, &model, &tokenizer, &decoding_config, &device)?;
            println!();

            if !text.is_empty() {
                segments_out.push(TranscriptionSegment {
                    start: seg.start_time as f32,
                    end: seg.end_time as f32,
                    text: text.clone(),
                    tokens,
                    confidence: 1.0,
                    token_timestamps: vec![],
                    speaker: None,
                });
                parts.push((text, speaker_emb));
            }
        }

        let speaker_embeddings: Vec<Vec<f32>> = parts.iter().map(|(_, e)| e.clone()).collect();
        let texts: Vec<String> = parts.into_iter().map(|(t, _)| t).collect();

        if args.diarize {
            println!("Running speaker diarization...");
            let diarizer = SpeakerDiarizer::new().with_similarity_threshold(args.diarize_threshold);
            let labels = diarizer.assign_labels(&speaker_embeddings);
            for (seg, label) in segments_out.iter_mut().zip(labels) {
                seg.speaker = Some(label);
            }
        }

        let result = TranscriptionResult {
            text: texts.join(" "),
            segments: segments_out,
            language: Some(args.language.clone()),
        };

        let output_body = match args.format {
            Format::Srt => result_to_srt(&result),
            Format::Vtt => result_to_vtt(&result),
            Format::Json => serde_json::to_string_pretty(&result)
                .context("serialising TranscriptionResult to JSON")?,
            Format::Text => result.text.clone(),
        };

        println!("\nTranscription result:\n----------------------------------------");
        println!("{}", output_body);
        println!("----------------------------------------");

        if let Some(output_path) = args.output {
            std::fs::write(&output_path, &output_body)?;
            println!("Saved to: {}", output_path);
        }

        return Ok(());
    }

    // Streaming path (no VAD): on-demand chunk iteration
    let mut stream = AudioChunkIterator::default_whisper(audio_file)?;

    // WGPU default=1 (54 MB; safe after model weights); CPU default=4 (216 MB; safe on ≥1 GB RAM).
    let enc_batch = args
        .encoder_batch_size
        .unwrap_or(if is_cpu_backend { 4 } else { 1 })
        .max(1);

    println!(
        "Transcribing with {} decoding (encoder_batch_size={})...",
        args.preset, enc_batch
    );

    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let mut parts: Vec<(String, Vec<f32>)> = Vec::new();
    let mut chunk_index = 0;

    // Streaming encoder-batch loop: accumulate enc_batch chunks, process them together
    loop {
        let sub: Vec<whisperforge_core::AudioChunk> =
            (&mut stream).take(enc_batch).collect::<Result<_>>()?;
        if sub.is_empty() {
            break;
        }

        let audio_views: Vec<AudioData> = sub
            .iter()
            .map(|c| AudioData::new(c.samples.clone(), 16000, 1))
            .collect();
        let mel = mel_fn(&audio_views, &device).context("mel batch")?;
        let [n_sub, n_mels, n_frames] = mel.dims();
        let mel = if n_frames > 3000 {
            mel.slice([0..n_sub, 0..n_mels, 0..3000])
        } else {
            mel
        };
        let enc = model.forward_encoder(mel);

        // Detect language once (on the first encoder output) when `--language auto`.
        if decoding_config.language == "auto" {
            let [_, frames, d_model] = enc.dims();
            let sot = tokenizer
                .token_to_id("<|startoftranscript|>")
                .unwrap_or(50258);
            let probe = enc.clone().slice([0..1, 0..frames, 0..d_model]);
            let (code, _) = detect_language(&model, probe, &tokenizer, sot, &device)
                .context("language auto-detection")?;
            println!("Detected language: {code}");
            decoding_config = decoding_config.with_language(code);
        }

        for (j, chunk) in sub.iter().enumerate() {
            chunk_index += 1;
            let [_, frames, d_model] = enc.dims();
            let enc_slice = enc.clone().slice([j..j + 1, 0..frames, 0..d_model]);

            println!(
                "\n[chunk {}  {:.1}s – {:.1}s]",
                chunk_index, chunk.start_sec, chunk.end_sec
            );
            print!(">>> ");
            std::io::stdout().flush().ok();

            let speaker_emb = if args.diarize {
                extract_speaker_embedding(enc_slice.clone()).unwrap_or_default()
            } else {
                vec![]
            };

            let (text, tokens) =
                transcribe_chunk(enc_slice, &model, &tokenizer, &decoding_config, &device)?;
            println!();

            if !text.is_empty() {
                segments.push(TranscriptionSegment {
                    start: chunk.start_sec,
                    end: chunk.end_sec,
                    text: text.clone(),
                    tokens,
                    confidence: 1.0,
                    token_timestamps: vec![],
                    speaker: None,
                });
                parts.push((text, speaker_emb));
            }
        }
    }

    let speaker_embeddings: Vec<Vec<f32>> = parts.iter().map(|(_, e)| e.clone()).collect();
    let texts: Vec<String> = parts.into_iter().map(|(t, _)| t).collect();

    // Assign speaker labels when --diarize is set.
    if args.diarize {
        println!("Running speaker diarization...");
        let diarizer = SpeakerDiarizer::new().with_similarity_threshold(args.diarize_threshold);
        let labels = diarizer.assign_labels(&speaker_embeddings);
        for (seg, label) in segments.iter_mut().zip(labels) {
            seg.speaker = Some(label);
        }
    }

    let result = TranscriptionResult {
        text: texts.join(" "),
        segments,
        language: Some(args.language.clone()),
    };

    let output_body = match args.format {
        Format::Srt => result_to_srt(&result),
        Format::Vtt => result_to_vtt(&result),
        Format::Json => serde_json::to_string_pretty(&result)
            .context("serialising TranscriptionResult to JSON")?,
        Format::Text => result.text.clone(),
    };

    println!("\nTranscription result:\n----------------------------------------");
    println!("{}", output_body);
    println!("----------------------------------------");

    if let Some(output_path) = args.output {
        std::fs::write(&output_path, &output_body)?;
        println!("Saved to: {}", output_path);
    }

    Ok(())
}

pub fn run(args: TranscribeArgs, models_dir: Option<PathBuf>) -> Result<()> {
    println!("WhisperForge v{}", env!("CARGO_PKG_VERSION"));

    let resolved = resolve(args.device)?;
    match resolved {
        ResolvedDevice::Cpu => {
            println!("Backend: Flex (CPU)");
            run_backend::<Flex<f32>>(args, models_dir, FlexDevice, true, |chunks, dev| {
                batch_mel_spectrograms::<Flex<f32>>(chunks, 400, 160, 80, dev)
            })
        }
        #[cfg(feature = "gpu")]
        ResolvedDevice::Wgpu => {
            use burn::backend::wgpu::WgpuDevice;
            let device = WgpuDevice::default();

            #[cfg(feature = "cubecl-stft")]
            {
                use burn_wgpu::CubeBackend;
                use whisperforge_core::audio::batch_mel_spectrograms_wgpu;
                println!("Backend: WGPU (GPU, CubeCL STFT)");
                run_backend::<CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>>(
                    args,
                    models_dir,
                    device,
                    false,
                    |chunks, dev| batch_mel_spectrograms_wgpu(chunks, 400, 160, 80, dev),
                )
            }
            #[cfg(not(feature = "cubecl-stft"))]
            {
                use burn::backend::Wgpu;
                println!("Backend: WGPU (GPU)");
                run_backend::<Wgpu>(args, models_dir, device, false, |chunks, dev| {
                    batch_mel_spectrograms::<Wgpu>(chunks, 400, 160, 80, dev)
                })
            }
        }
        #[cfg(feature = "cuda")]
        ResolvedDevice::Cuda => {
            use burn_cuda::{Cuda, CudaDevice};
            type B = Cuda<f32, i32>;
            println!("Backend: CUDA (CubeCL)");
            let device = CudaDevice::default();
            run_backend::<B>(args, models_dir, device, false, |chunks, dev| {
                batch_mel_spectrograms::<B>(chunks, 400, 160, 80, dev)
            })
        }
    }
}
