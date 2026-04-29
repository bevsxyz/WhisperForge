use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArrayDevice;
use clap::Parser;
use std::cmp::Ordering;
use std::io::Write;
use tokenizers::Tokenizer;
use whisperforge_align::{AudioSegmenter, SrtEntry, SrtWriter};
use whisperforge_core::{
    audio, audio::AudioData, forward_decoder_cached, load_whisper, DecodingConfig, HybridDecoder,
    KvCache, TranscriptionResult, TranscriptionSegment, Whisper,
};

#[derive(Parser, Debug)]
#[command(name = "whisperforge")]
#[command(about = "A fast Whisper transcription tool in Rust")]
struct Args {
    #[arg(short, long)]
    audio_file: Option<String>,

    #[arg(short, long, default_value = "tiny_en_converted")]
    model: String,

    #[arg(short, long, default_value = "en")]
    language: String,

    #[arg(short, long)]
    output: Option<String>,

    /// Output format: text, srt, json
    #[arg(long, default_value = "text")]
    output_format: String,

    /// Decoding preset: fast, balanced, accurate
    #[arg(long, default_value = "balanced")]
    decoding_preset: String,

    /// Beam size (overrides preset)
    #[arg(long)]
    beam_size: Option<usize>,

    /// Temperature for sampling (overrides preset)
    #[arg(long)]
    temperature: Option<f32>,

    /// Length penalty (overrides preset)
    #[arg(long)]
    length_penalty: Option<f32>,

    /// No-speech detection threshold
    #[arg(long)]
    no_speech_threshold: Option<f32>,

    /// Task: transcribe or translate
    #[arg(long, default_value = "transcribe")]
    task: String,

    /// Enable voice activity detection (VAD) filtering
    #[arg(long)]
    vad_enabled: bool,

    /// VAD detection threshold (0.0-1.0, higher = stricter)
    #[arg(long)]
    vad_threshold: Option<f32>,

    /// Use the WGPU GPU backend instead of CPU (requires wgpu feature)
    #[arg(long)]
    wgpu: bool,

    /// Debug inference with different encoder inputs
    #[arg(long)]
    debug_inference: bool,
}

/// Transcribe one audio chunk (≤ 30 s), streaming greedy tokens to stdout.
/// Returns the final decoded text and the token ids for segment building.
fn transcribe_chunk<B: Backend>(
    audio: &AudioData,
    model: &Whisper<B>,
    tokenizer: &Tokenizer,
    config: &DecodingConfig,
    device: &B::Device,
) -> Result<(String, Vec<u32>)> {
    let mel = audio::compute_mel_spectrogram(audio, 400, 160, 80, device)?;

    let expected = 3000usize;
    let [batch, n_mels, n_frames] = mel.dims();
    let mel = if n_frames > expected {
        mel.slice([0..batch, 0..n_mels, 0..expected])
    } else {
        mel
    };

    let tok = |s: &str, fb: u32| tokenizer.token_to_id(s).unwrap_or(fb);
    let sot = tok("<|startoftranscript|>", 50258);
    let en = tok("<|en|>", 50259);
    let transcribe_tok = tok("<|transcribe|>", 50359);
    let no_timestamps = tok("<|notimestamps|>", 50363);
    let eot = tok("<|endoftext|>", 50257);
    let no_speech = tok("<|nospeech|>", 50362);

    let encoder_output = model.forward_encoder(mel);
    let decoder = HybridDecoder::new(config.clone());
    let vocab_size = 51864usize;
    let init_tokens = [sot, en, transcribe_tok, no_timestamps];
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

/// Split audio into ≤ 30 s chunks with `overlap_samples` overlap.
fn chunk_audio_fixed(
    audio: &AudioData,
    chunk_samples: usize,
    overlap_samples: usize,
) -> Vec<AudioData> {
    let n = audio.samples.len();
    if n <= chunk_samples {
        return vec![audio.clone()];
    }
    let step = chunk_samples.saturating_sub(overlap_samples).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + chunk_samples).min(n);
        chunks.push(AudioData {
            samples: audio.samples[start..end].to_vec(),
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        });
        if end == n {
            break;
        }
        start += step;
    }
    chunks
}

/// Format a `TranscriptionResult` as an SRT subtitle string.
fn result_to_srt(result: &TranscriptionResult) -> String {
    let mut writer = SrtWriter::new();
    for (i, seg) in result.segments.iter().enumerate() {
        writer.add_entry(SrtEntry::new(
            i + 1,
            seg.start as f64,
            seg.end as f64,
            seg.text.clone(),
        ));
    }
    writer.to_string()
}

fn run<B: Backend>(args: Args, device: B::Device) -> Result<()> {
    let audio_file = args
        .audio_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Audio file is required (use --audio-file)"))?;

    println!("Loading audio: {}", audio_file);
    let audio_data = audio::load_wav_file(audio_file)?;
    let processed_audio = audio_data.to_16khz_mono()?;

    println!(
        "Audio loaded: {:.2}s, {}Hz",
        processed_audio.duration(),
        processed_audio.sample_rate
    );

    let model_path = format!("models/{}", args.model);
    println!("Loading model from: {}", model_path);
    let model: Whisper<B> = load_whisper(&model_path, &device)?;
    println!("Model loaded successfully!");

    let tokenizer_path = "models/tokenizer.json";
    println!("Loading tokenizer from: {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let mut decoding_config = match args.decoding_preset.as_str() {
        "fast" => DecodingConfig::fast(),
        "accurate" => DecodingConfig::accurate(),
        _ => DecodingConfig::balanced(),
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
    decoding_config = decoding_config.with_language(args.language.clone());

    println!(
        "Decoding config: preset={}, beam_size={}, temperatures={:?}, length_penalty={}",
        args.decoding_preset,
        decoding_config.beam_size,
        decoding_config.temperatures,
        decoding_config.length_penalty
    );

    let chunk_samples = 30 * processed_audio.sample_rate as usize;
    let overlap_samples = processed_audio.sample_rate as usize;
    let step_samples = chunk_samples - overlap_samples;

    let chunks: Vec<(AudioData, f32, f32)> = if args.vad_enabled {
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
        segments
            .iter()
            .map(|s| {
                (
                    s.to_audio_data(processed_audio.sample_rate),
                    s.start_time as f32,
                    s.end_time as f32,
                )
            })
            .collect()
    } else {
        chunk_audio_fixed(&processed_audio, chunk_samples, overlap_samples)
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let start = (i * step_samples) as f32 / processed_audio.sample_rate as f32;
                let end = start + chunk.samples.len() as f32 / processed_audio.sample_rate as f32;
                (chunk, start, end)
            })
            .collect()
    };

    let total = chunks.len();
    println!(
        "Transcribing {} chunk(s) with {} decoding...",
        total, args.decoding_preset
    );

    let mut segments: Vec<TranscriptionSegment> = Vec::with_capacity(total);
    let mut parts: Vec<String> = Vec::with_capacity(total);

    for (i, (chunk, start_s, end_s)) in chunks.iter().enumerate() {
        if total > 1 {
            println!(
                "\n[chunk {}/{}  {:.1}s – {:.1}s]",
                i + 1,
                total,
                start_s,
                end_s
            );
        }
        print!(">>> ");
        std::io::stdout().flush().ok();

        let (text, tokens) =
            transcribe_chunk(chunk, &model, &tokenizer, &decoding_config, &device)?;
        println!();

        if !text.is_empty() {
            segments.push(TranscriptionSegment {
                start: *start_s,
                end: *end_s,
                text: text.clone(),
                tokens,
                confidence: 1.0,
                token_timestamps: vec![],
            });
            parts.push(text);
        }
    }

    let result = TranscriptionResult {
        text: parts.join(" "),
        segments,
        language: Some(args.language.clone()),
    };

    let output_body = match args.output_format.as_str() {
        "srt" => result_to_srt(&result),
        "json" => serde_json::to_string_pretty(&result)
            .context("serialising TranscriptionResult to JSON")?,
        _ => result.text.clone(),
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

fn main() -> Result<()> {
    let args = Args::parse();

    println!("WhisperForge v0.1.0");
    println!("Loading model: {}", args.model);

    if args.wgpu {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::Wgpu;
        println!("Backend: WGPU (GPU)");
        let device = WgpuDevice::default();
        return run::<Wgpu>(args, device);
    }

    println!("Backend: NdArray (CPU)");
    let device = NdArrayDevice::default();
    run::<NdArray<f32>>(args, device)
}
