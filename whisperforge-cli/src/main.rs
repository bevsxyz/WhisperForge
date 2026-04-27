use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use burn_ndarray::NdArrayDevice;
use clap::Parser;
use tokenizers::Tokenizer;
use whisperforge_core::{audio, load_whisper, DecodingConfig, HybridDecoder, Whisper};

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

    /// Debug inference with different encoder inputs
    #[arg(long)]
    debug_inference: bool,
}

type Backend = NdArray<f32>;

fn main() -> Result<()> {
    let args = Args::parse();

    println!("WhisperForge v0.1.0");

    let audio_file = args
        .audio_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Audio file is required (use --audio-file)"))?;

    println!("Loading model: {}", args.model);

    // Load audio
    println!("Loading audio: {}", audio_file);
    let audio_data = audio::load_wav_file(audio_file)?;
    let processed_audio = audio_data.to_16khz_mono()?;

    println!(
        "Audio loaded: {:.2}s, {}Hz",
        processed_audio.duration(),
        processed_audio.sample_rate
    );

    // Set up device
    let device = NdArrayDevice::default();

    // Determine model path
    let model_path = format!("models/{}", args.model);
    println!("Loading model from: {}", model_path);

    // Load model using the new loader
    let model: Whisper<Backend> = load_whisper(&model_path, &device)?;
    println!("Model loaded successfully!");

    // Load tokenizer
    let tokenizer_path = "models/tokenizer.json";
    println!("Loading tokenizer from: {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Set up decoding config based on preset and overrides
    let mut decoding_config = match args.decoding_preset.as_str() {
        "fast" => DecodingConfig::fast(),
        "accurate" => DecodingConfig::accurate(),
        _ => DecodingConfig::balanced(), // default to balanced
    };

    // Apply parameter overrides
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

    if args.vad_enabled {
        let vad_threshold = args.vad_threshold.unwrap_or(0.5);
        println!("VAD enabled: threshold={}", vad_threshold);
    }

    // Compute mel spectrogram
    println!("Computing mel spectrogram...");
    let mel_features = audio::compute_mel_spectrogram(
        &processed_audio,
        400, // n_fft
        160, // hop_length
        80,  // n_mels
        &device,
    )?;

    // Whisper expects exactly 3000 mel frames (30 seconds at 100 fps)
    // n_audio_ctx = 1500 (after conv downsampling by 2)
    let expected_frames = 3000;
    let [batch, n_mels, n_frames] = mel_features.dims();

    let mel_features = if n_frames < expected_frames {
        // Pad with zeros
        let padding =
            Tensor::<Backend, 3>::zeros([batch, n_mels, expected_frames - n_frames], &device);
        Tensor::cat(vec![mel_features, padding], 2)
    } else if n_frames > expected_frames {
        // Trim
        mel_features.slice([0..batch, 0..n_mels, 0..expected_frames])
    } else {
        mel_features
    };

    println!("Mel spectrogram shape: {:?}", mel_features.dims());

    // Prepare start tokens for English model
    // <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
    let sot = tokenizer
        .token_to_id("<|startoftranscript|>")
        .unwrap_or(50258);
    let en = tokenizer.token_to_id("<|en|>").unwrap_or(50259);
    let transcribe = tokenizer.token_to_id("<|transcribe|>").unwrap_or(50359);
    let no_timestamps = tokenizer.token_to_id("<|notimestamps|>").unwrap_or(50363);
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(50257);

    println!(
        "Special tokens - SOT: {}, EN: {}, TRANSCRIBE: {}, NO_TS: {}, EOT: {}",
        sot, en, transcribe, no_timestamps, eot
    );

    println!("Transcribing with {} decoding...", args.decoding_preset);

    // Encode audio
    let encoder_output = model.forward_encoder(mel_features);
    println!("Encoder output shape: {:?}", encoder_output.dims());

    // Check encoder output statistics (only on debug)
    if args.debug_inference {
        let enc_data = encoder_output.to_data();
        let enc_vec: Vec<f32> = enc_data.to_vec().unwrap();
        let enc_min = enc_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let enc_max = enc_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let enc_mean = enc_vec.iter().sum::<f32>() / enc_vec.len() as f32;

        println!(
            "Encoder output stats: Min={:.4}, Max={:.4}, Mean={:.4}",
            enc_min, enc_max, enc_mean
        );
    }

    // Start with initial tokens
    let initial_tokens: Vec<u32> = vec![sot, en, transcribe, no_timestamps];
    println!("Initial tokens: {:?}", initial_tokens);

    let decoder = HybridDecoder::new(decoding_config.clone());
    let vocab_size = 51864usize;
    let mut context_tokens = initial_tokens.clone();
    let mut all_step_logits: Vec<Vec<f32>> = Vec::new();

    println!("Transcribing (max {} tokens)...", decoding_config.max_length);

    for _step in 0..decoding_config.max_length {
        let token_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(
                context_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                [1, context_tokens.len()],
            ),
            &device,
        );

        let logits = model.forward_decoder(token_tensor, encoder_output.clone());
        let [batch, seq_len, _] = logits.dims();
        let step_probs: Vec<f32> = logits
            .slice([0..batch, (seq_len - 1)..seq_len, 0..vocab_size])
            .squeeze::<1>()
            .into_data()
            .to_vec()
            .with_context(|| "Failed to extract step logits")?;

        // Greedy peek: determine next token for context feeding and EOT detection.
        let greedy_next = step_probs
            .iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &f32), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as u32)
            .unwrap_or(eot);

        all_step_logits.push(step_probs);

        if greedy_next == eot {
            break;
        }
        context_tokens.push(greedy_next);
    }

    // Apply quality-gated temperature fallback over collected logits.
    let no_speech_token = tokenizer.token_to_id("<|nospeech|>").unwrap_or(50362);
    let tokens: Vec<u32> = decoder.decode_with_fallback(
        &all_step_logits,
        no_timestamps,
        vocab_size,
        eot,
        no_speech_token,
        |ids| tokenizer.decode(ids, false).unwrap_or_default(),
    )?;

    // Remove special tokens and decode
    let output_tokens: Vec<u32> = tokens
        .into_iter()
        .filter(|&t| t < 50257) // Filter out special tokens
        .collect();

    let text = tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;

    println!("\nTranscription result:\n----------------------------------------");
    println!("{}", text);
    println!("----------------------------------------");

    if let Some(output_path) = args.output {
        std::fs::write(&output_path, &text)?;
        println!("Saved to: {}", output_path);
    }

    Ok(())
}
