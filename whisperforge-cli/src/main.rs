use anyhow::Result;
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
        "Decoding config: preset={}, beam_size={}, temperature={}, length_penalty={}",
        args.decoding_preset,
        decoding_config.beam_size,
        decoding_config.temperature,
        decoding_config.length_penalty
    );

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
    let mut tokens: Vec<u32> = vec![sot, en, transcribe, no_timestamps];
    println!("Initial tokens: {:?}", tokens);
    let max_tokens = 224; // Max decoder context is 448, leave room

    // Decode using selected strategy
    // Note: Currently using greedy approach in loop for compatibility
    // Beam search decoder integration coming in next iteration
    let _decoder = HybridDecoder::new(decoding_config);

    for step in 0..max_tokens {
        // Create token tensor
        let token_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(
                tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                [1, tokens.len()],
            ),
            &device,
        );

        // Get logits
        let logits = model.forward_decoder(token_tensor, encoder_output.clone());
        let logits_shape = logits.dims();

        // Get last token's logits
        let batch_size = logits_shape[0];
        let seq_len = logits_shape[1];
        let vocab_size = logits_shape[2];
        let last_logits = logits.slice([0..batch_size, (seq_len - 1)..seq_len, 0..vocab_size]);
        // Shape is [1, 1, 51864] -> squeeze sequence dimension to get [1, 51864]
        let last_logits = last_logits.squeeze::<1>();

        // Debug: Analyze logits for first few steps
        if args.debug_inference && step < 5 {
            let logits_data = last_logits.to_data();
            let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();

            // Find top 5 tokens and their probabilities
            let mut token_probs: Vec<(usize, f32)> = logits_vec
                .iter()
                .enumerate()
                .map(|(i, &prob)| (i, prob))
                .collect();
            token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\nStep {} logits analysis:", step);
            println!("  Current tokens: {:?}", &tokens);
            println!("  Top 5 tokens and probs:");
            for (i, (token_id, prob)) in token_probs.iter().take(5).enumerate() {
                let token_str = tokenizer
                    .decode(&[*token_id as u32], false)
                    .unwrap_or("<?>".to_string());
                println!(
                    "    {}: token={} ({}) prob={:.4}",
                    i + 1,
                    token_id,
                    token_str,
                    prob
                );
            }

            // Check specific tokens
            let eot_prob = logits_vec[eot as usize];
            let transcribe_prob = logits_vec[transcribe as usize];
            println!("  EOT ({}): prob={:.4}", eot, eot_prob);
            println!("  TRANSCRIBE ({}): prob={:.4}", transcribe, transcribe_prob);
        }

        let next_token = last_logits.argmax(0).into_scalar() as u32;
        if args.debug_inference || step < 10 {
            println!(
                "Step {}: selected token {} ({})",
                step,
                next_token,
                tokenizer
                    .decode(&[next_token], false)
                    .unwrap_or("<?>".to_string())
            );
        }

        // For debugging: allow a few steps even with EOT to see if model can generate content
        if next_token == eot && step > 2 {
            println!("EOT detected after {} steps, stopping generation", step);
            break;
        }

        // Safety check: prevent infinite loops
        if tokens.len() >= 50 {
            println!("Reached max token limit (50), stopping generation");
            break;
        }

        tokens.push(next_token);
    }

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
