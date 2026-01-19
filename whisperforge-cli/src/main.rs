use anyhow::Result;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use burn_ndarray::NdArrayDevice;
use clap::Parser;
use tokenizers::Tokenizer;
use whisperforge_core::{Whisper, audio, load_whisper};

#[derive(Parser, Debug)]
#[command(name = "whisperforge")]
#[command(about = "A fast Whisper transcription tool in Rust")]
struct Args {
    #[arg(short, long)]
    audio_file: String,

    #[arg(short, long, default_value = "tiny_en_converted")]
    model: String,

    #[arg(short, long, default_value = "en")]
    language: String,

    #[arg(short, long)]
    output: Option<String>,
}

type Backend = NdArray<f32>;

fn main() -> Result<()> {
    let args = Args::parse();

    println!("WhisperForge v0.1.0");
    println!("Loading model: {}", args.model);

    // Load audio
    println!("Loading audio: {}", args.audio_file);
    let audio_data = audio::load_wav_file(&args.audio_file)?;
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

    // Simple greedy decoding
    println!("Transcribing...");

    // Encode audio
    let encoder_output = model.forward_encoder(mel_features);
    println!("Encoder output shape: {:?}", encoder_output.dims());

    // Start with initial tokens
    let mut tokens: Vec<u32> = vec![sot, en, transcribe, no_timestamps];
    let max_tokens = 224; // Max decoder context is 448, leave room

    for _ in 0..max_tokens {
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
        // println!("Logits shape: {:?}", logits_shape);
        // std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Get last token's logits and find argmax
        let batch_size = logits_shape[0];
        let seq_len = logits_shape[1];
        let vocab_size = logits_shape[2];
        let last_logits = logits.slice([0..batch_size, (seq_len - 1)..seq_len, 0..vocab_size]);
        // Shape is [1, 1, 51864] -> squeeze sequence dimension to get [1, 51864]
        let last_logits = last_logits.squeeze::<1>();

        let next_token = last_logits.argmax(0).into_scalar() as u32;

        if next_token == eot {
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
