use anyhow::Result;
use burn::{
    backend::NdArray,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
};
use burn_ndarray::NdArrayDevice;
use clap::Parser;
use tokenizers::Tokenizer;
use whisperforge_core::{audio, WhisperConfig, WhisperModel};

#[derive(Parser, Debug)]
#[command(name = "whisperforge")]
#[command(about = "A fast Whisper transcription tool in Rust")]
struct Args {
    #[arg(short, long)]
    audio_file: String,

    #[arg(short, long, default_value = "tiny.en")]
    model: String,

    #[arg(short, long, default_value = "en")]
    language: String,

    #[arg(short, long)]
    output: Option<String>,
}

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

    // Set up device and model
    let device = NdArrayDevice::default();

    let (config, model_filename) = match args.model.as_str() {
        "tiny.en" | "tiny" => (WhisperConfig::tiny_en(), "whisper-tiny.en"),
        "base" => (WhisperConfig::base(), "whisper-base"),
        "small" => (WhisperConfig::small(), "whisper-small"),
        "medium" => (WhisperConfig::medium(), "whisper-medium"),
        "large-v2" => (WhisperConfig::large_v2(), "whisper-large-v2"),
        _ => return Err(anyhow::anyhow!("Unsupported model: {}", args.model)),
    };

    println!("Creating model...");
    let model = WhisperModel::<NdArray<f32>>::new(config, &device);

    // Load model weights
    let model_path = format!("models/{}.mpk", model_filename);
    println!("Loading weights from: {}", model_path);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(model_path.into(), &device).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load model weights from {}: {}",
            model_filename,
            e
        )
    })?;

    let model = model.load_record(record);

    // Load tokenizer
    let tokenizer_path = format!("models/{}-tokenizer.json", model_filename);
    println!("Loading tokenizer from: {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
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

    println!("Mel spectrogram shape: {:?}", mel_features.dims());

    // Prepare start tokens for English model
    // <|startoftranscript|> <|transcribe|> <|notimestamps|>
    let sot = tokenizer
        .token_to_id("<|startoftranscript|>")
        .unwrap_or(50257); // Fallback
    let transcribe = tokenizer.token_to_id("<|transcribe|>").unwrap_or(50358);
    let no_timestamps = tokenizer.token_to_id("<|notimestamps|>").unwrap_or(50362);
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256);

    let start_tokens = vec![sot, transcribe, no_timestamps];
    println!("Start tokens: {:?}", start_tokens);

    // Transcribe
    println!("Transcribing...");
    let result_tokens = model.generate_greedy(
        mel_features,
        start_tokens,
        100, // Max len
        eot,
    )?;

    // Decode tokens
    if let Some(tokens) = result_tokens.first() {
        let text = tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!("\nTranscription result:\n----------------------------------------");
        println!("{}", text);
        println!("----------------------------------------");

        if let Some(output_path) = args.output {
            std::fs::write(&output_path, &text)?;
            println!("Saved to: {}", output_path);
        }
    }

    Ok(())
}
