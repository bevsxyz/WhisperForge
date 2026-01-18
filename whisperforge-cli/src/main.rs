use anyhow::Result;
use clap::Parser;
use whisperforge_core::{audio, WhisperConfig, WhisperModel};
use burn::backend::{NdArray, NdArrayDevice};

#[derive(Parser, Debug)]
#[command(name = "whisperforge")]
#[command(about = "A fast Whisper transcription tool in Rust")]
struct Args {
    #[arg(short, long)]
    audio_file: String,

    #[arg(short, long, default_value = "tiny")]
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
    println!("Audio loaded: {:.2}s, {}Hz", processed_audio.duration(), processed_audio.sample_rate);

    // Set up device and model
    let device = NdArrayDevice::default();
    
    let config = match args.model.as_str() {
        "tiny" => WhisperConfig::tiny(),
        "base" => WhisperConfig::base(),
        "small" => WhisperConfig::small(),
        "medium" => WhisperConfig::medium(),
        "large-v2" => WhisperConfig::large_v2(),
        _ => return Err(anyhow::anyhow!("Unsupported model: {}", args.model)),
    };

    println!("Creating model...");
    let model = WhisperModel::<NdArray>::new(config, &device);

    // Compute mel spectrogram
    println!("Computing mel spectrogram...");
    let mel_features = audio::compute_mel_spectrogram(
        &processed_audio,
        400,  // n_fft
        160,  // hop_length
        80,   // n_mels
        &device,
    )?;

    println!("Mel spectrogram shape: {:?}", mel_features.dims());

    // For now, just show success (actual transcription will be implemented next)
    println!("Model loaded successfully!");
    println!("Ready for transcription (implementation in progress)");

    if let Some(output_path) = args.output {
        println!("Output would be saved to: {}", output_path);
    }

    Ok(())
}
