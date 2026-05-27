use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;
mod device;
mod stream_ui;

#[derive(Parser, Debug)]
#[command(
    name = "wforge",
    version,
    about = "A fast Whisper transcription tool in Rust"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Transcribe an audio file to text, SRT, or JSON
    Transcribe(commands::transcribe::TranscribeArgs),
    /// Convert a HuggingFace Whisper safetensors model to Burn `.mpk` format
    Convert(commands::convert::ConvertArgs),
    /// List converted models available under the models directory
    ListModels(commands::list_models::ListModelsArgs),
    /// Stream realtime transcription from microphone input
    Stream(commands::stream::StreamArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Transcribe(args) => commands::transcribe::run(args),
        Command::Convert(args) => commands::convert::run(args),
        Command::ListModels(args) => commands::list_models::run(args),
        Command::Stream(args) => commands::stream::run(args),
    }
}
