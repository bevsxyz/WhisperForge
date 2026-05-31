use std::path::PathBuf;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;

mod commands;
mod device;
mod interactive;
mod stream_ui;

use commands::list::MODELS_DIR_ENV;

#[derive(Parser, Debug)]
#[command(
    name = "wforge",
    version,
    about = "A fast Whisper transcription tool in Rust"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,

    /// Directory holding converted models. Defaults to `$WF_MODELS_DIR`, then `./models/`
    /// if present, then the platform cache dir (`~/.cache/whisperforge/models`).
    #[arg(long, env = MODELS_DIR_ENV, global = true)]
    models_dir: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Transcribe an audio file to text, SRT, or JSON
    Transcribe(commands::transcribe::TranscribeArgs),
    /// Download (or import) a Whisper model and convert it to Burn `.mpk` format
    #[command(alias = "convert")]
    Pull(commands::pull::PullArgs),
    /// List models, audio devices, or compute backends
    List(commands::list::ListArgs),
    /// Stream realtime transcription from microphone input
    Stream(commands::stream::StreamArgs),
    /// Generate shell completion script (bash, zsh, fish, powershell, elvish)
    Completions {
        /// Target shell
        #[arg(value_enum)]
        shell: Shell,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let models_dir = cli.models_dir;
    match cli.cmd {
        Command::Transcribe(args) => commands::transcribe::run(args, models_dir),
        Command::Pull(args) => commands::pull::run(args, models_dir),
        Command::List(args) => commands::list::run(args, models_dir),
        Command::Stream(args) => commands::stream::run(args, models_dir),
        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            clap_complete::generate(shell, &mut cmd, "wforge", &mut std::io::stdout());
            Ok(())
        }
    }
}
