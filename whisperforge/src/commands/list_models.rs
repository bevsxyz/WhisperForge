use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use whisperforge_core::load::{ModelPrecision, WhisperModelConfig};

/// Default models directory used when neither `--models-dir` nor `WF_MODELS_DIR` is set.
pub const DEFAULT_MODELS_DIR: &str = "models";

/// Env var that overrides the default models directory; CLI `--models-dir` still wins.
pub const MODELS_DIR_ENV: &str = "WF_MODELS_DIR";

/// Fixed stem for the weights/config inside a model's own directory
/// (`<models_dir>/<name>/model.mpk` + `model.cfg`).
pub const MODEL_STEM: &str = "model";

/// Per-model tokenizer filename (`<models_dir>/<name>/tokenizer.json`).
pub const TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Parser, Debug)]
pub struct ListModelsArgs {
    /// Directory to scan for `*.mpk` model files. Defaults to `$WF_MODELS_DIR` or `./models/`.
    #[arg(long, env = MODELS_DIR_ENV)]
    pub models_dir: Option<PathBuf>,
}

/// Resolve the models directory the user wants to use, honoring (in order):
/// the explicit `--models-dir` flag, the `WF_MODELS_DIR` env var, then `./models/`.
pub fn resolve_models_dir(cli_arg: Option<&Path>) -> PathBuf {
    if let Some(p) = cli_arg {
        return p.to_path_buf();
    }
    if let Ok(env_dir) = std::env::var(MODELS_DIR_ENV)
        && !env_dir.is_empty()
    {
        return PathBuf::from(env_dir);
    }
    PathBuf::from(DEFAULT_MODELS_DIR)
}

/// Directory holding a single converted model's files (`model.mpk`/`model.cfg`
/// + its own `tokenizer.json`), e.g. `dir/<name>/`.
pub fn model_dir_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

/// Base path (no extension) of a model's weights/config inside its own directory,
/// e.g. `dir/<name>/model`. `load_whisper` appends `.mpk`/`.cfg`.
pub fn model_base_path(dir: &Path, name: &str) -> PathBuf {
    model_dir_path(dir, name).join(MODEL_STEM)
}

/// Path to a model's own `tokenizer.json` (per-model, not shared across the dir).
pub fn model_tokenizer_path(dir: &Path, name: &str) -> PathBuf {
    model_dir_path(dir, name).join(TOKENIZER_FILE)
}

pub fn run(args: ListModelsArgs) -> Result<()> {
    let dir = resolve_models_dir(args.models_dir.as_deref());

    if !dir.exists() {
        anyhow::bail!(
            "models directory '{}' does not exist. Set --models-dir or {} to point at one.",
            dir.display(),
            MODELS_DIR_ENV,
        );
    }

    let mut rows: Vec<Row> = Vec::new();
    for entry in std::fs::read_dir(&dir)
        .with_context(|| format!("reading models directory '{}'", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        // Each model is a directory `<dir>/<name>/` holding `model.mpk` + `model.cfg`.
        if !path.is_dir() {
            continue;
        }
        let mpk_path = path.join(MODEL_STEM).with_extension("mpk");
        if !mpk_path.exists() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let size = std::fs::metadata(&mpk_path).map(|m| m.len()).unwrap_or(0);
        let cfg_path = path.join(MODEL_STEM).with_extension("cfg");
        let (precision, n_audio_layer, n_mels) = match read_cfg(&cfg_path) {
            Ok(cfg) => (
                precision_label(cfg.precision),
                cfg.audio_encoder_config.n_audio_layer.to_string(),
                cfg.audio_encoder_config.n_mels.to_string(),
            ),
            Err(_) => ("?".into(), "?".into(), "?".into()),
        };
        rows.push(Row {
            name,
            precision,
            n_audio_layer,
            n_mels,
            size_mb: size as f64 / (1024.0 * 1024.0),
        });
    }

    rows.sort_by(|a, b| a.name.cmp(&b.name));

    if rows.is_empty() {
        println!(
            "No models found in '{}'. Convert one with `wforge convert --output {}/<name>`.",
            dir.display(),
            dir.display(),
        );
        return Ok(());
    }

    println!("Models in {}:", dir.display());
    println!(
        "{:<28} {:<9} {:>14} {:>7} {:>10}",
        "name", "precision", "n_audio_layer", "n_mels", "size"
    );
    println!("{}", "-".repeat(28 + 1 + 9 + 1 + 14 + 1 + 7 + 1 + 10));
    for row in rows {
        println!(
            "{:<28} {:<9} {:>14} {:>7} {:>9.1}M",
            row.name, row.precision, row.n_audio_layer, row.n_mels, row.size_mb
        );
    }

    Ok(())
}

struct Row {
    name: String,
    precision: String,
    n_audio_layer: String,
    n_mels: String,
    size_mb: f64,
}

fn read_cfg(path: &Path) -> Result<WhisperModelConfig> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading config {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing config {}", path.display()))
}

fn precision_label(p: Option<ModelPrecision>) -> String {
    match p {
        Some(ModelPrecision::Fp32) => "fp32".into(),
        Some(ModelPrecision::Int8) => "int8".into(),
        None => "fp32?".into(),
    }
}
