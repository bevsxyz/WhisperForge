use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use whisperforge_core::load::{ModelPrecision, WhisperModelConfig};

/// Default models directory used when neither `--models-dir` nor `WF_MODELS_DIR` is set.
pub const DEFAULT_MODELS_DIR: &str = "models";

/// Env var that overrides the default models directory; CLI `--models-dir` still wins.
pub const MODELS_DIR_ENV: &str = "WF_MODELS_DIR";

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

/// Base path (no extension) of a model living under `dir`, e.g. `dir/<name>`.
pub fn model_base_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
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
        if path.extension().is_none_or(|e| e != "mpk") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let cfg_path = path.with_extension("cfg");
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
            "No models found in '{}'. Convert one with `wf convert --output {}/<name>`.",
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
