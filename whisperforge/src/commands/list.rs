use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use whisperforge_core::list_input_devices;
use whisperforge_core::load::{ModelPrecision, WhisperModelConfig};

use crate::device::{DeviceChoice, resolve};

/// Project-local models directory, used when present in the cwd (back-compat /
/// project-local override). When absent we fall back to the platform cache dir.
pub const LOCAL_MODELS_DIR: &str = "models";

/// Env var that overrides the default models directory; CLI `--models-dir` still wins.
pub const MODELS_DIR_ENV: &str = "WF_MODELS_DIR";

/// Fixed stem for the weights/config inside a model's own directory
/// (`<models_dir>/<name>/model.mpk` + `model.cfg`).
pub const MODEL_STEM: &str = "model";

/// Per-model tokenizer filename (`<models_dir>/<name>/tokenizer.json`).
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// What `wforge list` should show.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum ListWhat {
    /// Converted models under the models directory.
    Models,
    /// Audio input devices available for `stream`.
    Devices,
    /// Compute backends compiled into this build.
    Backends,
    /// Everything (default).
    #[default]
    All,
}

#[derive(Parser, Debug)]
pub struct ListArgs {
    /// What to list (default: all).
    #[arg(value_enum, default_value_t = ListWhat::All)]
    pub what: ListWhat,

    /// Emit machine-readable JSON instead of formatted tables.
    #[arg(long)]
    pub json: bool,
}

/// Resolve the models directory the user wants to use, honoring (in order):
/// 1. the explicit `--models-dir` flag,
/// 2. the `WF_MODELS_DIR` env var,
/// 3. `./models/` if it exists in the cwd (project-local override / back-compat),
/// 4. the platform cache dir (`~/.cache/whisperforge/models` on Linux, etc.).
///
/// The cache dir is separate from hf-hub's own download cache; only our converted
/// `model.{mpk,cfg}` + `tokenizer.json` land here.
pub fn resolve_models_dir(cli_arg: Option<&Path>) -> PathBuf {
    if let Some(p) = cli_arg {
        return p.to_path_buf();
    }
    if let Ok(env_dir) = std::env::var(MODELS_DIR_ENV)
        && !env_dir.is_empty()
    {
        return PathBuf::from(env_dir);
    }
    let local = PathBuf::from(LOCAL_MODELS_DIR);
    if local.is_dir() {
        return local;
    }
    cache_models_dir().unwrap_or(local)
}

/// Platform cache models directory: `<cache>/whisperforge/models`. `None` only when no
/// home/cache dir can be determined (rare; callers fall back to `./models`).
pub fn cache_models_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "whisperforge")
        .map(|dirs| dirs.cache_dir().join("models"))
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

/// Metadata for one converted model directory, as surfaced by `list models` and the
/// interactive model picker.
#[derive(serde::Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub precision: String,
    pub n_audio_layer: String,
    pub n_mels: String,
    pub size_mb: f64,
}

/// Scan `dir` for converted models (subdirs holding `model.mpk`) and return them sorted
/// by name. The directory is assumed to exist; callers handle the missing-dir case.
pub fn scan_models(dir: &Path) -> Result<Vec<ModelInfo>> {
    let mut rows: Vec<ModelInfo> = Vec::new();
    for entry in std::fs::read_dir(dir)
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
        rows.push(ModelInfo {
            name,
            precision,
            n_audio_layer,
            n_mels,
            size_mb: size as f64 / (1024.0 * 1024.0),
        });
    }

    rows.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(rows)
}

pub fn run(args: ListArgs, models_dir: Option<PathBuf>) -> Result<()> {
    if args.json {
        return run_json(args.what, models_dir.as_deref());
    }
    match args.what {
        ListWhat::Models => print_models(models_dir.as_deref())?,
        ListWhat::Devices => print_devices()?,
        ListWhat::Backends => print_backends()?,
        ListWhat::All => {
            print_models(models_dir.as_deref())?;
            println!();
            print_devices()?;
            println!();
            print_backends()?;
        }
    }
    Ok(())
}

/// Emit the requested section(s) as a single pretty-printed JSON object.
fn run_json(what: ListWhat, models_dir: Option<&Path>) -> Result<()> {
    let mut obj = serde_json::Map::new();
    if matches!(what, ListWhat::Models | ListWhat::All) {
        obj.insert("models".into(), models_json(models_dir)?);
    }
    if matches!(what, ListWhat::Devices | ListWhat::All) {
        obj.insert("devices".into(), devices_json()?);
    }
    if matches!(what, ListWhat::Backends | ListWhat::All) {
        obj.insert("backends".into(), backends_json()?);
    }
    println!("{}", serde_json::to_string_pretty(&obj)?);
    Ok(())
}

fn models_json(models_dir: Option<&Path>) -> Result<serde_json::Value> {
    let dir = resolve_models_dir(models_dir);
    let rows = if dir.exists() {
        scan_models(&dir)?
    } else {
        Vec::new()
    };
    Ok(serde_json::to_value(rows)?)
}

fn devices_json() -> Result<serde_json::Value> {
    let devices = list_input_devices()?;
    let arr: Vec<_> = devices
        .into_iter()
        .map(|(host, name)| serde_json::json!({ "host": host, "name": name }))
        .collect();
    Ok(serde_json::Value::Array(arr))
}

fn backends_json() -> Result<serde_json::Value> {
    let auto = resolve(DeviceChoice::Auto)?;
    Ok(serde_json::json!({
        "cpu": true,
        "wgpu": cfg!(feature = "gpu"),
        "cuda": cfg!(feature = "cuda"),
        "auto": format!("{auto:?}").to_lowercase(),
    }))
}

/// Print the converted-model table (or a `pull` hint when none exist).
fn print_models(models_dir: Option<&Path>) -> Result<()> {
    let dir = resolve_models_dir(models_dir);

    let rows = if dir.exists() {
        scan_models(&dir)?
    } else {
        Vec::new()
    };

    if rows.is_empty() {
        println!(
            "No models found in '{}'. Get one with `wforge pull tiny.en`.",
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

/// Print available audio input devices, grouped by host (for `stream --input-device`).
fn print_devices() -> Result<()> {
    let devices = list_input_devices()?;
    if devices.is_empty() {
        println!("Audio input devices: none found.");
        return Ok(());
    }
    println!("Audio input devices:");
    let mut current_host: Option<&str> = None;
    for (host, name) in &devices {
        if current_host != Some(host.as_str()) {
            println!("  Host: {host}");
            current_host = Some(host.as_str());
        }
        println!("    - {name}");
    }
    Ok(())
}

/// Print which compute backends were compiled in and which one `--device auto` resolves to.
fn print_backends() -> Result<()> {
    println!("Compute backends (compiled into this build):");
    println!("  cpu   : yes (always available)");
    println!(
        "  wgpu  : {}",
        if cfg!(feature = "gpu") {
            "yes"
        } else {
            "no  (rebuild with the default features or `--features gpu`)"
        }
    );
    println!(
        "  cuda  : {}",
        if cfg!(feature = "cuda") {
            "yes"
        } else {
            "no  (rebuild with `--features cuda`)"
        }
    );
    let auto = resolve(DeviceChoice::Auto)?;
    println!("  --device auto resolves to: {auto:?}");
    Ok(())
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
