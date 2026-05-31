//! Interactive fallbacks for missing CLI arguments.
//!
//! When a required model isn't given (or doesn't exist) and we're attached to a TTY,
//! offer a picker over the converted models instead of erroring out. In non-interactive
//! contexts (piped/scripted) we error with a clear pointer at `wforge list models` and
//! `wforge pull`.

use std::io::IsTerminal;
use std::path::Path;

use anyhow::{Context, Result};
use dialoguer::Select;

use crate::commands::list::scan_models;

/// True when both stdin and stdout are TTYs (safe to show an interactive prompt).
pub fn is_interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

/// Resolve a usable model name. `requested` is the user's `-m/--model` value (if any).
///
/// - requested + exists → returns it.
/// - requested + missing → picker on a TTY, else a not-found error.
/// - omitted → the sole model if there's exactly one; otherwise picker on a TTY,
///   `tiny.en` if present, else an error pointing at `list models` / `pull`.
pub fn resolve_model(models_dir: &Path, requested: Option<&str>) -> Result<String> {
    let available: Vec<String> = if models_dir.exists() {
        scan_models(models_dir)?
            .into_iter()
            .map(|m| m.name)
            .collect()
    } else {
        Vec::new()
    };

    if let Some(name) = requested {
        if available.iter().any(|m| m == name) {
            return Ok(name.to_string());
        }
        if is_interactive() && !available.is_empty() {
            eprintln!("Model '{name}' not found in {}.", models_dir.display());
            return pick(&available);
        }
        anyhow::bail!(not_found_msg(name, models_dir));
    }

    if available.len() == 1 {
        return Ok(available.into_iter().next().expect("len checked"));
    }
    if is_interactive() && !available.is_empty() {
        return pick(&available);
    }
    if available.iter().any(|m| m == "tiny.en") {
        return Ok("tiny.en".to_string());
    }
    anyhow::bail!(no_model_msg(models_dir));
}

/// Show a Select over `models`; returns the chosen name or errors if cancelled.
fn pick(models: &[String]) -> Result<String> {
    let idx = Select::new()
        .with_prompt("Select a model")
        .items(models)
        .default(0)
        .interact_opt()
        .context("reading model selection")?
        .context("no model selected")?;
    Ok(models[idx].clone())
}

fn not_found_msg(name: &str, models_dir: &Path) -> String {
    format!(
        "model '{name}' not found in {}.\n  See what's available: wforge list models\n  Get it: wforge pull {name}",
        models_dir.display()
    )
}

fn no_model_msg(models_dir: &Path) -> String {
    format!(
        "no model specified and none usable in {}.\n  Pick one with -m <name> (see: wforge list models)\n  Or get one: wforge pull tiny.en",
        models_dir.display()
    )
}
