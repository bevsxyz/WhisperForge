# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Check compilation
cargo check --all

# Tests — ALWAYS exclude whisperforge-align (has known pre-existing failures)
cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli

# Single test with output
cargo test -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture --exact

# Format + lint (run before every commit)
cargo fmt --all && cargo clippy --all-targets --all-features

# Run the CLI
cargo run -p whisperforge-cli -- -a audio.wav -m tiny_en_converted

# Backtrace on failure
RUST_BACKTRACE=1 cargo test test_name -- --nocapture
```

Model files (`.mpk`, `.cfg`, tokenizer) are git-ignored. Download from HuggingFace and convert with `whisperforge-convert`.

## Architecture

Five-crate Rust workspace using [Burn 0.20](https://burn.dev/) for CUDA-accelerated ML inference.

| Crate | Role |
|-------|------|
| `whisperforge-core` | Whisper model + SOTA decoding — primary development area |
| `whisperforge-cli` | `whisperforge` binary |
| `whisperforge-convert` | One-shot HuggingFace safetensors → Burn NamedMpk conversion |
| `whisperforge-align` | VAD, segmentation, SRT output (has known test failures; work cautiously) |
| `whisperforge-diarize` | Placeholder — not yet integrated |

### Data flow

```
Audio → FFmpeg/hound load → resample to 16 kHz mono
  → Mel spectrogram (STFT: N_FFT=400, hop=160, n_mels=80)
  → Encoder (16 layers, multi-head self-attention)
  → Decoder (16 layers, cross-attention to encoder)
  → SOTA decoding (beam search → quality check → temperature fallback)
  → TranscriptionResult
```

Key files in `whisperforge-core`:
- [src/model.rs](whisperforge-core/src/model.rs) — **FROZEN**. Do not modify; architecture is locked at Burn 0.20.
- [src/load.rs](whisperforge-core/src/load.rs) — loads converted `.mpk` + JSON config via `NamedMpkFileRecorder`.
- [src/audio.rs](whisperforge-core/src/audio.rs) — mel spectrogram, audio I/O.
- [src/decoding.rs](whisperforge-core/src/decoding.rs) — `DecodingConfig`, `BeamSearchDecoder`, `GreedyDecoder`, `HybridDecoder`. Active development area.

All model types are generic over `B: Backend`. Default alias is `NdArray<f32>` (CPU); swap to `Cuda` for GPU.

## Decoding: current state vs plan

The decoding module (`decoding.rs`) implements beam search and a greedy fallback, but **does not yet implement** the full faster-whisper SOTA strategy. What's still missing:

- Temperature fallback sequence `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` — currently the config has a single `temperature` scalar, not a `Vec<f32>`.
- Quality metrics — `compression_ratio_threshold` and `log_prob_threshold` exist in `DEVELOPMENT_PLAN.md` but are **not** in the actual `DecodingConfig` struct. The `HybridDecoder` only falls back on beam search failure, not on quality assessment.
- `best_of`, `patience`, `repetition_penalty`, `condition_on_previous_text` — planned but absent.

When extending decoding, add `flate2` for the compression ratio metric (`len(text) / len(gzip(text))`).

## Hard-won lessons from this codebase

These are non-obvious issues that caused real bugs; read before touching the relevant code.

### Cross-attention weight names in model conversion

The EOT-domination bug (model predicts `<|endoftext|>` immediately) was caused by incorrect tensor name mapping in `whisperforge-convert/src/convert.rs`. OpenAI safetensors use `decoder.layers.X.encoder_attn.*`; the Burn model uses `blocks.X.cross_attn.*`. Verify this mapping exactly if conversion produces pathological outputs.

### Audio: do NOT use `hound::into_samples::<f32>()`

This produced zero spectrograms. The correct approach is manual `i16 → f32` conversion: divide the raw `i16` sample by `i16::MAX as f32`. See [whisperforge-core/src/audio.rs](whisperforge-core/src/audio.rs) for the pattern.

### Burn 0.20 API differences

- `.squeeze()` takes **no arguments** in 0.20 (changed from older versions).
- Always use `NamedMpkFileRecorder::<FullPrecisionSettings>::new()` for saving/loading models.
- The Burn API evolves quickly; when a function can't be found, search the crate source rather than assuming the name from older docs.

### whisperforge-align test failures

The align crate has pre-existing test failures unrelated to any current work. Always exclude it: `cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli`. Don't spend time investigating failures there unless align is the task at hand.

## Code conventions

- **Error handling**: `anyhow::Result<T>` everywhere; `.with_context(|| format!("…"))` on every `?`. No `.unwrap()` outside `main()` or tests.
- **Imports**: std → external crates (alphabetical) → local `crate::` modules. No glob imports outside `#[cfg(test)]`.
- **Hot paths**: no tensor clones; use `.slice()` / views; `Vec::with_capacity()` for pre-allocation; prefer batch tensor ops over element-wise loops.
- **Tests**: `fn test_name() -> Result<()>` with `Ok(())` at end; arrange/act/assert structure; test names follow `test_<function>_<scenario>`.
- **Documentation**: doc comments on all public items; include `# Examples`, `# Errors`, `# Performance` sections where relevant.
- **Model architecture**: never modify `src/model.rs` — treat it as a read-only dependency.
