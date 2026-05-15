# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
# Check compilation
cargo check --all

# Tests — ALWAYS use --release and exclude whisperforge-align (has known pre-existing failures)
cargo test --release -p whisperforge-core -p whisperforge-convert -p whisperforge-cli

# Single test with output
cargo test --release -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture --exact

# Format (auto-fix) + lint
cargo fmt --all && cargo clippy --all-targets --all-features

# Git hooks — run once after cloning
mise run setup

# Run the CLI
cargo run --release -p whisperforge-cli -- -a audio.wav -m tiny_en_converted
```

Model files (`.mpk`, `.cfg`, tokenizer) are git-ignored. Download from HuggingFace and convert with `whisperforge-convert`.

## Architecture

Five-crate Rust workspace using [Burn 0.21](https://burn.dev/) for GPU-accelerated ML inference.

| Crate | Role |
|-------|------|
| `whisperforge-core` | Whisper model + decoding — primary development area |
| `whisperforge-cli` | `whisperforge` binary |
| `whisperforge-convert` | One-shot HuggingFace safetensors → Burn NamedMpk conversion |
| `whisperforge-align` | VAD, segmentation, SRT output (has known test failures; work cautiously) |
| `whisperforge-diarize` | Speaker diarization (Option A shipped; Option B deferred) |

### Data flow

```
Audio → symphonia probe+decode (WAV/MP3/FLAC/OGG/M4A) → resample to 16 kHz mono
  → zero-pad/truncate to 480 000 samples (30 s)
  → center=True reflect-pad n_fft/2=200 samples each end
  → STFT: N_FFT=400, hop=160, Hann → power spectrum |STFT|² → drop last frame → 3000 frames
  → Slaney mel filterbank (enorm normalised) → log10 mel clamp → [1, 80, 3000]
  → Encoder: Conv1d stride-2 → n_audio_layer transformer blocks
  → Decoder: n_text_layer transformer blocks → HybridDecoder temperature fallback
```

| Model | n_audio_layer | n_text_layer | n_mels |
|-------|--------------|--------------|--------|
| tiny.en | 4 | 4 | 80 |
| base | 6 | 6 | 80 |
| small | 12 | 12 | 80 |
| medium | 24 | 24 | 80 |
| large-v2/v3 | 32 | 32 | 128 |

Key files in `whisperforge-core/src/`:
- [model.rs](whisperforge-core/src/model.rs) — **FROZEN**. Never modify; architecture locked at Burn 0.21.
- [audio.rs](whisperforge-core/src/audio.rs) — mel spectrogram, audio I/O, `batch_mel_spectrograms`.
- [decoding.rs](whisperforge-core/src/decoding.rs) — `HybridDecoder`, quality-gated temperature fallback.
- [kv_cache.rs](whisperforge-core/src/kv_cache.rs) — O(n) decoder via cached K,V.
- [stft_gpu.rs](whisperforge-core/src/stft_gpu.rs) — CubeCL DFT kernel (feature `cubecl-stft`).
- `CODEINDEX.md` — auto-generated public API surface (gitignored); check here before reading a source file.

## Hard-won lessons

- **Power spectrum**: `|STFT|²` (`norm_sqr()`) required. `norm()` gives √-scaled spectrograms → near-silence or EOT domination.
- **Slaney mel scale**: linear below 1 kHz, log above. Apply `enorm = 2/(upper_hz - lower_hz)` per filter. HTK scale degrades WER measurably.
- **center=True STFT**: reflect-pad `n_fft/2 = 200` samples each side; drop last output frame. Without this, edge frames are zero-padded.
- **EOT suppression at step 0**: if greedy argmax is EOT at step 0, take next-best non-EOT; mask EOT to −∞ in `all_logits[0]` for all fallback temperatures.
- **Convert tensor names**: `decoder.layers.X.encoder_attn.*` → `blocks.X.cross_attn.*`. Wrong mapping causes EOT domination.
- **Symphonia over hound**: handles format dispatch automatically; hound silently clips integer WAV.
- **Stale artifacts**: run `cargo clean` before concluding a crate is broken after a toolchain change.
- **Burn 0.21**: `.squeeze()` takes no arguments; use `NamedMpkFileRecorder::<FullPrecisionSettings>::new()`; `PaddingConfig1d::Explicit` takes two args: `Explicit(left, right)` — symmetric padding is `Explicit(1, 1)` not `Explicit(1)`.
- **Windows wgpu incompatibility**: wgpu-hal 29.0.3 depends on `windows` 0.61.3 but gpu-allocator 0.28.0 pulls in 0.62.2, causing version conflict on Windows. Fix: use target-specific dependencies in whisperforge-cli to disable wgpu feature on Windows (CPU fallback). TODO: Remove when Burn/wgpu update resolves this.

## Roadmap

Phases 1–7 + A–B + B.5 complete. **Current: Phase C (Quantization), then D.**

### Phase B.5 — CubeCL Mel Pipeline ✅ COMPLETE

- GPU mel filterbank matmul on `CubeBackend<WgpuRuntime,f32,i32,u32>` (bare, no Fusion)
- `compute_mel_spectrogram_wgpu` + `batch_mel_spectrograms_wgpu` in `audio.rs` (feature `cubecl-stft`)
- CLI wired via `mel_fn` closure; `--features cubecl-stft` enables the GPU STFT path
- GPU vs CPU DFT correctness test in `stft_gpu.rs`
- stft_gpu.rs updated for cubecl 0.10 API (`ArrayArg::from_raw_parts` no stride/type-arg, scalars passed directly, launch returns `()`, `read_one_unchecked`)

### Phase C — Quantization ✅ COMPLETE

INT8 post-training quantization (~4× size reduction: 150 MB → 37 MB).
- `--quantize int8` flag in `whisperforge-convert` (uses `Module::quantize_weights`)
- `Precision` enum (Fp32/Int8) in convert pipeline; metadata recorded in `.cfg` sidecar
- Load path unchanged — recorder transparently handles quantized DType
- **Known limitation**: NdArray CPU backend has Burn 0.21 quantization bug (unwrap panic during quantized *conversion*). WGPU/WGSL has no INT8 element type, so Burn cannot place QFloat tensors on a WGPU device. Fix (in `load.rs`): INT8 models are transparently loaded on `Flex<f32>` CPU first, dequantized via `Dequantizer` mapper, re-serialized to FP32 bytes, then loaded on the target backend. Only triggers when `.cfg` reports `precision: int8`. The 38 MB file expands to ~150 MB in RAM; disk size advantage is preserved.

### Phase D — WASM Target ⬜ PLANNED

New crate `whisperforge-wasm`; `wasm-bindgen` wrapper; `async fn transcribe(audio_samples, sample_rate, model_bytes, config_bytes) -> String`; browser example with `getUserMedia`. Requires Phase B streaming + Phase C quantization.

## Code conventions

- **Error handling**: `anyhow::Result<T>`; `.with_context(|| format!("…"))` on every `?`. No `.unwrap()` outside `main()` or tests.
- **Imports**: std → external crates (alphabetical) → local `crate::`. No glob imports outside `#[cfg(test)]`.
- **Hot paths**: no tensor clones; use `.slice()` / views; `Vec::with_capacity()`; batch tensor ops over element-wise loops.
- **Tests**: `fn test_name() -> Result<()>` with `Ok(())` at end; arrange/act/assert; names follow `test_<function>_<scenario>`.
- **model.rs**: treat as read-only — never modify.

## Working with Claude

### Slash commands

| Command | Purpose |
|---------|---------|
| `/test` | Run the correct test suite (release, excluding whisperforge-align) |
| `/bench` | Run LJSpeech WER benchmark |
| `/check` | Full quality gate: fmt check → clippy → compile |
| `/phase` | Current phase status and next commits |

### Hooks (auto-configured in `.claude/settings.json`)

- **PreToolUse on Edit/Write**: blocks edits to `model.rs`
- **PostToolUse on Edit/Write**: regenerates `CODEINDEX.md` on any `.rs` edit (ripgrep public symbols, <100 ms, nothing injected into context)

### Agent patterns

Spawn a Plan agent for phase-level planning (new phase kickoff, architectural decisions). Avoid spawning agents for per-commit or sub-phase work — each agent reloads the full context window.
