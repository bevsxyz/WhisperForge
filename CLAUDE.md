# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
# Check compilation
cargo check --all

# Tests — ALWAYS use --release and exclude whisperforge-align (has known pre-existing failures)
cargo test --release -p whisperforge-core -p whisperforge

# Single test with output
cargo test --release -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture --exact

# Format (auto-fix) + lint
cargo fmt --all && cargo clippy --all-targets --all-features

# Git hooks — run once after cloning
mise run setup

# Release (bumps version, generates CHANGELOG via git-cliff, tags, pushes)
cargo release patch   # or minor / major

# Run the CLI (transcribe is a subcommand post-Phase E merger)
cargo run --release -p whisperforge -- transcribe -a audio.wav -m tiny_en_converted

# Convert a HuggingFace Whisper model to Burn format
cargo run --release -p whisperforge -- convert --model-id openai/whisper-tiny.en --output models/tiny_en_converted

# List converted models under ./models (override with --models-dir or WF_MODELS_DIR)
cargo run --release -p whisperforge -- list-models

# Build with native CUDA (requires CUDA toolkit + nvcc on host)
cargo build --release -p whisperforge --features cuda
wf transcribe -a audio.wav -m tiny_en_converted --device cuda
```

`wf transcribe` and `wf list-models` honor `WF_MODELS_DIR` (default `./models/`). `--models-dir <PATH>` on either subcommand overrides the env var.

`wf transcribe --device <auto|cpu|wgpu|cuda>` picks the backend at runtime. `auto` prefers CUDA (when built with `--features cuda`), then WGPU (when built with the default `gpu` feature), then CPU.

## Commit Message Convention

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/) format:
```
type(scope)?: description

[optional body]
```

Types: `feat` (feature), `fix` (bug fix), `docs` (docs), `refactor` (code change), `perf` (performance), `style`, `test`, `chore`, `ci`, `build`, `revert`.

Examples:
```
feat: add streaming inference
fix(decoding): handle EOT suppression at step 0
docs: explain mel-scale setup
chore: release v0.4.0
ci: disable wgpu on windows
```

The `commit-msg` hook validates this; non-conforming commits are blocked. `git-cliff` auto-generates CHANGELOG from commits using this format.

Model files (`.mpk`, `.cfg`, tokenizer) are git-ignored. Download from HuggingFace and convert with `wf convert`.

## Architecture

Four-crate Rust workspace using [Burn 0.21](https://burn.dev/) for GPU-accelerated ML inference.

| Crate | Role |
|-------|------|
| `whisperforge-core` | Whisper model + decoding — primary development area |
| `whisperforge` | `wf` binary; hosts `wf transcribe` and `wf convert` subcommands (post-Phase E merger) |
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
- **Windows wgpu incompatibility (compile-time)**: `wgpu-hal 29.0.3` pulls `windows = 0.61.3`; `gpu-allocator 0.28.0` pulls `windows = 0.62.2`. Cargo can't resolve both on Windows when the `gpu` feature is on, so the build fails to *compile*, not at runtime — no runtime fallback (`catch_unwind`, adapter probe, etc.) can paper over it. Workaround: the Windows release CI job uses `--no-default-features`, shipping a CPU-only Windows binary. To check whether this is still needed, run `grep -A1 'name = "windows"' Cargo.lock` — if both versions still appear under `wgpu-hal` / `gpu-allocator` dep trees, the carve-out stays. Drop it once upstream converges on a single `windows` major.
- **`burn-cuda` type alias index width is `u8`, not `u32`**: `burn_cuda::Cuda<F, I>` resolves to `CubeBackend<CudaRuntime, F, I, u8>`. Mirroring the `burn-wgpu` pattern verbatim (`<…, u32>`) compiles but constructs the wrong type. Always use the public `Cuda<f32, i32>` alias rather than spelling the `CubeBackend` generics yourself.
- **NamedMpk record output is non-deterministic byte-wise**: the `.cfg` sidecar is byte-identical across runs, but `.mpk` files have the same size and content semantically while differing at the byte level (MessagePack iteration order). Diff `.cfg` and compare `.mpk` sizes — don't require byte-identical `.mpk` for "same conversion" checks.

## Roadmap

Phases 1–7 + A–B + B.5 + C + E complete. Phase D (WASM) deferred. **Next: Phase F = streaming realtime.**

### Phase B.5 — CubeCL Mel Pipeline ✅ COMPLETE

- GPU mel filterbank matmul on `CubeBackend<WgpuRuntime,f32,i32,u32>` (bare, no Fusion)
- `compute_mel_spectrogram_wgpu` + `batch_mel_spectrograms_wgpu` in `audio.rs` (feature `cubecl-stft`)
- CLI wired via `mel_fn` closure; `--features cubecl-stft` enables the GPU STFT path
- GPU vs CPU DFT correctness test in `stft_gpu.rs`
- stft_gpu.rs updated for cubecl 0.10 API (`ArrayArg::from_raw_parts` no stride/type-arg, scalars passed directly, launch returns `()`, `read_one_unchecked`)

### Phase C — Quantization ✅ COMPLETE

INT8 post-training quantization (~4× size reduction: 150 MB → 37 MB). Lives in `whisperforge::commands::convert` post-Phase E merger.
- `--quantize int8` flag on `wf convert` (uses `Module::quantize_weights`)
- `Precision` enum (Fp32/Int8) in convert pipeline; metadata recorded in `.cfg` sidecar
- Load path unchanged — recorder transparently handles quantized DType
- **Known limitation**: NdArray CPU backend has Burn 0.21 quantization bug (unwrap panic during quantized *conversion*). WGPU/WGSL has no INT8 element type, so Burn cannot place QFloat tensors on a WGPU device. Fix (in `load.rs`): INT8 models are transparently loaded on `Flex<f32>` CPU first, dequantized via `Dequantizer` mapper, re-serialized to FP32 bytes, then loaded on the target backend. Only triggers when `.cfg` reports `precision: int8`. The 38 MB file expands to ~150 MB in RAM; disk size advantage is preserved.

### Phase D — WASM Target ⏸ DEFERRED

Rolled back after C++ transitive deps blocked the JS-tokenizer path. Re-evaluation gated on a viable pure-Rust tokenizer story or a different IO boundary. See [memory/phase_d_wasm_blocker.md](../memory/phase_d_wasm_blocker.md).

### Phase E — Foundation: crate merger + CLI UX + device selection ✅ COMPLETE (with carve-outs)

Goal was to clean the crate / CLI surface before streaming work. Targets 0.4.0 (breaking).

- ✅ `whisperforge-cli` → `whisperforge` (single `wf` binary, `autobins = false`)
- ✅ `whisperforge-convert` folded into `wf convert` (workspace shrunk 5 → 4 crates)
- ✅ `wf list-models` + `WF_MODELS_DIR` / `--models-dir` honored by `transcribe` and `list-models`
- ✅ Friendly model-not-found error pointing at `list-models` / `convert`
- ✅ `--task translate` removed (was parsed-then-runtime-errored)
- ✅ `--cpu` removed; replaced with `--device <auto|cpu|wgpu|cuda>` defaulting to `auto`
- ✅ Native CUDA via optional `burn-cuda` (feature `cuda`); `auto` preference is cuda → wgpu → cpu
- ⏸ Commit 6 (VRAM-aware encoder-batch auto-tune) — **deferred**. Heuristics had a poor cost/risk ratio (extra deps for sysinfo + wgpu adapter probe + WSL/integrated-GPU reporting quirks vs. a modest UX win nobody had asked for). Users override with `--encoder-batch-size` for now.
- ⏸ Commit 7 (Windows wgpu runtime fallback) — **deferred**. Upstream `windows`-crate version conflict between wgpu-hal and gpu-allocator is *compile-time*; no runtime probe can rescue it. Windows still ships CPU-only via release.yml `--no-default-features`. Revisit when `grep -A1 'name = "windows"' Cargo.lock` shows one version.

### Phase F — Streaming Realtime ⬜ NEXT

Token-level output, mic input via `cpal`, ring-buffer chunking, KV-cache across windows. Headline goal: end-to-end realtime ASR. Subsequent phases (EOU detection, denoising, Moonshine) become tractable once the streaming framework lands.

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
| `/release` | Pre-release gate + guided `cargo release` (dry-run → confirm → tag/push) |

### Hooks (auto-configured in `.claude/settings.json`)

- **PreToolUse on Edit/Write**: blocks edits to `model.rs`
- **PostToolUse on Edit/Write**: regenerates `CODEINDEX.md` on any `.rs` edit (ripgrep public symbols, <100 ms, nothing injected into context)

### Agent patterns

Spawn a Plan agent for phase-level planning (new phase kickoff, architectural decisions). Avoid spawning agents for per-commit or sub-phase work — each agent reloads the full context window.
