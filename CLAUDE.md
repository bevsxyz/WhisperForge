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

# Stream realtime transcription from microphone
cargo run --release -p whisperforge -- stream --model tiny_en_converted

# Stream from file (synthetic mic, offline speed, JSON output).
# --from-file requires a 16 kHz mono WAV (FakeMic does not resample).
cargo run --release -p whisperforge -- stream --model tiny_en_converted \
  --from-file test_data/LJ001-0001_16k.wav \
  --no-realtime --json

# Profile streaming per-window latency (p50/p99 for encoder/decode/total) → one JSON line.
# Heavy config (28 s window, 128 tokens); see Phase F "Live-mic perf ceiling".
cargo run --release -p whisperforge --bin stream_bench -- \
  --audio test_data/LJ001-0001_16k.wav --device cpu

# Build with native CUDA (requires CUDA toolkit + nvcc on host)
cargo build --release -p whisperforge --features cuda
wforge transcribe -a audio.wav -m tiny_en_converted --device cuda
```

`wforge transcribe` and `wforge list-models` honor `WF_MODELS_DIR` (default `./models/`). `--models-dir <PATH>` on either subcommand overrides the env var.

`wforge transcribe --device <auto|cpu|wgpu|cuda>` picks the backend at runtime. `auto` prefers CUDA (when built with `--features cuda`), then WGPU (when built with the default `gpu` feature), then CPU.

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

Model files (`.mpk`, `.cfg`, tokenizer) are git-ignored. Download from HuggingFace and convert with `wforge convert`.

## Architecture

Four-crate Rust workspace using [Burn 0.21](https://burn.dev/) for GPU-accelerated ML inference.

| Crate | Role |
|-------|------|
| `whisperforge-core` | Whisper model + decoding — primary development area |
| `whisperforge` | `wforge` binary; hosts `wforge transcribe` and `wforge convert` subcommands (post-Phase E merger) |
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

### Crate alignment & future refactoring

**Current state (aligned):**
- `whisperforge-core`: streaming algorithms (Chunker, Committer, Endpointer) in lib; batch decode (`decode_window`) in lib
- `whisperforge`: CLI orchestration (device dispatch, sink management, stream main loop) in binary ✓
- `whisperforge-align`: `BatchedTranscriber` available as library interface with public methods (`transcribe_single()`, `transcribe_batch_with_tokens()`)

**Known gap:** Binary's `transcribe` command uses inline `transcribe_chunk()` (lines 97–185) instead of calling `BatchedTranscriber`. This is intentional — the paths operate at different levels (binary works post-encoder for batching efficiency; library API starts from raw audio). See [whisperforge-align/src/batching.rs](whisperforge-align/src/batching.rs) **module documentation** for the full integration plan if you want to consolidate later (includes VAD path integration steps and streaming path tradeoffs).

## Hard-won lessons

- **Power spectrum**: `|STFT|²` (`norm_sqr()`) required. `norm()` gives √-scaled spectrograms → near-silence or EOT domination.
- **Slaney mel scale**: linear below 1 kHz, log above. Apply `enorm = 2/(upper_hz - lower_hz)` per filter. HTK scale degrades WER measurably.
- **center=True STFT**: reflect-pad `n_fft/2 = 200` samples each side; drop last output frame. Without this, edge frames are zero-padded.
- **EOT suppression at step 0**: if greedy argmax is EOT at step 0, take next-best non-EOT; mask EOT to −∞ in `all_logits[0]` for all fallback temperatures.
- **Convert tensor names**: `decoder.layers.X.encoder_attn.*` → `blocks.X.cross_attn.*`. Wrong mapping causes EOT domination.
- **Symphonia over hound**: handles format dispatch automatically; hound silently clips integer WAV.
- **Stale artifacts**: run `cargo clean` before concluding a crate is broken after a toolchain change.
- **Burn 0.21**: `.squeeze()` takes no arguments; use `NamedMpkFileRecorder::<FullPrecisionSettings>::new()`; `PaddingConfig1d::Explicit` takes two args: `Explicit(left, right)` — symmetric padding is `Explicit(1, 1)` not `Explicit(1)`.
- **`windows` crate version conflict — verify the culprit with `cargo tree -i windows --target all`, don't guess**: a duplicate `windows` crate in `Cargo.lock` (e.g. `0.61.3` *and* `0.62.2`) used to break Windows compiles when the `gpu` feature was on, and was assumed to be a wgpu-hal vs gpu-allocator conflict. It wasn't — the older `windows` major was actually pulled by `hf-hub 0.3.2`'s `ureq`/`reqwest` stack; bumping `hf-hub` to 0.5 collapsed the lockfile to a single `windows` major and unblocked the Windows GPU build. Lesson: when two crates appear to fight over a transitive, run `cargo tree -i <crate> --target all` to find the *actual* reverse-deps before designing a workaround — `grep`-ing the lockfile only tells you which versions are present, not who's pulling them.
- **`burn-cuda` type alias index width is `u8`, not `u32`**: `burn_cuda::Cuda<F, I>` resolves to `CubeBackend<CudaRuntime, F, I, u8>`. Mirroring the `burn-wgpu` pattern verbatim (`<…, u32>`) compiles but constructs the wrong type. Always use the public `Cuda<f32, i32>` alias rather than spelling the `CubeBackend` generics yourself.
- **NamedMpk record output is non-deterministic byte-wise**: the `.cfg` sidecar is byte-identical across runs, but `.mpk` files have the same size and content semantically while differing at the byte level (MessagePack iteration order). Diff `.cfg` and compare `.mpk` sizes — don't require byte-identical `.mpk` for "same conversion" checks.
- **Language/task tokens are wired, not hardcoded**: the decoder init sequence is `[<|sot|>, <|lang|>, <|task|>, <|notimestamps|>]`, built from `DecodingConfig.{language,task}` (transcribe) / `DecodeContext.{language_token,task_token}` (stream). `--language auto` runs `whisperforge_core::language::detect_language` (one `<|sot|>` forward pass, argmax restricted to language-token ids). **`--task translate` is X→English only** — Whisper has no other-target translation head. The "translate into X" trick (force a target `--language` that differs from the spoken audio, keep `--task transcribe`) is an *emergent, unsupported* coercion: best-effort on `large`/`large-v1`, mostly broken on `large-v2`+, drifts back to the spoken language on long audio (openai/whisper#649). Don't treat it as reliable MT. Helpers live in [language.rs](whisperforge-core/src/language.rs).

## Roadmap

Phases 1–7 + A–B + B.5 + C + E + F complete. Phase D (WASM) deferred.

### Phase B.5 — CubeCL Mel Pipeline ✅ COMPLETE

- GPU mel filterbank matmul on `CubeBackend<WgpuRuntime,f32,i32,u32>` (bare, no Fusion)
- `compute_mel_spectrogram_wgpu` + `batch_mel_spectrograms_wgpu` in `audio.rs` (feature `cubecl-stft`)
- CLI wired via `mel_fn` closure; `--features cubecl-stft` enables the GPU STFT path
- GPU vs CPU DFT correctness test in `stft_gpu.rs`
- stft_gpu.rs updated for cubecl 0.10 API (`ArrayArg::from_raw_parts` no stride/type-arg, scalars passed directly, launch returns `()`, `read_one_unchecked`)

### Phase C — Quantization ✅ COMPLETE

INT8 post-training quantization (~4× size reduction: 150 MB → 37 MB). Lives in `whisperforge::commands::convert` post-Phase E merger.
- `--quantize int8` flag on `wforge convert` (uses `Module::quantize_weights`)
- `Precision` enum (Fp32/Int8) in convert pipeline; metadata recorded in `.cfg` sidecar
- Load path unchanged — recorder transparently handles quantized DType
- **Known limitation — INT8 is disk-format-only on every backend in Burn 0.21**: the `load.rs::load_whisper_from_bytes` fallback (load on `Flex<f32>` CPU, run `Dequantizer` mapper, re-serialize to FP32, then load on the target) is **unconditional** for any `.cfg` reporting `precision: int8`. Reasoning per backend: NdArray CPU has a quantized-*conversion* unwrap panic; WGPU/WGSL has no i8 element type; **CUDA looks viable on paper — `supports_dtype(QFloat) == true` and `q_matmul` is wired — but `burn-cubecl 0.21` leaves `q_slice`, `q_gather`, `q_select`, and `q_expand` as `unimplemented!()`, so Whisper's first slice (encoder windowing / KV cache) panics mid-forward**. We tried the obvious gate (`b50d0b8`, reverted in `76db690`) — `supports_dtype` is *not* a reliable proxy for "can run quantized inference end-to-end". Until Burn lands the missing q-ops, INT8 stays a 4× on-disk saving and inference runs FP32; the 38 MB file expands to ~150 MB in RAM. Don't re-enable the gate without first checking `~/.cargo/registry/src/.../burn-cubecl-*/src/ops/qtensor.rs` for those `unimplemented!()` lines.

### Phase D — WASM Target ⏸ DEFERRED

Rolled back after C++ transitive deps blocked the JS-tokenizer path. Re-evaluation gated on a viable pure-Rust tokenizer story or a different IO boundary. See [memory/phase_d_wasm_blocker.md](../memory/phase_d_wasm_blocker.md).

### Phase E — Foundation: crate merger + CLI UX + device selection ✅ COMPLETE (with carve-outs)

Goal was to clean the crate / CLI surface before streaming work. Targets 0.4.0 (breaking).

- ✅ `whisperforge-cli` → `whisperforge` (single `wforge` binary, `autobins = false`)
- ✅ `whisperforge-convert` folded into `wforge convert` (workspace shrunk 5 → 4 crates)
- ✅ `wforge list-models` + `WF_MODELS_DIR` / `--models-dir` honored by `transcribe` and `list-models`
- ✅ Friendly model-not-found error pointing at `list-models` / `convert`
- ✅ `--task translate` removed (was parsed-then-runtime-errored)
- ✅ `--cpu` removed; replaced with `--device <auto|cpu|wgpu|cuda>` defaulting to `auto`
- ✅ Native CUDA via optional `burn-cuda` (feature `cuda`); `auto` preference is cuda → wgpu → cpu
- ⏸ Commit 6 (VRAM-aware encoder-batch auto-tune) — **deferred**. Heuristics had a poor cost/risk ratio (extra deps for sysinfo + wgpu adapter probe + WSL/integrated-GPU reporting quirks vs. a modest UX win nobody had asked for). Users override with `--encoder-batch-size` for now.
- ⏸ Commit 7 (Windows wgpu runtime fallback) — **deferred**. Upstream `windows`-crate version conflict between wgpu-hal and gpu-allocator is *compile-time*; no runtime probe can rescue it. Windows still ships CPU-only via release.yml `--no-default-features`. Revisit when `grep -A1 'name = "windows"' Cargo.lock` shows one version.

### Phase F — Streaming Realtime ✅ COMPLETE (UAT-certified)

**UAT (2026-05-31):** `scripts/uat-stream.ps1` passed on a native Windows box across **all three backends (cpu / cuda / wgpu)** — accuracy 14/14 LJ keywords at `--max-window-secs 28`, and real-time keep-up with **zero dropped samples at stride 1 s** on every backend, including WGPU. All three produced byte-identical transcripts, confirming decode parity. (The WSL2-measured ~3.5 s/window WGPU figure in the "Live-mic perf ceiling" note below is a software-rasterizer artifact, not inherent to WGPU on a real GPU.)

End-to-end realtime ASR: `cpal` mic input → Silero VAD gating (32 ms frames, ONNX) → growing-buffer chunker (5 s cap / 1 s stride; forces EOU at cap to bound per-window decode) → per-window encode + greedy decode → LocalAgreement-2 committer (longest stable prefix across successive decodes) → silence + punctuation endpointer → `<|prevtext|>` prompt context carried across utterances → output sinks: crossterm terminal UI (committed / tentative line), NDJSON (`--json`), WAV recorder (`--record-to`), transcript file (`--transcript-to`). Synthetic-mic (`--from-file --no-realtime`) enables offline testing and CI without a physical device.

- `cpal` mic input → `rubato`-resampled 16 kHz mono ring buffer
- Silero VAD gates the sliding window; only speech frames drive encoder forwards
- LocalAgreement-2 committer: tokens stable across two consecutive windows are permanently committed; remainder is tentative until confirmed
- Silence + punctuation endpointer fires EOU at configurable thresholds (`--silence-secs`, `--punct-silence-secs`)
- `<|prevtext|>` prompt prefix is opt-in (`--prompt-tokens 60`) — disabled by default since carrying prevtext on every window biases the decoder into hallucination loops on quiet audio (the classic `*sigh* *sigh* *sigh*` / `I'm going to do something.` failure mode). Matches faster-whisper's `condition_on_previous_text=False` default for streaming.
- **Confidence gate (faster-whisper parity)** — each decoded window is rejected before the committer sees it when `avg_logprob < --logprob-threshold` (default `-1.0`) **or** gzip `compression_ratio > --compression-ratio-threshold` (default `2.4`). LocalAgreement-2 only rejects *unstable* output, so a *confident* repetition loop would otherwise commit; the compression-ratio check is what actually kills the `*sigh* *sigh*` failure mode. Lives in `passes_quality_gate` ([stream_decode.rs](whisperforge-core/src/stream_decode.rs)); reuses `decoding::compression_ratio`. A rejected window degrades to an empty ingest (a normal per-frame occurrence on silence). Per-window `avg_logprob`/`min_logprob` are observable via the `decode_metrics` NDJSON event (`--json`) or `RUST_LOG=whisperforge=debug`.
- `--from-file <wav> [--no-realtime]` swaps mic for a WAV feeder — use this on WSL2 until PulseAudio is bridged
- **Cross-backend UAT:** `scripts/uat-stream.ps1` runs the binary in offline-JSON mode on each `--device` (cpu/cuda/wgpu), checks LJ keyword coverage and real-time keep-up (zero dropped samples), and prints a PASS/FAIL matrix. Build with `cargo build --release -p whisperforge --features cuda` first. Run on a native Windows box for CUDA/WGPU.

**WSL2 note:** `cpal` on WSL2 requires a PulseAudio bridge. Use `--from-file` for all testing on WSL2.

**Live-mic perf ceiling — tiny.en + CPU is decode-bound, not encoder-bound.** Autoregressive decode is O(tokens × ~100 ms) on Burn's CPU backend, so per-window cost is dominated by however many tokens the model emits. **WGPU under WSL2** (software rasterizer / no real GPU) is *worse* than CPU here — the per-token dispatch tax pushes the autoregressive loop to ~3.5 s per window. **But on a real discrete GPU (native Windows/Linux) WGPU sustains real-time at stride 1 s** — verified by the 3-backend UAT above; the ~3.5 s figure is a WSL2 artifact, not inherent to WGPU. **Defaults are still tuned for CPU** (the lowest-common-denominator backend): `--device cpu` (via `auto` fallback), `stride_secs=1.0`, `max_window_secs=5.0`, hardcoded `max_new_tokens=32`. The 5 s cap bounds the decoder at ~25 tokens (≈ 2.5 s wall-clock) on real speech, comfortably under the 1 s stride. Drops are surfaced on stderr (`[audio] dropped … samples last second …`) so silent loss is impossible; per-window mel/enc/dec latency is at `tracing::debug` (enable with `RUST_LOG=whisperforge=debug`).

Long-form utterances stay continuous via **stride-based buffer trimming on cap-hit** (commit B9). On each cap-hit window (`max_window_secs` reached), the main loop drops the oldest 1.5 s (VAD-frame-aligned) from the chunker buffer and calls `Committer::on_trim`, which **force-commits the tentative tail** of the most recent decode (everything past the last LCP boundary) and clears `last_candidate`. The next stride establishes a fresh LCP baseline; the stride after that resumes ordinary LCP commits. The utterance continues as one continuous commit stream with no endpoint event. A `cap_pending_handler` flag on the chunker escalates to `forced_eou` if the trim path is unavailable on two consecutive caps (e.g. no prior commits to anchor against) — utterances split into endpoint events only in that fallback case.

The trim point is intentionally NOT anchored to Whisper's timestamp tokens. The original plan was to leave `<|notimestamps|>` out of the streaming decode init and trim at the latest emitted segment-end timestamp inside the committed prefix. Empirically, greedy decode on tiny.en with timestamps on emits mostly timestamps and little content even under `ApplyTimestampRules`-style logit filtering — reliable timestamps require temperature-fallback sampling like the one-shot `HybridDecoder`. The stride-based heuristic accepts one "wasted" stride per trim in exchange for working with the existing reliable greedy + `<|notimestamps|>` path.

**Known content gaps at the trim seam.** Even with force-commit, ~30% of mid-utterance content is dropped at trim boundaries — Whisper's non-causal encoder produces different tokens for the same audio across different buffer sizes, so LCP doesn't reliably confirm tokens that span a trim. The current implementation prefers structural continuity (one endpoint event per utterance, no duplicated text) over content fidelity. **This gap is accepted, not a bug:** the happy path (dictation/conversation with natural pauses) resets cleanly per-utterance and never hits a cap, so it's unaffected. For *continuous multi-minute monologue with no pauses*, the **supported path is `--max-window-secs 28`** (no cap-hits on most utterances → no trim seams). Truly fixing the trim-seam case requires temperature-fallback sampling + working timestamps + timestamp-anchored trimming — a deferred follow-up, not on the current roadmap.

A **sliding-buffer** alternative (drop oldest `stride_secs` per stride) was prototyped much earlier and reverted because the encoder-output mismatch across buffer sizes produced duplicated text at every slide seam. The current cap-hit-only trim is dramatically less frequent (one trim per ~1.5 s of new audio vs. one slide per `stride_secs`) so the duplication cost is amortized and `Committer::on_trim`'s clearing of `last_candidate` prevents direct re-derivation of force-committed tokens.

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
| `/stream-test` | Run stream from test WAV in offline JSON mode |

### Hooks (auto-configured in `.claude/settings.json`)

- **PreToolUse on Edit/Write**: blocks edits to `model.rs`
- **PostToolUse on Edit/Write**: regenerates `CODEINDEX.md` on any `.rs` edit (ripgrep public symbols, <100 ms, nothing injected into context)

### Agent patterns

Spawn a Plan agent for phase-level planning (new phase kickoff, architectural decisions). Avoid spawning agents for per-commit or sub-phase work — each agent reloads the full context window.
