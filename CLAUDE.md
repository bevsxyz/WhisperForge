# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Check compilation
cargo check --all

# Tests — ALWAYS use --release and exclude whisperforge-align (has known pre-existing failures)
# Debug builds are unusably slow: NdArray CPU decoder forward pass is 30s+ per step unoptimized.
cargo test --release -p whisperforge-core -p whisperforge-convert -p whisperforge-cli

# Single test with output
cargo test --release -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture --exact

# Format (auto-fix) + lint
cargo fmt --all && cargo clippy --all-targets --all-features

# Git hooks — run once after cloning; hooks call mise tasks on commit/push
mise run setup

# Run the CLI
cargo run --release -p whisperforge-cli -- -a audio.wav -m tiny_en_converted

# Backtrace on failure
RUST_BACKTRACE=1 cargo test --release test_name -- --nocapture
```

Model files (`.mpk`, `.cfg`, tokenizer) are git-ignored. Download from HuggingFace and convert with `whisperforge-convert`.

## Architecture

Five-crate Rust workspace using [Burn 0.20](https://burn.dev/) for CUDA-accelerated ML inference.

| Crate | Role |
|-------|------|
| `whisperforge-core` | Whisper model + decoding — primary development area |
| `whisperforge-cli` | `whisperforge` binary |
| `whisperforge-convert` | One-shot HuggingFace safetensors → Burn NamedMpk conversion |
| `whisperforge-align` | VAD, segmentation, SRT output (has known test failures; work cautiously) |
| `whisperforge-diarize` | Placeholder — not yet integrated |

### Data flow

```
Audio → hound load (i16 → f32 via /i16::MAX) → resample to 16 kHz mono
  → zero-pad or truncate to exactly 480 000 samples (30 s at 16 kHz)
  → center=True reflect-pad n_fft/2=200 samples each end  →  480 400 samples
  → STFT: N_FFT=400, hop=160, Hann window  →  power spectrum |STFT|² (NOT |STFT|)
  → drop last frame (matches Python magnitudes[..., :-1])  →  3001 → 3000 frames
  → Slaney mel filter bank (NOT HTK): linear below 1 kHz, log above, enorm normalised
  → log10 mel + clamp to max-8 dB  →  [1, 80, 3000]
  → Encoder: Conv1d stem (stride-2 halves time to 1500) → n_audio_layer transformer blocks
  → Decoder: n_text_layer transformer blocks (masked self-attn → cross-attn → MLP)
  → HybridDecoder (quality-gated temperature fallback) → decoded text
```

Layer counts vary by model size — **"16 layers" mentioned in older docs is wrong**:

| Model | n_audio_layer | n_text_layer | n_mels |
|-------|--------------|--------------|--------|
| tiny.en | 4 | 4 | 80 |
| base | 6 | 6 | 80 |
| small | 12 | 12 | 80 |
| medium | 24 | 24 | 80 |
| large-v2/v3 | 32 | 32 | 128 |

Key files in `whisperforge-core`:
- [src/model.rs](whisperforge-core/src/model.rs) — **FROZEN**. Do not modify; architecture is locked at Burn 0.20.
- [src/load.rs](whisperforge-core/src/load.rs) — loads converted `.mpk` + JSON config via `NamedMpkFileRecorder`.
- [src/audio.rs](whisperforge-core/src/audio.rs) — mel spectrogram, audio I/O.
- [src/decoding.rs](whisperforge-core/src/decoding.rs) — `DecodingConfig`, `BeamSearchDecoder`, `GreedyDecoder`, `HybridDecoder`. Active development area.

All model types are generic over `B: Backend`. Default alias is `NdArray<f32>` (CPU); swap to `Cuda` for GPU.

## What is actually wired vs what isn't

### CLI decoding (`whisperforge-cli/src/main.rs`)

The greedy loop collects per-step logits up to `decoding_config.max_length` tokens, then passes them to `HybridDecoder.decode_with_fallback()` for quality-gated temperature fallback. EOT is suppressed at step 0 so the model always generates at least one text token. `--decoding-preset`, `--beam-size`, `--temperature`, and `--length-penalty` all take effect.

`--vad-enabled` / `--vad-threshold` are fully wired. When `--vad-enabled`, `AudioSegmenter` (which delegates to `VoiceActivityDetector::detect()`) segments audio into voice spans; silence is skipped; segments feed the transcription loop with accurate timestamps.

### Unimplemented scaffolding

- `BatchedTranscriber` in `whisperforge-align` — stub that returns placeholder text; real transcription not wired.

### Implemented scaffolding (Phase 4)

- `WhisperTranscriber<B>` in `whisperforge-core/src/transcribe.rs` — wraps `Whisper<B>` + tokenizer + config; implements `WhisperInference<B>`. Phase 5 will replace the approximate 0–30 s segment timing with cross-attention-derived per-token timestamps.
- `TranscriptionResult` / `TranscriptionSegment` — fully serializable (serde); populated by the CLI with chunk-boundary timestamps.
- `--output-format text|srt|json` in the CLI — text is plain join; srt uses `SrtWriter`; json uses `serde_json`.

### Long audio

Audio longer than 30 s is automatically chunked into ≤30 s windows with 1 s overlap and transcribed sequentially. `--vad-enabled` uses `AudioSegmenter` segments as the chunk boundaries instead of fixed windows.

## Decoding: current state

`decoding.rs` is fully implemented through Phase 2. `HybridDecoder.decode_with_fallback()` runs the complete faster-whisper SOTA strategy:

- `temperatures: Vec<f32>` — defaults to `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` ✅
- `compression_ratio_threshold: f32` (2.4) via `flate2` gzip ✅
- `log_prob_threshold: f32` (-1.0) ✅
- `no_speech_threshold: f32` (0.6) ✅
- Quality-gated temperature fallback loop ✅
- LJSpeech WER benchmark — 0.8% average WER on `tiny.en` ✅
- CLI wired to `decode_with_fallback`, EOT suppressed at step 0 ✅

## Hard-won lessons

### Mel spectrogram: use power spectrum, not magnitude

`|STFT|²` (`norm_sqr()`) is required — Whisper's log-mel formula expects power. Using `norm()` (magnitude) produces spectrograms that are √-scaled relative to reference, causing the model to output near-silence predictions or EOT domination. This single change dropped WER from ~100% to ~40%.

### Mel filter bank: Slaney scale, not HTK

Python `librosa`/Whisper uses the Slaney mel scale: linear below 1 kHz (`f / (200/3)`), logarithmic above (`min_log_mel + ln(f/1000) / logstep` where `logstep = ln(6.4)/27`). **Not** HTK (`2595 * log10(1 + f/700)`). Also apply Slaney triangle normalisation: `enorm = 2 / (upper_hz - lower_hz)` per filter — filters do **not** sum to 1.0. Using HTK gives measurably different filter shapes and degrades WER.

### center=True STFT reflection padding (matches Python default)

`torch.stft` defaults to `center=True`: reflect-pad `n_fft/2 = 200` samples on each side before the STFT, then drop the last output frame (`magnitudes[..., :-1]`). Left pad: `samples[200], samples[199], ..., samples[1]`. Right pad (for 480 000-sample input): `samples[479998], ..., samples[479799]`. Without this, the first and last audio frames are windowed against zeros rather than reflected signal, and frame count differs from the Python pipeline.

### EOT suppression at step 0 is required

When the model has high confidence in `<|endoftext|>` as its first generated token (often seen on clean speech), the greedy loop exits immediately with zero text logits. `decode_with_fallback` then passes the quality gate on the empty sequence (avg log-prob ≈ log(p_eot) > -1.0) and returns nothing. Fix: at step 0 only, if the greedy argmax is EOT, take the next-best non-EOT token instead. Also mask EOT to `−∞` in `all_logits[0]` before passing to `decode_with_fallback` so the fallback sees the same suppression at every temperature.

### Cross-attention weight names in model conversion

The EOT-domination bug (model predicts `<|endoftext|>` immediately) was caused by incorrect tensor name mapping in `whisperforge-convert/src/convert.rs`. OpenAI safetensors use `decoder.layers.X.encoder_attn.*`; the Burn model uses `blocks.X.cross_attn.*`. Verify this mapping exactly if conversion produces pathological outputs.

### Audio: do NOT use `hound::into_samples::<f32>()`

This produced zero spectrograms. The correct approach is manual `i16 → f32` conversion: divide the raw `i16` sample by `i16::MAX as f32`. See [whisperforge-core/src/audio.rs](whisperforge-core/src/audio.rs) for the pattern.

### Burn 0.20 API differences

- `.squeeze()` takes **no arguments** in 0.20 (changed from older versions).
- Always use `NamedMpkFileRecorder::<FullPrecisionSettings>::new()` for saving/loading models.
- When a function can't be found, search the Burn crate source — don't assume names from older docs.

### whisperforge-align test failures

Pre-existing; always exclude: `cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli`.

## Roadmap

Seven phases toward WhisperX feature parity in Rust. Work phases in order — each unblocks the next.

Each entry below is a single commit. Every commit must leave `cargo check --all` clean and the test suite passing before it lands.

### Phase 1 — Integration ✅ COMPLETE
`load.rs` config from disk; HybridDecoder wired into CLI; `temperatures: Vec<f32>` fallback sequence. See git log.

### Phase 2 — SOTA Decoding ✅ COMPLETE
Compression ratio, log-prob, no-speech quality gates; `decode_with_fallback()` temperature loop; mel preprocessing matched to Python Whisper (power spectrum, Slaney scale, center padding). WER 0.8% on `tiny.en`. See git log.

### Phase 3 — Multi-model + VAD ✅ COMPLETE

```
✅ feat: complete tensor name mapping in whisperforge-convert
```
- Done: `convert.rs` maps `decoder.layers.X.encoder_attn.*` → `block.cross_attn.*` and handles all encoder/decoder layers for any model size.

```
✅ test: verify base and small model loading and forward pass
```
- Done: `#[ignore]` integration tests in `load.rs` exercise `load_whisper()` for base (512-dim) and small (768-dim), guarded against missing model files. `model_shapes.rs` adds `test_large_v2_config_has_128_mels` to document CI exclusion of medium/large-v2.

```
✅ feat: wire VoiceActivityDetector into CLI pipeline
```
- Done: `whisperforge-cli` depends on `whisperforge-align`. When `--vad-enabled`, `AudioSegmenter` (wrapping `VoiceActivityDetector::detect()`) produces voice spans; silence is skipped; segments flow to the transcription loop with accurate timestamps. Functional tests added to `vad.rs` and `segmentation.rs`.

```
✅ feat: 30-second chunked transcription with overlap for long audio
```
- Done: `transcribe_chunk()` extracted; `chunk_audio_fixed()` splits at ≤30 s with 1 s overlap; VAD path uses `AudioSegmenter` segments. Parts joined with space.

After phase 3: all model sizes work; hour-long audio transcribes correctly.

### Phase 4 — Structured Output + SRT ✅ COMPLETE
`WhisperTranscriber<B>` implements `WhisperInference<B>`; `--output-format text|srt|json` all wired; chunk-boundary timestamps (approximate — Phase 5 replaces with cross-attention peaks). See git log.

### Phase 5 — Word-Level Timestamps
Two options. Ship Option A first; Option B is a separate follow-on.

**Option A — attention-based (~100ms precision, 2 commits)**

```
✅ feat: capture cross-attention weights during decoding
```
- Done: `whisperforge-core/src/attn_extract.rs` — `forward_decoder_with_cross_attn()` replicates the TextDecoder forward pass, capturing the cross-attention softmax matrix at each block for the last query position. Averaged over all layers and heads. No modification to frozen `model.rs` — accesses public fields directly.

```
✅ feat: extract per-token timestamps from cross-attention peaks
```
- Done: `transcribe_with_timestamps()` in `transcribe.rs` calls `forward_decoder_with_cross_attn` at each greedy step. Timestamp = `argmax(avg_layer_head_weights) * 2 * 160 / 16000` seconds. `TranscriptionSegment.token_timestamps` populated; segment `start`/`end` set to actual speech boundaries instead of the full 0–30 s placeholder. `token_timestamps` is serde-skipped when empty.

**Option B — forced alignment (~20ms precision, 3 additional commits)**

```
feat: add wav2vec2 ONNX model loading in whisperforge-align
```
- Use `ort` crate. Load a wav2vec2 ctc model (e.g. `wav2vec2-base-960h`). `batching.rs` is the right home for this.

```
feat: DTW forced alignment of wav2vec2 phoneme posteriors to token sequence
```
- Run wav2vec2 on the audio to get per-frame phoneme log-probs. DTW-align the token sequence against the posterior matrix. Output word-boundary frame indices.

```
perf: optimize alignment for long audio with chunked wav2vec2 inference
```
- Run wav2vec2 in 30s chunks matching the Whisper chunks. Stitch boundary indices.

After phase 5 (Option A): per-token timestamps in all output formats.

### Phase 6 — CUDA + Performance

```
feat: add burn-cuda backend behind --cuda flag
```
- Add `burn-cuda` as an optional workspace dependency. Add `--cuda` to the CLI; swap the `NdArray<f32>` type alias to `Cuda<f32>` at runtime. Verify the existing tests still pass on CPU with no flag.

```
perf: KV-cache for encoder cross-attention in decoder
```
- Cache the encoder's K/V projections after the first decode step (they are constant for the entire sequence). This is the single largest speedup for beam search — cross-attn dominates decode time.

```
perf: KV-cache for decoder self-attention
```
- Cache the growing decoder K/V; append the new K/V slice at each step rather than recomputing from scratch.

```
perf: batch spectrogram computation across 30-second chunks
```
- Stack multiple mel spectrograms into a batch tensor before the encoder. Measure throughput improvement on a multi-chunk file.

```
perf: INT8 quantization for VRAM reduction
```
- Check Burn 0.20 quantization API. If supported: quantize weights at load time, benchmark VRAM usage for `large-v2`. If not supported: open a tracking issue and plan a Burn version bump.

After phase 6: `large-v2` faster than realtime on CUDA; VRAM ≤2GB with INT8.

### Phase 7 — Speaker Diarization

```
feat: integrate pyannote-rs speaker embeddings in whisperforge-diarize
```
- Replace the empty stub. Load the pyannote ONNX speaker embedding model via `ort`. Extract a speaker embedding vector per audio segment.

```
feat: speaker clustering and segment label assignment
```
- Cosine similarity + agglomerative clustering over segment embeddings. Map speaker spans to `TranscriptionSegment` entries by time overlap.

```
feat: add --diarize flag with speaker labels in SRT and JSON output
```
- `[SPEAKER_00]: text` in SRT; `"speaker": "SPEAKER_00"` in JSON. End-to-end test on a two-speaker recording.

After phase 7: full WhisperX feature parity. `--diarize` produces speaker-labelled subtitles.

---

**Total: 26 commits across 7 phases.** Phases 1–4 (10 commits) are the useful-product milestone. Phase 6 risk: Burn 0.20 INT8 maturity is uncertain — benchmark early in that phase and decide whether a Burn version bump is warranted before the quantization commit.

## Code conventions

- **Error handling**: `anyhow::Result<T>` everywhere; `.with_context(|| format!("…"))` on every `?`. No `.unwrap()` outside `main()` or tests.
- **Imports**: std → external crates (alphabetical) → local `crate::` modules. No glob imports outside `#[cfg(test)]`.
- **Hot paths**: no tensor clones; use `.slice()` / views; `Vec::with_capacity()` for pre-allocation; prefer batch tensor ops over element-wise loops.
- **Tests**: `fn test_name() -> Result<()>` with `Ok(())` at end; arrange/act/assert structure; test names follow `test_<function>_<scenario>`.
- **Documentation**: doc comments on all public items with `# Examples`, `# Errors`, `# Performance` sections where relevant.
- **Model architecture**: never modify `src/model.rs` — treat it as a read-only dependency.

## Working with Claude

### Slash commands

| Command | Purpose |
|---------|---------|
| `/test` | Run the correct test suite (release, excluding whisperforge-align) |
| `/bench` | Run LJSpeech WER benchmark with nocapture output |
| `/check` | Full quality gate: fmt check → clippy → compile |
| `/phase` | Summarize roadmap status and list next commits |

### Agent patterns

- **Before starting a phase**: spawn a `Plan` agent on the phase's CLAUDE.md section — produces ordered commit sequence with file-level targets before touching code.
- **After multi-file changes**: spawn an `Explore` agent to run `cargo check --all 2>&1` in isolation and report errors grouped by crate — keeps compiler noise out of main context.
- **When WER regresses**: spawn a general-purpose agent to diff the mel/decoding pipeline against the last known-good commit, cross-referencing the Hard-won lessons section.

### Hooks (auto-configured in `.claude/settings.json`)

- **PreToolUse on Edit/Write**: blocks edits to `model.rs`
- **PostToolUse on Edit/Write**: on any `.rs` edit, auto-runs `cargo fmt --all` then surfaces `cargo check` errors inline
