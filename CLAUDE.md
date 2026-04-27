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
| `whisperforge-core` | Whisper model + decoding — primary development area |
| `whisperforge-cli` | `whisperforge` binary |
| `whisperforge-convert` | One-shot HuggingFace safetensors → Burn NamedMpk conversion |
| `whisperforge-align` | VAD, segmentation, SRT output (has known test failures; work cautiously) |
| `whisperforge-diarize` | Placeholder — not yet integrated |

### Data flow

```
Audio → FFmpeg/hound load → resample to 16 kHz mono
  → Mel spectrogram (STFT: N_FFT=400, hop=160, n_mels=80)  →  [1, 80, 3000]
  → Encoder: Conv1d stem (stride-2 halves time to 1500) → n_audio_layer transformer blocks
  → Decoder: n_text_layer transformer blocks (masked self-attn → cross-attn → MLP)
  → HybridDecoder (beam search + temperature fallback) → decoded text

[Current reality: the CLI still runs a hand-rolled greedy loop capped at 50 tokens.
 Phase 1 of the roadmap replaces this with the actual HybridDecoder.]
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

`HybridDecoder` is instantiated but assigned to `_decoder` and **never called**. The actual token generation (lines 213–296) is a hand-rolled greedy loop capped at **50 tokens**, regardless of `DecodingConfig.max_length = 448`. The `--decoding-preset`, `--beam-size`, and `--temperature` flags build a `DecodingConfig` but it has no effect on output today.

`--vad-enabled` / `--vad-threshold` exist as CLI args and print a message, but **no VAD runs**.

### Unimplemented scaffolding

- `WhisperInference<B>` trait in `lib.rs` — defined, never implemented for `Whisper<B>`.
- `TranscriptionSegment` / `TranscriptionResult` in `lib.rs` — defined, never populated by the CLI (it returns plain `String`).

### Known bug in `load_whisper` (`whisperforge-core/src/load.rs:121–129`)

After loading weights, the function replaces the decoder's final `ln` and encoder's `ln_post` with freshly-initialised defaults sourced from `WhisperConfig::tiny_en()`. This is a workaround for corrupted layer-norm parameters in the converted model. It is **hardcoded to tiny_en dimensions** — loading `base`, `medium`, or `large-v2` will silently inject wrong layer-norm parameters. Fix this before supporting multi-model loading.

## Decoding: current state vs plan

`decoding.rs` has `BeamSearchDecoder` and `HybridDecoder` (falls back to greedy on beam failure), but the full faster-whisper SOTA strategy is not yet wired in. Missing:

- Temperature fallback sequence `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` — `DecodingConfig` has a single `temperature` scalar.
- Quality metrics — `compression_ratio_threshold` and `log_prob_threshold` are not in `DecodingConfig`; add `flate2` for the compression ratio (`len(text) / len(gzip(text))`).
- The CLI greedy loop needs to be replaced with a call to `HybridDecoder` and the 50-token cap raised to `max_length`.

## Hard-won lessons

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

### Phase 1 — Integration
Wire up code that already exists but isn't called. No new algorithms.

```
fix: use loaded model config for layer-norm defaults in load_whisper
```
- `whisperforge-core/src/load.rs:121–129`: replace hardcoded `WhisperConfig::tiny_en()` with the config that was just loaded from disk. Add a test asserting the layer-norm dims match the loaded model. This is the gate for all multi-model work.

```
feat: wire HybridDecoder into CLI replacing hand-rolled greedy loop
```
- `whisperforge-cli/src/main.rs`: delete lines 213–296, call `decoder.decode(encoder_output, ...)`, rename `_decoder` → `decoder`, raise the token cap from 50 to `decoding_config.max_length`.

```
feat: replace temperature scalar with fallback sequence in DecodingConfig
```
- `whisperforge-core/src/decoding.rs`: change `temperature: f32` to `temperatures: Vec<f32>`, default `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. Update all callers.

After phase 1: `tiny.en` produces full-length transcripts driven by beam search.

### Phase 2 — SOTA Decoding
Quality-gated temperature fallback — the key differentiator of faster-whisper.

```
feat: add compression ratio quality metric to DecodingConfig
```
- Add `flate2` to `whisperforge-core/Cargo.toml`. Implement `fn compression_ratio(text: &str) -> f32` (`text.len() as f32 / gzip(text).len() as f32`). Add `compression_ratio_threshold: f32` (default 2.4) to `DecodingConfig`.

```
feat: add log probability and no-speech quality metrics
```
- Add `log_prob_threshold: f32` (-1.0) and `no_speech_threshold: f32` (0.6) to `DecodingConfig`. Extract average log-prob from token scores; extract no-speech probability from the `<|nospeech|>` token's first-step logit.

```
feat: implement temperature fallback loop in HybridDecoder
```
- Loop over `temperatures` in `HybridDecoder`. After each attempt: compute `compression_ratio`, `avg_log_prob`, `no_speech_prob`. Return empty string early if `no_speech_prob > threshold`. Retry at next temperature if quality below threshold. Return best result on exhaustion.

```
test: benchmark SOTA decoding against OpenAI Whisper reference output
```
- Commit a handful of short Common Voice audio fixtures + their reference transcripts. Assert WER is within an acceptable delta vs. OpenAI Whisper Python output on the same clips.

After phase 2: decoding quality matches faster-whisper on standard benchmarks.

### Phase 3 — Multi-model + VAD

```
feat: complete tensor name mapping in whisperforge-convert
```
- `whisperforge-convert/src/convert.rs`: finish `load_encoder()` and `load_decoder()`. Key mapping: OpenAI `decoder.layers.X.encoder_attn.*` → Burn `blocks.X.cross_attn.*`. Run `cargo run -p whisperforge-convert` on `base` and verify no pathological outputs.

```
test: verify base and small model loading and forward pass
```
- Integration tests that load `base` and `small` weights, run the encoder on a synthetic mel spectrogram, and assert output shape. Document that `medium`/`large-v2` are excluded from CI (too large for automated test infra).

```
feat: wire VoiceActivityDetector into CLI pipeline
```
- `whisperforge-cli/src/main.rs`: add `whisperforge-align` as a dependency. When `--vad-enabled`, call `VoiceActivityDetector::detect()`, feed voice segments to the transcription loop, skip silence spans.

```
feat: 30-second chunked transcription with overlap for long audio
```
- VAD → merge gaps below threshold → split at ≤30s with 1s overlap → transcribe each chunk → stitch by stripping overlapping tokens at boundaries. End-to-end test on a ≥2-minute audio file.

After phase 3: all model sizes work; hour-long audio transcribes correctly.

### Phase 4 — Structured Output + SRT

```
feat: implement WhisperInference trait and populate TranscriptionResult
```
- `whisperforge-core/src/lib.rs`: implement `WhisperInference<B>` for `Whisper<B>`. Populate `TranscriptionSegment` start/end from VAD segment boundaries. Approximate word timestamps as `segment_duration / word_count` (placeholder until Phase 5).

```
feat: add --output-format srt via SrtWriter
```
- Parse `--output-format [text|srt|json]` in the CLI. Wire `SrtWriter` from `whisperforge-align` for `srt`. Smoke-test: `whisperforge -a audio.wav -m small --output-format srt > out.srt` and inspect the file.

```
feat: add --output-format json
```
- Serialize `TranscriptionResult` to JSON. This commit is the end of the "useful product" milestone — all three output formats work.

After phase 4: `whisperforge -a audio.wav -m small --output-format srt > output.srt` works.

### Phase 5 — Word-Level Timestamps
Two options. Ship Option A first; Option B is a separate follow-on.

**Option A — attention-based (~100ms precision, 2 commits)**

```
feat: capture cross-attention weights during decoding
```
- Modify the decoder forward pass to optionally return attention weights per step. Gate behind a `DecodingConfig` flag so there's no overhead by default.

```
feat: extract per-token timestamps from cross-attention peaks
```
- At each decoder step, argmax over encoder frame dimension of the cross-attention weight → `frame_index * hop / sample_rate`. Populate `word_start`/`word_end` in `TranscriptionSegment`.

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
