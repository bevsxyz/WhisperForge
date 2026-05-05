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

Five-crate Rust workspace using [Burn 0.20](https://burn.dev/) for GPU-accelerated ML inference.

| Crate | Role |
|-------|------|
| `whisperforge-core` | Whisper model + decoding — primary development area |
| `whisperforge-cli` | `whisperforge` binary |
| `whisperforge-convert` | One-shot HuggingFace safetensors → Burn NamedMpk conversion |
| `whisperforge-align` | VAD, segmentation, SRT output (has known test failures; work cautiously) |
| `whisperforge-diarize` | Placeholder — not yet integrated |

### Data flow

```
Audio → symphonia probe+decode (WAV/MP3/FLAC/OGG/M4A; format dispatch automatic) → resample to 16 kHz mono
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
- [src/audio.rs](whisperforge-core/src/audio.rs) — mel spectrogram, audio I/O, `batch_mel_spectrograms`.
- [src/decoding.rs](whisperforge-core/src/decoding.rs) — `DecodingConfig`, `BeamSearchDecoder`, `GreedyDecoder`, `HybridDecoder`.
- [src/kv_cache.rs](whisperforge-core/src/kv_cache.rs) — `KvCache<B>` and `forward_decoder_cached`: O(n) decoder via cached cross-attn K,V and growing self-attn K,V.

All model types are generic over `B: Backend`. CLI defaults to `NdArray<f32>` (CPU); pass `--wgpu` at runtime to use the WGPU backend (Vulkan/DX12/Metal — works on Intel, AMD, NVIDIA without CUDA).

## What is actually wired vs what isn't

### CLI decoding (`whisperforge-cli/src/main.rs`)

All chunks are mel-encoded in sub-batched encoder passes (`batch_mel_spectrograms` → `forward_encoder`; default: 1 chunk/pass on GPU, 4 on CPU; override with `--encoder-batch-size`). Each chunk is then decoded sequentially using `forward_decoder_cached` with a `KvCache` — O(n) per step instead of O(n²). The greedy loop passes per-step logits to `HybridDecoder.decode_with_fallback()` for quality-gated temperature fallback. EOT is suppressed at step 0. `--decoding-preset`, `--beam-size`, `--temperature`, and `--length-penalty` all take effect.

`--vad-enabled` / `--vad-threshold` are fully wired. When `--vad-enabled`, `AudioSegmenter` (which delegates to `VoiceActivityDetector::detect()`) segments audio into voice spans; silence is skipped; segments feed the transcription loop with accurate timestamps.

`--wgpu` selects the WGPU GPU backend (Vulkan/DX12/Metal). The binary is generic over `B: Backend`; `main()` dispatches to `run::<Wgpu>` or `run::<NdArray<f32>>` based on this flag.

### Unimplemented scaffolding

- `BatchedTranscriber` in `whisperforge-align` — stub that returns placeholder text; real transcription not wired.

### Implemented (Phases 4–7)

- `WhisperTranscriber<B>` in `whisperforge-core/src/transcribe.rs` — wraps `Whisper<B>` + tokenizer + config; implements `WhisperInference<B>`. `transcribe_with_timestamps()` uses cross-attention peaks for per-token timestamps.
- `TranscriptionResult` / `TranscriptionSegment` — fully serializable (serde); `token_timestamps` serde-skipped when empty.
- `--output-format text|srt|json` in the CLI — text is plain join; srt uses `SrtWriter`; json uses `serde_json`.
- `KvCache<B>` + `forward_decoder_cached` — O(n) decoder; cross-attn K,V static per chunk; self-attn K,V grow per step.
- `batch_mel_spectrograms` — all chunks mel+encoded in one batch before the decode loop.
- `extract_speaker_embedding<B>` — mean-pools encoder output → L2-normalised `Vec<f32>` speaker fingerprint (Option A).
- `SpeakerDiarizer` + `cluster_embeddings` in `whisperforge-diarize` — agglomerative single-linkage clustering, `SPEAKER_NN` label assignment.
- `--diarize` / `--diarize-threshold` in CLI — speaker labels in SRT (`[SPEAKER_NN]: text`) and JSON (`"speaker"` field). Option B upgrade path: `--diarize-model <path>` (planned, not yet wired).

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

### WAV loading: use symphonia, not hound

`hound` was replaced by `symphonia-default 0.5` to support MP3/FLAC/OGG/M4A alongside WAV. Symphonia handles format dispatch (float/int/compressed) automatically — no manual `(sample_format, bits_per_sample)` switch needed. The old hound `into_samples::<f32>()` hazard (silent or clipped output on integer WAV) is gone.

### Stale incremental artifacts can mimic compiler incompatibilities

Errors like missing methods, proc-macro failures, and macro hygiene breakage after a compiler version change are often stale `.rlib` files, not real bugs. Run `cargo clean` before concluding a crate is broken under the current toolchain. Confirmed: symphonia-core and tokenizers both appeared broken on Rust stable but worked cleanly after `cargo clean`.

### Burn 0.20 API differences

- `.squeeze()` takes **no arguments** in 0.20 (changed from older versions).
- Always use `NamedMpkFileRecorder::<FullPrecisionSettings>::new()` for saving/loading models.
- When a function can't be found, search the Burn crate source — don't assume names from older docs.

### whisperforge-align test failures

Pre-existing; always exclude: `cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli`.

## Roadmap

Phases 1–7 (CLI, KV cache, GPU backend, diarization) complete. **New direction**: library + streaming + WASM.

Five new phases toward browser-native and large-file transcription. Work in order — each unblocks the next.

Each phase is one or more commits. Every commit must leave `cargo check --all` clean and the test suite passing before it lands.

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

### Phase 5 — Word-Level Timestamps ✅ COMPLETE
Two options. Option A shipped; Option B is a separate follow-on.

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

### Phase 6 — GPU + Performance ✅ COMPLETE (INT8 deferred)

```
✅ fix: support float32 WAV files in load_wav_file
```
- Done: dispatch on `(sample_format, bits_per_sample)`; float32 WAV (externally resampled files) now load correctly instead of producing garbage spectrograms.

```
✅ perf: KV-cache encoder cross-attn and decoder self-attn
```
- Done: `whisperforge-core/src/kv_cache.rs` — `KvCache<B>` pre-computes encoder K,V once per chunk; accumulates decoder self-attn K,V per step with `Tensor::cat`. Reduces decode from O(n²) to O(n) total work. ~2.6× per-token speedup on `tiny.en` CPU. Three unit tests including numerical equivalence with `forward_decoder` (max_diff < 1e-4).

```
✅ feat: add WGPU GPU backend behind --wgpu flag
```
- Done: `burn = { features = ["wgpu"] }` in `whisperforge-cli`. `main()` is generic: dispatches to `run::<Wgpu>(args, WgpuDevice::default())` or `run::<NdArray<f32>>`. Works on Intel/AMD/NVIDIA via Vulkan, DX12, or Metal — no CUDA required.

```
✅ perf: batch spectrogram + encoder across all chunks
```
- Done: `batch_mel_spectrograms()` in `audio.rs` computes all chunk mels and cats to `[N, 80, 3000]`. `run()` calls `model.forward_encoder(batch_mel)` once → `[N, 1500, D]`, then slices per chunk. `transcribe_chunk` now takes `encoder_output: Tensor<B, 3>` directly.

```
✅ feat: multi-format audio (MP3, FLAC, OGG, M4A) — replace hound with symphonia
```
- Done: `load_audio_file` in `audio.rs` uses symphonia probe+decode loop; auto-detects format; outputs f32 samples regardless of source format. Replaces all hound-based loading.

```
✅ fix: sub-batch encoder to prevent OOM on long audio
```
- Done: encoder forward pass runs in sub-batches (GPU default=1, 54 MB; CPU default=4, 216 MB); `--encoder-batch-size` overrides. GPU default=1 chosen because model weights (~136 MB) + batch-4 attention matrix (216 MB) exhausts most consumer GPUs.

```
⏸ perf: INT8 quantization for VRAM reduction
```
- Deferred: Burn 0.20 does not expose a stable quantization API. Revisit when bumping to Burn 0.21+.

After phase 6: all chunks encoded in one GPU batch; KV cache gives O(n) decode; `--wgpu` enables GPU on any Vulkan/DX12/Metal device.

### Phase 7 — Speaker Diarization ✅ COMPLETE (Option A)

Two embedding strategies, both wired through the same clustering and CLI path. **Option A is the default and always works with no extra model file.** Option B is a drop-in upgrade — same flags, better embeddings.

#### Option A — Whisper encoder mean-pooling ✅ SHIPPED

```
✅ feat: speaker embeddings via Whisper encoder mean-pooling
```
- Done: `whisperforge-core/src/embed.rs` — `extract_speaker_embedding<B: Backend>(Tensor<B,3>) -> Result<Vec<f32>>`. Mean-pools the `[1, 1500, D]` encoder output over the time dimension then L2-normalises. Zero extra dependencies — the encoder output is already computed during transcription.

```
✅ feat: SpeakerDiarizer assigns speaker labels to segments
```
- Done: `whisperforge-diarize/src/clustering.rs` — `cosine_similarity`, `cluster_embeddings` (agglomerative single-linkage, pure Rust). `whisperforge-diarize/src/diarizer.rs` — `SpeakerDiarizer::assign_labels(&[Vec<f32>]) -> Vec<String>`, formats as `SPEAKER_NN`, handles zero-norm embeddings as `SPEAKER_UNKNOWN`. 11 unit tests.

```
✅ feat: --diarize flag with speaker labels in SRT and JSON output
```
- Done: `speaker: Option<String>` on `TranscriptionSegment` (serde-skipped when `None`). CLI extracts encoder embedding per chunk, clusters, assigns labels. SRT prefixed `[SPEAKER_NN]: text`; JSON automatic via serde. Flags: `--diarize`, `--diarize-threshold` (default 0.7).

**Quality note:** Works well for clearly distinct voices (different gender/accent). Same-gender similar-accent speakers may not separate cleanly — Option B addresses this.

#### Option B — ResNet293 via burn-onnx ⏸ DEFERRED (pending Burn 0.21 stable)

ResNet293 is the embedding model used by pyannote/speaker-diarization-3.1 (~0.6% EER on VoxCeleb1-O vs ~1.5% for the mean-pool baseline). Integrates as a **runtime upgrade**: `--diarize-model models/speaker_resnet293.mpk` switches the embedding source; clustering and CLI flags are unchanged.

**Why deferred:**
- `burn-import 0.20.1` fails on ResNet293's ONNX with `TypeMismatch { expected: "Tensor", actual: "Shape(1)" }` — the 0.20 type inferencer can't handle `Shape` ops used for dynamic reshaping in the PyTorch export.
- `burn-onnx` (Burn 0.21) fixes this, but Burn 0.21.0-pre.4 has a **breaking `BackendTypes` split** that would require modifying the frozen `model.rs`. Not safe until 0.21 stable ships with a migration guide.
- `ort` was ruled out to keep the workspace dependency-free of a second ML runtime.

**ONNX model is available:** `Wespeaker/wespeaker-voxceleb-resnet293-LM` on HuggingFace (public, no auth, 39MB ONNX). The file `models/speaker_resnet293.onnx` may already be present locally.

**When Burn 0.21 stable ships, implementation is:**
1. Replace `burn = "0.20"` with `burn = "0.21"` across workspace; verify `model.rs` compiles (watch for BackendTypes changes to `B: Backend` bounds).
2. Add `burn-onnx` as a build dependency to `whisperforge-diarize`.
3. Write `build.rs` using `burn_onnx::ModelGen` on `models/speaker_resnet293.onnx` → `src/generated/`.
4. Implement `SpeakerEmbeddingModel` wrapper in `src/speaker_model.rs`: takes `&[f32]` at 16 kHz, computes 80-dim log-mel, runs ResNet293, returns 256-dim L2-normalised embedding.
5. Add `--diarize-model <path>` CLI flag: if provided, use `SpeakerEmbeddingModel`; otherwise fall back to Option A. Both paths feed the same `SpeakerDiarizer::assign_labels`.

After Option B: pyannote-level diarization quality, no ort required, GPU-accelerated via `--wgpu`.

---

### Phase A — Library API + Bytes Loading ✅ COMPLETE

```
✅ feat(phase-a): library API — bytes loaders, file-io feature gate, wgpu default
```
- Done: `load_config_from_bytes`, `load_whisper_from_bytes` added (WASM-compatible, no filesystem).
- Done: `load_whisper`, `load_config`, `load_audio_file` gated behind `file-io` feature (on by default).
- Done: `symphonia` marked optional; pulled in only under `file-io`.
- Done: CLI default flipped to WGPU; `--cpu` flag opts out to NdArray.

After Phase A: library is embeddable and publishable to crates.io; bytes-based loading unblocks WASM.

### Phase B — Streaming Audio ✅ COMPLETE

```
✅ feat: implement Phase B streaming audio pipeline
```
- Done: `whisperforge-core/src/audio_stream.rs` — `AudioChunkIterator` pull-based iterator decodes symphonia packets on-demand, resamples incrementally with rubato state machine (chunk_size=128, 1 mono channel), yields 30s chunks with 1s overlap. Holds at most one packet + resampler state + overlap window (~2 MB) at a time.
- Done: CLI streaming loop replaces eager load-all-then-process — collects `enc_batch` chunks on-demand, encodes together, decodes sequentially. Peak working set = model + enc_batch × 30s audio + mel tensor + encoder output, regardless of file length.
- Done: VAD path still uses eager loading (VAD requires full audio signal); documented limitation.
- Done: `AudioChunk` and `AudioChunkIterator` exported from `lib.rs` under `file-io` feature.
- Async file I/O (`tokio::fs`) and WASM microphone input (`MediaStream`) deferred to Phase D — JS drives async in the browser anyway.

**Bugs fixed post-ship:**
- Stereo MP3: resampler created for N channels but received mono data → fixed by creating resampler with 1 channel.
- MP3 packet underflow: 576 mono frames/packet < 1024 chunk_size → no resampling occurred → fixed by lowering chunk_size to 128.
- EOF overlap loop: after last real chunk, overlap buffer recycled as new chunk forever → fixed by checking `samples.len() <= overlap_len`.

After Phase B: files of any length stream without OOM; `AudioChunkIterator` unblocks Phase D WASM microphone input.

### Phase B.5 — CubeCL Mel Pipeline 🔄 IN PROGRESS

**Goal:** Replace CPU `rustfft` STFT with CubeCL GPU kernels. Same kernels run natively (Vulkan/DX12/Metal) and in browser (WebGPU).

```
✅ perf: GPU mel filterbank + log compression via Burn tensor matmul
```
- Done: `compute_mel_spectrogram` now uses Burn tensor matmul for the mel filterbank (`[n_mels, n_freqs] @ [n_freqs, n_frames]`) and log10/normalization as GPU tensor ops. CPU STFT (rustfft) unchanged. Power spectrum is uploaded as a small tensor (~2 MB); matmul, log, clamp, normalize all run on device.
- Eliminated the CPU triple-nested loop (80 × 3001 × 201 = 48M multiplications) that was the primary non-STFT bottleneck.

```
✅ feat: CubeCL DFT kernel in stft_gpu.rs (feature-gated cubecl-stft)
```
- Done: `whisperforge-core/src/stft_gpu.rs` — `stft_power_kernel<F: Float>` `#[cube(launch_unchecked)]` kernel. One cube per STFT frame, 201 units per cube. Cooperative shared-memory load of Hann-windowed samples; runtime DFT inner loop (not unrolled; generates loop instruction). `compute_stft_power_gpu<R: Runtime>` launcher returns `Vec<f32>` power spectrum.
- Feature: `cubecl-stft` in `whisperforge-core/Cargo.toml` (adds `cubecl = "0.9"` dep). Exported as `pub use stft_gpu::compute_stft_power_gpu` under the feature gate.
- Not yet wired into the main pipeline: `Wgpu = Fusion<CubeBackend<WgpuRuntime,...>>` — the Fusion wrapper prevents generic `B: Backend` dispatch from seeing the inner Runtime. Full wiring requires switching the CLI from `Wgpu` to bare `CubeBackend<WgpuRuntime,...>` (no Fusion). Deferred to Phase D prep.

**Remaining:**
- Wire `compute_stft_power_gpu` into `compute_mel_spectrogram` via bare `CubeBackend<WgpuRuntime,f32,i32,u32>` in CLI
- Add correctness test comparing GPU DFT to CPU rustfft (needs `burn-wgpu` dev-dep)

### Phase C — Quantization

**Goal:** Reduce model size 4× (FP32 ~150 MB → INT8 ~37 MB) for faster WASM downloads.

- Audit Burn 0.20 quantization API (`QuantizationScheme`, `Calibration`)
- Post-training quantization in `whisperforge-convert` with `--quantize int8`
- Runtime support: `load_whisper` / `load_whisper_from_bytes` accept `Precision` enum

**Why after B.5:** CubeCL is orthogonal; quantization applies to any backend.

### Phase D — WASM Target

**Goal:** Compile to `wasm32-unknown-unknown`; model + inference run client-side via WebGPU.

- New crate `whisperforge-wasm`: thin `wasm-bindgen` wrapper
- Exported API: `async fn transcribe(audio_samples, sample_rate, model_bytes, config_bytes) -> String`
- Browser example: `index.html` + `main.js`; records via `getUserMedia`; streams results
- Uses Phase B streaming (handles real-time microphone input) + Phase C quantization (browser-viable model size)

---

**Total: Phases 1–7 + A–B complete; B.5 in progress (GPU mel matmul done, CubeCL STFT kernel written, wiring pending); Phases C–D planned.**

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
