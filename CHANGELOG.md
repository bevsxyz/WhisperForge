# Changelog

All notable changes to WhisperForge are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
All workspace crates are versioned together.

---

## [0.5.2] — 2026-06-03

## [0.5.2] — 2026-06-03

## [0.5.1] — 2026-06-03

## [0.5.0] — 2026-06-03

### Added
- Gate INT8 dequant on B::supports_dtype(QFloat); enable native INT8 on CUDA
- Add workspace deps and stream subcommand stub (Phase F C1)
- Mic capture → 16 kHz ring buffer (Phase F C2)
- Silero VAD ONNX wrapper (Phase F C3)
- VAD-gated sliding-window chunker (Phase F C4)
- Partial-window mel API (Phase F C5)
- Greedy KV-cached decode primitive (Phase F C6)
- LocalAgreement-2 committer (Phase F C7)
- Silence+punctuation endpointer (Phase F C8)
- Prompt-prefix context across utterances (Phase F C9)
- Terminal UI + full streaming pipeline (Phase F C10)
- JSON output + record/transcript file sinks (Phase F C11)
- Synthetic-mic harness + latency bench + integration test (Phase F C12)
- Pass UAT — growing buffer + detok + natural EOU + notimestamps
- Rubato resampler + lock-free cpal callback (B7/B8)
- Expose dropped-sample counter; warn when decoder lags
- Tune live-mic defaults + add latency tracing
- Cap max_window_secs 28→10 + max_new_tokens 128→48
- Tighten live-mic defaults (5s window / 1s stride / 32 max_new_tokens)
- Long-form continuity via stride-based cap-hit trim (B9)
- Per-window decode metrics (avg/min logprob) via on_decode_metrics sink
- Faster-whisper-style logprob + compression-ratio quality gate
- Wire up --language and add --task for multilingual decode
- Cache-dir-aware models dir, scan_models, unified list command
- Pull replaces convert with a polymorphic model arg
- Transcribe UX — positional audio, grouped flags, enums, vtt
- Stream presets + grouped flags
- Global --models-dir, interactive model picker, shell completions

### Build
- Add mise eval hooks to Claude environment
- Make release builds portable; ship stream_bench as example
- Scaffold package-manager manifests and distribution guide
- Rename homebrew tap to generic bevsxyz/homebrew-tap
- Enable homebrew + npm publish jobs; document PAT/token scopes

### Changed
- Rename binary to wforge; ship wf as release-artifact shim
- Dedup list_input_devices to use core impl (B10)
- Make BatchedTranscriber a proper library interface with integration roadmap
- Per-model directory with own tokenizer

### Documentation
- Add v0.4.0 entry to workspace-root CHANGELOG.md
- Record Burn 0.21 CUDA INT8 inference is broken (q_slice unimplemented)
- Refresh streaming section to match current state (B11)
- Record Phase F UAT pass and fix doc staleness
- Surface stream_bench latency profiler + apply quality gate
- Document the experimental translate-into-X language coercion
- Update CLAUDE.md, README, and skills for the 0.5.0 CLI

### Fixed
- Pre-release-hook regenerates root CHANGELOG.md too
- Bump hf-hub to 0.5 and drop Windows GPU carve-out
- Use Start-Process so binary stderr isn't raised as a terminating error
- Run accuracy pass at --max-window-secs 28 (supported long-form path)
- Correct mise hook commands from eval to activate
- Simplify mise hook to actually work

### Performance
- Enable mold linker and sccache for faster builds

[0.5.0]: https://github.com/bevsxyz/WhisperForge/releases/tag/whisperforge-v0.5.0

---

## [0.4.0] — 2026-05-22

**Breaking — Phase E: foundation refactor (crate merger + CLI UX + device selection).**

### Removed (breaking)

- `whisperforge-convert` crate removed — model conversion lives in `wf convert`.
- `whisperforge` binary alias dropped — only `wf` ships.
- `--cpu` flag on the transcribe path — replaced by `--device cpu`.
- `--task` flag on the transcribe path — parsed-then-runtime-errored before; less code, clearer help.
- `--wgpu` flag (vestigial; was already replaced by feature-driven dispatch).

### Added

- `wf` subcommand surface: `transcribe`, `convert`, `list-models`. No more implicit-default subcommand — must specify one explicitly.
- `wf list-models`: tabulates `.mpk` models under the models directory with precision, `n_audio_layer`, `n_mels`, file size.
- `--device <auto|cpu|wgpu|cuda>` on `wf transcribe`. `auto` prefers CUDA → WGPU → CPU based on compiled-in features.
- Native CUDA backend via optional `burn-cuda` 0.21.0 (opt-in `--features cuda`; requires CUDA toolkit at build time).
- `--models-dir <PATH>` and `WF_MODELS_DIR` env var honored by `transcribe` and `list-models`.
- Friendly missing-model error pointing at `wf list-models` and the matching `wf convert` invocation.

### Changed

- `whisperforge-cli` crate renamed to `whisperforge`. Workspace shrinks 5 → 4 crates.
- mise `release-check` smoke-tests `--device cuda` and the missing-model friendly error.
- Project docs (CLAUDE.md + README + per-crate READMEs) updated for the new CLI surface; Roadmap marks Phase E ✅ COMPLETE and Phase D (WASM) ⏸ DEFERRED. Next: Phase F = streaming realtime.

### Migration

| Before | After |
|---|---|
| `whisperforge-cli -a foo.wav -m tiny_en` | `wf transcribe -a foo.wav -m tiny_en` |
| `cargo run -p whisperforge-convert -- --output models/tiny_en` | `wf convert --output models/tiny_en` |
| `wf -a foo.wav --cpu` | `wf transcribe -a foo.wav --device cpu` |
| `wf -a foo.wav --wgpu` | `wf transcribe -a foo.wav --device wgpu` (or default `auto`) |
| `wf -a foo.wav --task translate` | (removed — never worked) |

### Deferred (Phase E carve-outs, queued for later phases)

- VRAM-aware `--encoder-batch-size` auto-tune. Users override with `--encoder-batch-size` until a workload justifies the heuristic cost.
- Windows wgpu runtime fallback. Upstream `windows`-crate conflict between `wgpu-hal` and `gpu-allocator` is compile-time; no runtime probe can rescue it. Windows binary still ships CPU-only.

[0.4.0]: https://github.com/bevsxyz/WhisperForge/releases/tag/v0.4.0

---

## [0.2.0] — 2026-04-30

### Added

**Core transcription pipeline (`whisperforge-core`)**
- Whisper model architecture for all sizes (tiny.en through large-v2/v3), built on Burn 0.20.
- Audio pipeline: WAV loading with format dispatch (f32/i16/i24/i32), rubato resampling to 16 kHz mono, Slaney-scale mel spectrogram matching Python Whisper (power spectrum, center-padding, 80-band filter bank).
- `HybridDecoder` with quality-gated temperature fallback matching faster-whisper SOTA: compression ratio gate (2.4), log-probability gate (−1.0), no-speech threshold (0.6), temperature sequence [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
- `KvCache<B>` + `forward_decoder_cached`: O(n) per step via static cross-attention K,V and growing self-attention K,V — ~2.6× speedup over naive O(n²) decoder.
- `batch_mel_spectrograms`: all audio chunks mel-encoded in a single `forward_encoder` call before sequential decoding.
- `transcribe_with_timestamps`: per-token timestamps via cross-attention peak, ~100 ms precision.
- `extract_speaker_embedding`: mean-pool encoder output + L2-normalise → speaker fingerprint.
- `TranscriptionResult` / `TranscriptionSegment`: fully serde-serializable; `token_timestamps` and `speaker` fields skipped when absent.
- 0.8% average WER on LJSpeech `tiny.en` benchmark.

**VAD and alignment (`whisperforge-align`)**
- `VoiceActivityDetector` via `earshot` with configurable threshold.
- `AudioSegmenter`: splits audio into voice spans ≤30 s, filters silence.
- `BatchedTranscriber<B>`: real batch transcription pipeline — `batch_mel_spectrograms` → `forward_encoder` → per-segment KV-cached greedy decode → `HybridDecoder`.
- `SrtWriter` / `SrtEntry`: generates well-formed SRT subtitle files.

**Speaker diarization (`whisperforge-diarize`)**
- `SpeakerDiarizer`: agglomerative single-linkage clustering of speaker embeddings.
- `cluster_embeddings`: cosine similarity with configurable threshold, `SPEAKER_NN` label assignment.

**Model conversion (`whisperforge-convert`)**
- Converts OpenAI Whisper safetensors (via HuggingFace `hf-hub`) to Burn `NamedMpk` format.
- Correct tensor name mapping for all encoder and decoder layers including cross-attention.
- Config auto-detection for all model sizes from tensor shapes.

**CLI binary (`whisperforge` / `wf`)**
- Two binary aliases: `whisperforge` and `wf` (same binary, shorter alias).
- `--audio-file` / `-a`: WAV input.
- `--model` / `-m`: model directory under `models/` (default: `tiny_en_converted`).
- `--output-format text|srt|json`: plain text, SRT subtitles, or JSON with segments and timestamps.
- `--decoding-preset fast|balanced|accurate`: pre-configured beam size and temperature sequences.
- `--beam-size`, `--temperature`, `--length-penalty`, `--no-speech-threshold`: per-run overrides.
- `--vad-enabled` / `--vad-threshold`: voice activity detection; silence segments skipped.
- `--wgpu`: WGPU GPU backend (Vulkan, DX12, or Metal — no CUDA required).
- `--diarize` / `--diarize-threshold`: speaker labels (`[SPEAKER_NN]:`) in SRT and JSON.
- `--task transcribe|translate`, `--language`: Whisper task and language selection.
- Automatic 30 s chunking with 1 s overlap for long audio.

### Notes

- Model files (`.mpk`, `.cfg`, `tokenizer.json`) are not bundled — download from HuggingFace and convert with `whisperforge-convert`. See README for instructions.
- Requires Rust 1.85+ (Rust 2024 edition).
- `--wgpu` requires Vulkan/DX12/Metal drivers.
- `whisperforge-align` has known pre-existing test failures; always excluded from the test suite.
- Phase 7 Option B (ResNet293 speaker embeddings via burn-onnx) is deferred pending Burn 0.21 stable.

[0.2.0]: https://github.com/bevsxyz/WhisperForge/releases/tag/v0.2.0
