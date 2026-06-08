## [0.5.4] - 2026-06-04

### 🐛 Bug Fixes

- *(release-check)* Update smoke tests for Phase G CLI rename
- *(windows)* Align CRT to dynamic to resolve ONNX __imp_ link errors (#7)

### ⚙️ Miscellaneous Tasks

- Release v0.5.4
## [0.5.3] - 2026-06-04

### 🐛 Bug Fixes

- *(ci)* Link legacy_stdio_definitions.lib on Windows MSVC target (#6)

### ⚙️ Miscellaneous Tasks

- Remove sccache from CI and local dev
- Release v0.5.3
## [0.5.2] - 2026-06-03

### 🐛 Bug Fixes

- *(release)* Emit v{{version}} workspace tag; deduplicate CHANGELOG hook

### ⚙️ Miscellaneous Tasks

- Add changelog
- Release v0.5.2
## [whisperforge-v0.5.1] - 2026-06-03

### 📚 Documentation

- *(release)* Backfill 0.5.0 CHANGELOG; fix git-cliff tag pattern

### ⚙️ Miscellaneous Tasks

- Release v0.5.1
## [whisperforge-v0.5.0] - 2026-06-03

### 🚀 Features

- *(load)* Gate INT8 dequant on B::supports_dtype(QFloat); enable native INT8 on CUDA
- *(stream)* Add workspace deps and stream subcommand stub (Phase F C1)
- *(stream)* Mic capture → 16 kHz ring buffer (Phase F C2)
- *(stream)* Silero VAD ONNX wrapper (Phase F C3)
- *(stream)* VAD-gated sliding-window chunker (Phase F C4)
- *(stream)* Partial-window mel API (Phase F C5)
- *(stream)* Greedy KV-cached decode primitive (Phase F C6)
- *(stream)* LocalAgreement-2 committer (Phase F C7)
- *(stream)* Silence+punctuation endpointer (Phase F C8)
- *(stream)* Prompt-prefix context across utterances (Phase F C9)
- *(stream)* Terminal UI + full streaming pipeline (Phase F C10)
- *(stream)* JSON output + record/transcript file sinks (Phase F C11)
- *(stream)* Synthetic-mic harness + latency bench + integration test (Phase F C12)
- *(stream)* [**breaking**] Pass UAT — growing buffer + detok + natural EOU + notimestamps
- *(stream)* Rubato resampler + lock-free cpal callback (B7/B8)
- *(stream)* Expose dropped-sample counter; warn when decoder lags
- *(stream)* Tune live-mic defaults + add latency tracing
- *(stream)* [**breaking**] Cap max_window_secs 28→10 + max_new_tokens 128→48
- *(stream)* [**breaking**] Tighten live-mic defaults (5s window / 1s stride / 32 max_new_tokens)
- *(stream)* [**breaking**] Long-form continuity via stride-based cap-hit trim (B9)
- *(stream)* Per-window decode metrics (avg/min logprob) via on_decode_metrics sink
- *(stream)* Faster-whisper-style logprob + compression-ratio quality gate
- Wire up --language and add --task for multilingual decode
- *(cli)* [**breaking**] Cache-dir-aware models dir, scan_models, unified list command
- *(cli)* [**breaking**] Pull replaces convert with a polymorphic model arg
- *(cli)* [**breaking**] Transcribe UX — positional audio, grouped flags, enums, vtt
- *(cli)* Stream presets + grouped flags
- *(cli)* Global --models-dir, interactive model picker, shell completions
- *(cli)* [**breaking**] Cache-dir-aware models dir, scan_models, unified list command
- *(cli)* [**breaking**] Pull replaces convert with a polymorphic model arg
- *(cli)* [**breaking**] Transcribe UX — positional audio, grouped flags, enums, vtt
- *(cli)* Stream presets + grouped flags
- *(cli)* Global --models-dir, interactive model picker, shell completions

### 🐛 Bug Fixes

- *(release)* Pre-release-hook regenerates root CHANGELOG.md too
- *(convert)* Bump hf-hub to 0.5 and drop Windows GPU carve-out
- *(uat)* Use Start-Process so binary stderr isn't raised as a terminating error
- *(uat)* Run accuracy pass at --max-window-secs 28 (supported long-form path)
- *(build)* Correct mise hook commands from eval to activate
- *(build)* Simplify mise hook to actually work

### 💼 Other

- Streaming quality gate + cross-backend UAT (Phase F close-out)
- Add mise eval hooks to Claude environment
- Make release builds portable; ship stream_bench as example
- *(packaging)* Scaffold package-manager manifests and distribution guide
- *(dist)* Rename homebrew tap to generic bevsxyz/homebrew-tap
- *(dist)* Enable homebrew + npm publish jobs; document PAT/token scopes

### 🚜 Refactor

- *(cli)* [**breaking**] Rename binary to wforge; ship wf as release-artifact shim
- *(stream)* Dedup list_input_devices to use core impl (B10)
- *(align)* Make BatchedTranscriber a proper library interface with integration roadmap
- *(models)* [**breaking**] Per-model directory with own tokenizer
- *(models)* [**breaking**] Per-model directory with own tokenizer

### 📚 Documentation

- *(changelog)* Add v0.4.0 entry to workspace-root CHANGELOG.md
- *(claude-md)* Record Burn 0.21 CUDA INT8 inference is broken (q_slice unimplemented)
- *(readme)* Refresh streaming section to match current state (B11)
- *(stream)* Record Phase F UAT pass and fix doc staleness
- *(bench)* Surface stream_bench latency profiler + apply quality gate
- Document the experimental translate-into-X language coercion
- Update CLAUDE.md, README, and skills for the 0.5.0 CLI
- Document the experimental translate-into-X language coercion
- Update CLAUDE.md, README, and skills for the 0.5.0 CLI

### ⚡ Performance

- Enable mold linker and sccache for faster builds

### 🎨 Styling

- *(stream)* Fix clippy identity_op in STATE_ELEMS
- *(align)* Fix clippy doc-comment lints (rust 1.95, -D warnings)
- *(align)* Fix clippy doc-comment lints (rust 1.95, -D warnings)
- Fix clippy lints introduced by MSRV bump to 1.95

### 🧪 Testing

- *(stream)* Commit 16 kHz LJ001-0001 fixture for streaming UAT

### ⚙️ Miscellaneous Tasks

- *(release)* Trigger release workflow on whisperforge-v* tags too
- Add wforge stream permission + fmt cleanup
- Install libasound2-dev for cpal audio support in CI runner
- *(release)* Install libasound2-dev for Linux builds
- Drop redundant mise hooks from Claude settings
- Drop redundant mise hooks from Claude settings
- Add dist release pipeline and crates.io trusted publishing
- Release v0.5.0
## [whisperforge-v0.4.0] - 2026-05-22

### 🚀 Features

- *(cli)* Add 'wf list-models', friendlier model-not-found errors, drop --task translate
- *(cli)* Runtime --device flag (auto/cpu/wgpu/cuda)
- *(cli)* Native CUDA backend via burn-cuda (feature 'cuda')

### 🐛 Bug Fixes

- Initialize mise in pre-release-hook for git-cliff discovery
- *(release)* Use bash for pre-release-hook so mise activate parses

### 🚜 Refactor

- [**breaking**] Rename whisperforge-cli crate to whisperforge
- Fold whisperforge-convert into whisperforge as 'wf convert' subcommand

### ⚙️ Miscellaneous Tasks

- Add mise activation hook to PreToolUse for Bash
- Simplify release task to rely on mise environment
- *(dev)* Pin rust 1.95.0 across mise.toml and rust-toolchain.toml; drop broken bash hook
- *(release)* Phase E wrap — docs + Roadmap update
- Ignore Claude harness lock file + bare whisperforge/CHANGELOG.md
- Release v0.4.0
## [whisperforge-diarize-v0.3.2] - 2026-05-15

### 🐛 Bug Fixes

- Dequantize INT8 models on CPU before loading onto WGPU
- Add --version flag to CLI

### ⚙️ Miscellaneous Tasks

- Set up release infrastructure with git-cliff and cargo-release
- Add release task to mise.toml
- Ignore per-crate CHANGELOG.md files
- Improve release hook command
- Release v0.3.2
## [0.3.1] - 2026-05-14

### 🐛 Bug Fixes

- Resolve clippy warnings and FlexDevice backend migration
- Resolve remaining clippy warnings (borrow, FlexDevice, diverging)
- Reduce keywords to 5 per crate (crates.io limit)
- Resolve windows wgpu version conflict via target-specific dependencies
- Add gpu feature flag to control wgpu compilation on windows
- Guard wgpu imports behind gpu feature flag

### 📚 Documentation

- *(seo)* Enhance crates.io keywords, descriptions, and release notes for discoverability
- Fix version and backend references in README

### ⚙️ Miscellaneous Tasks

- *(release)* Add individual crate READMEs and bump to 0.3.1
- Disable wgpu on windows release build (cpu fallback available)
## [0.3.0] - 2026-05-14

### 🚀 Features

- Multi-format audio (MP3, FLAC, OGG, M4A) — replace hound with symphonia
- *(phase-a)* Library API — bytes loaders, file-io feature gate, wgpu default
- Implement Phase B streaming audio pipeline
- *(phase-b5)* GPU mel filterbank matmul + CubeCL STFT kernel
- *(phase-b5)* GPU mel filterbank matmul + CubeCL STFT kernel
- *(phase-c)* INT8 post-training quantization for model conversion

### 🐛 Bug Fixes

- Resolve all clippy -D warnings for CI
- Move wgpu feature from core to cli only — fixes dry-run publish
- Sub-batch encoder to prevent OOM on long audio (GPU=4, CPU=16)
- Encoder batch GPU=1 CPU=4 to prevent OOM
- Align CLI tokenizers to workspace version (0.19 → 0.20)
- Pin toolchain to 1.89.0 — fixes dev build (lazy_static/tokenizers compat)
- Revert toolchain pin to stable — phantom incompatibility after cargo clean
- Streaming audio resampler channel count mismatch
- Use chunk_size=128 so resampler fits any codec packet
- Stop overlap loop at EOF in AudioChunkIterator

### 📚 Documentation

- *(claude.md)* Add Phases A-D roadmap (library, streaming, quantization, wasm)
- *(claude.md)* Mark Phase B complete, document streaming bugs fixed
- *(phase-c)* Mark complete, document NdArray quantization limitation
- *(readme)* Improve formatting with badges and professional styling
- *(readme)* Add installation section and improve quick start

### ⚙️ Miscellaneous Tasks

- Strip absolute paths from CODEINDEX.md in PostToolUse hook
- *(release)* Bump version to 0.3.0
- *(release)* Add version constraints to workspace dependencies for crates.io publishing
## [0.2.0] - 2026-04-30

### 🚀 Features

- Initial WhisperForge implementation
- Add OpenAI to Burn 0.20 model converter and loader
- Implement SOTA beam search decoding module with CLI integration
- Add VAD parameters to CLI
- Wire HybridDecoder into CLI replacing hand-rolled greedy loop
- Replace temperature scalar with fallback sequence in DecodingConfig
- Add compression ratio quality metric to DecodingConfig
- Implement temperature fallback loop in HybridDecoder
- Stream tokens to stdout and fix generation budget in CLI
- Wire VoiceActivityDetector into CLI pipeline
- 30-second chunked transcription with overlap for long audio
- Implement WhisperInference trait and populate TranscriptionResult
- Add --output-format text|srt|json to CLI
- Cross-attention per-token timestamps (Phase 5 Option A)
- Wire VoiceActivityDetector into CLI pipeline
- Add WGPU GPU backend behind --wgpu flag
- Speaker embeddings via Whisper encoder mean-pooling
- SpeakerDiarizer assigns speaker labels to segments
- --diarize flag with speaker labels in SRT and JSON output
- Wire BatchedTranscriber with real transcription pipeline

### 🐛 Bug Fixes

- Update to Burn 0.20 API compatibility
- Resolve EOT domination issue in Whisper model conversion
- Resolve all 4 failing tests in whisperforge-align
- Use loaded model config for layer-norm defaults in load_whisper
- Load encoder ln_post and decoder ln from safetensors; remove workaround
- Correct mel spectrogram preprocessing to match Python Whisper
- Suppress EOT at step 0 in CLI greedy loop
- Support float32 WAV files in load_wav_file

### 🚜 Refactor

- Delegate conversion logic to library in whisperforge-convert

### 📚 Documentation

- Update project status - converter complete, mel spectrogram next
- Add Claude CLI optimization - README, CONTRIBUTING, and .claude configuration
- Update CLAUDE.md to reflect Phase 2 completion
- Update CLAUDE.md to reflect Phase 1 complete and Phase 3 progress
- Align CLAUDE.md with Phases 5 and 6 completion
- Document Phase 7 completion and Option B upgrade path

### ⚡ Performance

- KV-cache encoder cross-attn and decoder self-attn
- Batch spectrogram + encoder across all chunks

### 🧪 Testing

- Add LJSpeech fixtures for SOTA decoding benchmark (git LFS)
- Benchmark SOTA decoding against LJSpeech reference transcriptions
- Suppress EOT at step 0 in WER benchmark; 0.8% avg WER on tiny.en
- Add synthetic encoder shape tests for base and small configs
- Verify base and small model loading and forward pass

### ⚙️ Miscellaneous Tasks

- Add model files to gitignore
- Replace opencode session docs with consolidated CLAUDE.md
- Update docs to reflect current codebase reality
- Add Claude Code project settings with co-author attribution disabled
- Add mise-based git hooks for fmt and compile checks
- Defer Option B diarization to Burn 0.21 stable
- Add Claude Code hooks, permissions, and slash commands
- Prepare v0.2.0 — edition 2024, wf alias, CI, changelog, crates.io metadata
