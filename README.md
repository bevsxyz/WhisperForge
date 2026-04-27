# WhisperForge

A Rust rewrite of WhisperX using [Burn 0.20](https://burn.dev/) for CUDA-accelerated speech transcription with word-level timestamps and speaker diarization.

## Goals

- SOTA decoding matching faster-whisper (beam search + temperature fallback + quality metrics)
- All Whisper model sizes (tiny.en through large-v2/v3)
- Word-level timestamps (<50ms precision via forced alignment)
- Speaker diarization with speaker labels in SRT/JSON output
- 8× realtime on CUDA for large-v2; ≤2GB VRAM

## Current Status

The model architecture, audio pipeline, and beam-search decoders are all implemented. The CLI currently runs a hand-rolled 50-token greedy loop and does not call the actual decoders. See `CLAUDE.md` for the detailed gap analysis and the 7-phase roadmap to close it.

| Layer | Status |
|-------|--------|
| Whisper model (all sizes) | Complete — frozen in `model.rs` |
| Audio / mel spectrogram | Complete |
| Beam search + hybrid decoder | Implemented, not wired to CLI |
| Model loader (NamedMpk) | Works for `tiny.en`; layer-norm bug blocks other sizes |
| Model converter (HF → Burn) | Config inference done; tensor mapping incomplete |
| VAD + segmentation | Implemented in `whisperforge-align`, not integrated |
| SRT output | Implemented in `whisperforge-align`, not integrated |
| Word-level timestamps | Not started |
| CUDA backend | Not started |
| Speaker diarization | Stub only |

## Tech Stack

- **[Burn 0.20](https://burn.dev/)** — ML framework (NdArray CPU now; CUDA in Phase 6)
- **rubato** — audio resampling
- **hound** — WAV I/O
- **tokenizers** — BPE tokenization
- **earshot** — voice activity detection (imported, not yet wired)
- **pyannote-rs** — speaker diarization (planned Phase 7)

## Quick Start

```bash
# Check compilation
cargo check --all

# Tests (always exclude whisperforge-align — known pre-existing failures)
cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli

# Format + lint
cargo fmt --all && cargo clippy --all-targets --all-features

# Run CLI (requires converted model files)
cargo run -p whisperforge-cli -- -a audio.wav -m tiny_en_converted
```

## Model Files

Models are git-ignored. Download from HuggingFace and convert:

```bash
cargo run -p whisperforge-convert -- --model openai/whisper-tiny.en --output models/tiny_en_converted
```

Required files per model:
```
models/
├── tiny_en_converted.mpk   # Burn NamedMpk weights
├── tiny_en_converted.cfg   # JSON config
└── tokenizer.json          # BPE tokenizer
```

## Architecture

Five-crate workspace:

| Crate | Role |
|-------|------|
| `whisperforge-core` | Whisper model, audio, decoding — primary dev area |
| `whisperforge-cli` | `whisperforge` binary |
| `whisperforge-convert` | HuggingFace safetensors → Burn NamedMpk conversion |
| `whisperforge-align` | VAD, segmentation, SRT output |
| `whisperforge-diarize` | Speaker diarization (placeholder) |

See `CLAUDE.md` for full architecture details, known bugs, hard-won lessons, and the development roadmap.

## License

[Add your license here]
