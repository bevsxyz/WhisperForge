# WhisperForge

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![Burn 0.21](https://img.shields.io/badge/burn-0.21-blueviolet.svg)](https://burn.dev/)
[![GitHub release](https://img.shields.io/github/v/release/bevsxyz/WhisperForge?label=version)](https://github.com/bevsxyz/WhisperForge/releases)

[![whisperforge-core on crates.io](https://img.shields.io/crates/v/whisperforge-core.svg)](https://crates.io/crates/whisperforge-core)
[![whisperforge on crates.io](https://img.shields.io/crates/v/whisperforge.svg)](https://crates.io/crates/whisperforge)
[![whisperforge-align on crates.io](https://img.shields.io/crates/v/whisperforge-align.svg)](https://crates.io/crates/whisperforge-align)
[![whisperforge-diarize on crates.io](https://img.shields.io/crates/v/whisperforge-diarize.svg)](https://crates.io/crates/whisperforge-diarize)

A high-performance Rust implementation of OpenAI's Whisper speech-to-text model with GPU acceleration via WGPU (Vulkan/DX12/Metal). Features SOTA decoding algorithms, per-token timestamps, and speaker diarization.

[📖 Documentation](#architecture) · [🚀 Quick Start](#quick-start) · [💬 Model Files](#model-files) · [🔧 CLI Reference](#cli-reference)

## Features

- SOTA decoding matching faster-whisper (beam search + temperature fallback + quality gates)
- All Whisper model sizes (tiny.en through large-v2/v3)
- Per-token timestamps via cross-attention peaks (~100 ms precision)
- Speaker diarization with `SPEAKER_NN` labels in SRT/JSON output
- GPU acceleration via WGPU — works on any Vulkan, DX12, or Metal device (no CUDA required)
- Voice activity detection — silence segments skipped automatically
- Single `wf` binary

## Project Status

✅ **Production Ready** — All core features complete (Phases A–C).

| Component | Status |
|---|---|
| Whisper model loader (all sizes: tiny.en → large-v3) | ✅ Complete |
| Audio pipeline (mel spectrogram, resampling, VAD) | ✅ Complete |
| SOTA HybridDecoder (beam search + temperature fallback) | ✅ Complete |
| KV-cache O(n) decoder | ✅ Complete |
| Per-token timestamps (cross-attention) | ✅ Complete |
| SRT / JSON / text output | ✅ Complete |
| GPU acceleration (WGPU: Vulkan/DX12/Metal) | ✅ Complete |
| Speaker diarization (encoder embeddings) | ✅ Complete |
| INT8 quantization (~4× compression) | ✅ Complete |
| WASM target (browser support) | ⬜ Planned (Phase D) |

## Installation

### CLI Binary

**[📦 Install from crates.io](https://crates.io/crates/whisperforge)** — One command:
```bash
cargo install whisperforge
wf --help
```

**[🔨 Build from source](https://github.com/bevsxyz/WhisperForge)** — For development:
```bash
git clone https://github.com/bevsxyz/WhisperForge
cargo install --path ./whisperforge
```

### Library: Add to Your Project

For Rust projects, add WhisperForge crates to `Cargo.toml`:

```toml
[dependencies]
whisperforge-core = "0.3.1"        # Core: Whisper model & audio pipeline
whisperforge-align = "0.3.1"       # Optional: VAD, batched transcription, SRT
whisperforge-diarize = "0.3.1"     # Optional: Speaker diarization

[features]
gpu = ["whisperforge-core/cubecl-stft"]  # Optional: GPU via WGPU
```

Basic example:
```rust
use whisperforge_core::{Model, WhisperConfig};
use std::path::Path;

let config = WhisperConfig::new("tiny.en");
let model = Model::load(Path::new("models/tiny_en_converted"))?;
let transcript = model.transcribe(audio_samples, sample_rate)?;
println!("{}", transcript);
```

## Quick Start

### Using the CLI

After [installing](#installation), download a model and transcribe audio:

```bash
# Convert a Whisper model from HuggingFace
wf convert --model-id openai/whisper-tiny.en --output models/tiny_en_converted

# Transcribe (GPU by default; pass --cpu for the Flex CPU backend)
wf transcribe -a audio.wav -m tiny_en_converted

# SRT with speaker labels
wf transcribe -a audio.wav -m tiny_en_converted --output-format srt --diarize -o output.srt

# JSON output
wf transcribe -a audio.wav --output-format json
```

### For Contributors

```bash
# Check compilation
cargo check --all

# Tests (always use --release; exclude whisperforge-align — known pre-existing failures)
cargo test --release -p whisperforge-core -p whisperforge

# Format + lint
cargo fmt --all && cargo clippy --all-targets --all-features
```

## Model Files

Model weights are git-ignored. Convert from HuggingFace using the built-in `wf convert` subcommand:

```bash
# Basic: download and convert to Burn format
wf convert --model-id openai/whisper-tiny.en --output models/tiny_en_converted

# Quantized: INT8 compression (37 MB vs 150 MB for tiny.en)
wf convert --model-id openai/whisper-tiny.en --output models/tiny_en_quantized --quantize int8

# From local file instead of downloading
wf convert --local-safetensors /path/to/model.safetensors --output models/tiny_en_converted
```

**Converter options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `openai/whisper-tiny.en` | HuggingFace model identifier |
| `--output` | required | Output path without extension (generates `.mpk` + `.cfg`) |
| `--quantize` | `none` | Quantization: `none` (FP32) or `int8` (~4× compression) |
| `--local-safetensors` | — | Load from local safetensors file instead of downloading |

**Generated files:**
```
models/
├── <model_name>.mpk        # Burn model weights
├── <model_name>.cfg        # Metadata (precision, config)
└── tokenizer.json          # BPE tokenizer (shared across models)
```

Both the CLI and library load models from `models/`. When embedding the library, provide your model directory path at runtime.

## CLI Reference

`wf` — GPU-accelerated Whisper transcription from the command line.

```
wf <COMMAND>

Commands:
  transcribe   Transcribe an audio file to text, SRT, or JSON
  convert      Convert a HuggingFace Whisper safetensors model to Burn `.mpk` format

wf transcribe [OPTIONS]
  -a, --audio-file <FILE>          Input audio file (WAV, MP3, FLAC, OGG, M4A)
  -m, --model <MODEL>              Model name under models/ [default: tiny_en_converted]
  -l, --language <LANG>            Language code [default: en]
  -o, --output <FILE>              Write output to file
      --output-format <FMT>        text | srt | json [default: text]
      --decoding-preset <PRESET>   fast | balanced | accurate [default: balanced]
      --beam-size <N>              Override beam size
      --temperature <F>            Override sampling temperature
      --length-penalty <F>         Override length penalty
      --no-speech-threshold <F>    No-speech detection threshold
      --task <TASK>                transcribe | translate [default: transcribe]
      --vad-enabled                Enable voice activity detection
      --vad-threshold <F>          VAD detection threshold (0.0–1.0) [default: 0.5]
      --cpu                        Use the Flex CPU backend instead of the default WGPU
      --diarize                    Enable speaker diarization
      --diarize-threshold <F>      Cosine similarity threshold [default: 0.7]
      --encoder-batch-size <N>     Encoder forward-pass batch size

wf convert [OPTIONS]
      --model-id <ID>              HuggingFace model ID [default: openai/whisper-tiny.en]
      --output <PATH>              Output path without extension (required)
      --local-safetensors <PATH>   Load from local safetensors file instead of downloading
      --quantize <MODE>            none | int8 [default: none]
```

**Available models:**
- `tiny.en` — English-only, 39M parameters
- `base`, `small`, `medium`, `large-v2`, `large-v3` — Multilingual

All models are converted from OpenAI's [HuggingFace releases](https://huggingface.co/openai) via `wf convert`.

## Architecture

Four-crate workspace built on **Burn 0.21** (Rust 2024 edition, requires **Rust 1.85+**). All crates published on [crates.io](https://crates.io/):

| Crate | Role |
|-------|------|
| [`whisperforge-core`](https://crates.io/crates/whisperforge-core) | Whisper model, audio pipeline, KV-cached decoding |
| [`whisperforge`](https://crates.io/crates/whisperforge) | `wf` binary; `transcribe` + `convert` subcommands |
| [`whisperforge-align`](https://crates.io/crates/whisperforge-align) | VAD, segmentation, `BatchedTranscriber`, SRT output |
| [`whisperforge-diarize`](https://crates.io/crates/whisperforge-diarize) | Speaker embedding clustering, `SPEAKER_NN` label assignment |

## Tech Stack

- **[Burn 0.21](https://burn.dev/)** — GPU-accelerated ML inference framework
  - burn-flex backend: CPU (NdArray/Std) with automatic GPU dispatch
  - WGPU backend for Vulkan/DX12/Metal GPU acceleration
- **[rubato](https://github.com/HeroicKatora/rubato)** — high-quality audio resampling
- **[symphonia](https://github.com/pdeljanov/symphonia)** — multi-format audio decoding
- **[tokenizers](https://huggingface.co/docs/tokenizers/)** — BPE tokenization
- **[earshot](https://github.com/tazz4843/earshot-rs)** — voice activity detection
- **[flate2](https://github.com/rust-lang/flate2-rs)** — compression utilities
- **[hf-hub](https://github.com/huggingface/hf-hub)** — HuggingFace model hub integration

## Resources

- 📚 [Whisper Paper](https://arxiv.org/abs/2212.04356) — OpenAI's original research
- 🔗 [HuggingFace Models](https://huggingface.co/openai) — Pre-trained model weights
- 📘 [Burn Documentation](https://burn.dev/) — ML framework reference
- 🐛 [Issues](https://github.com/bevsxyz/WhisperForge/issues) — Bug reports & feature requests

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Made with ❤️ by [bevsxyz](https://github.com/bevsxyz)
