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
- Single `wforge` binary

## Project Status

✅ **Production Ready** — Core features complete (Phases A–C + E + F). Realtime streaming shipped.

| Component | Status |
|---|---|
| Whisper model loader (all sizes: tiny.en → large-v3) | ✅ Complete |
| Audio pipeline (mel spectrogram, resampling, VAD) | ✅ Complete |
| SOTA HybridDecoder (beam search + temperature fallback) | ✅ Complete |
| KV-cache O(n) decoder | ✅ Complete |
| Per-token timestamps (cross-attention) | ✅ Complete |
| SRT / JSON / text output | ✅ Complete |
| GPU acceleration (WGPU: Vulkan/DX12/Metal) | ✅ Complete |
| Native CUDA backend (CubeCL) | ✅ Complete (feature `cuda`) |
| Single `wforge` CLI: `transcribe` / `convert` / `list-models` | ✅ Complete |
| Speaker diarization (encoder embeddings) | ✅ Complete |
| INT8 quantization (~4× compression) | ✅ Complete |
| Realtime streaming (mic input, token-level output) | ✅ Complete (Phase F) |
| WASM target (browser support) | ⏸ Deferred (Phase D blocker) |

## Installation

### CLI Binary

**[📦 Install from crates.io](https://crates.io/crates/whisperforge)** — One command:
```bash
cargo install whisperforge
wforge --help
```

**[🔨 Build from source](https://github.com/bevsxyz/WhisperForge)** — For development:
```bash
git clone https://github.com/bevsxyz/WhisperForge
cargo install --path ./whisperforge
```

> The canonical binary is `wforge`. Pre-built release zips also ship a small `wf` shim (shell script on Linux/macOS, `wf.cmd` on Windows) for users who prefer the shorter name. `cargo install` only places `wforge` on your `$PATH`; alias `wf` in your shell config if you want both.

### Library: Add to Your Project

For Rust projects, add WhisperForge crates to `Cargo.toml`:

```toml
[dependencies]
whisperforge-core = "0.4"          # Core: Whisper model & audio pipeline
whisperforge-align = "0.4"         # Optional: VAD, batched transcription, SRT
whisperforge-diarize = "0.4"       # Optional: Speaker diarization
burn-flex = "0.21"                 # Required: CPU backend (or use burn-wgpu for GPU)
tokenizers = "0.22"                # Required: BPE tokenizer

[features]
gpu = ["whisperforge-core/cubecl-stft"]  # Optional: GPU via WGPU
```

Basic example (CPU backend):
```rust
use whisperforge_core::{
    DecodingConfig, WhisperTranscriber, audio::load_audio_file,
    load::load_whisper, transcribe::transcribe_audio,
};
use burn_flex::{Flex, FlexDevice};
use tokenizers::Tokenizer;

let device = FlexDevice;
let model = load_whisper::<Flex<f32>>("models/tiny_en_converted", &device)?;
let tokenizer = Tokenizer::from_file("models/tokenizer.json")?;
let transcriber = WhisperTranscriber::new(model, tokenizer, DecodingConfig::default());

let audio = load_audio_file("speech.wav")?;
let result = transcribe_audio(&transcriber, &audio, &device)?;
println!("{}", result.text);
```

## Quick Start

### Using the CLI

After [installing](#installation), grab a model and transcribe:

```bash
# 1. Convert a Whisper model from HuggingFace (writes models/tiny_en.{mpk,cfg} + tokenizer.json)
wforge convert --model-id openai/whisper-tiny.en --output models/tiny_en

# 2. Transcribe (auto-selects WGPU when compiled in; override with --device cpu|wgpu|cuda)
wforge transcribe -a audio.wav -m tiny_en

# Browse converted models (honors --models-dir / WF_MODELS_DIR)
wforge list-models

# SRT with speaker labels
wforge transcribe -a audio.wav -m tiny_en --output-format srt --diarize -o output.srt

# JSON output
wforge transcribe -a audio.wav --output-format json

# Native CUDA (CubeCL) — requires CUDA toolkit at build time
cargo install whisperforge --features cuda
wforge transcribe -a audio.wav -m tiny_en --device cuda
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

## Streaming (Phase F)

`wforge stream` delivers always-on local transcription that emits partial results as you speak. The pipeline is: microphone → Silero VAD (32 ms frames) → growing-buffer chunker (5 s cap / 1 s stride) → per-window Whisper encode + greedy decode → confidence gate (log-prob + compression-ratio, faster-whisper parity) → LocalAgreement-2 committer (tokens stable across two consecutive decodes are permanently emitted) → silence + punctuation endpointer → output sinks.

Long utterances stay on one continuous commit stream via **stride-based buffer trimming on cap-hit**: when the buffer reaches `--max-window-secs`, the chunker drops the oldest 1.5 s, the committer force-commits its tentative tail, and the next stride establishes a fresh LCP baseline. ~30% of mid-utterance content can be dropped at trim seams as a known, accepted trade-off (Whisper's non-causal encoder produces different tokens for the same audio across different buffer sizes). The happy path (dictation/conversation with natural pauses) resets per utterance and never hits a cap, so it's unaffected; for **continuous multi-minute monologue** the supported path is **`--max-window-secs 28`** (no caps → no trim seams). See CLAUDE.md § "Live-mic perf ceiling" for the full breakdown.

Streaming is UAT-certified across all three backends (cpu / cuda / wgpu) via `scripts/uat-stream.ps1`.

```bash
# Transcribe from the default microphone (committed text appears in real time)
wforge stream --model tiny_en_converted

# From file — no microphone needed, runs faster than real time
wforge stream --model tiny_en_converted \
  --from-file audio.wav --no-realtime --json

# Record the captured audio while streaming
wforge stream --model tiny_en_converted --record-to /tmp/session.wav

# Append each committed sentence to a transcript file
wforge stream --model tiny_en_converted --transcript-to /tmp/transcript.txt

# List available input devices
wforge stream --list-input-devices
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model <name>` | required | Converted model name under `models/` |
| `--device <auto\|cpu\|wgpu\|cuda>` | `auto` | Backend selection |
| `--input-device <name>` | system default | Named cpal input device |
| `--list-input-devices` | — | Print host/device list and exit |
| `--max-window-secs <f32>` | `5.0` | Hard cap on the growing decode buffer; cap-hit triggers a stride-based trim |
| `--stride-secs <f32>` | `1.0` | Stride between successive windows |
| `--vad-threshold <f32>` | `0.5` | Silero VAD speech probability threshold |
| `--silence-secs <f32>` | `2.0` | Hard end-of-utterance silence |
| `--punct-silence-secs <f32>` | `0.8` | Soft EOU after terminal punctuation |
| `--prompt-tokens <usize>` | `0` | `<\|prevtext\|>` carry-over tokens (off by default — non-zero can lock the decoder into hallucination loops on quiet audio) |
| `--logprob-threshold <f32>` | `-1.0` | Reject windows whose avg token log-prob falls below this (faster-whisper parity) |
| `--compression-ratio-threshold <f32>` | `2.4` | Reject windows whose gzip compression ratio exceeds this — kills repetition/hallucination loops |
| `--json` | off | NDJSON output to stdout |
| `--record-to <path>` | — | Tee 16 kHz mono WAV |
| `--transcript-to <path>` | — | Append `[mm:ss–mm:ss] text` lines |
| `--from-file <path>` | — | Feed a **16 kHz mono** WAV instead of microphone (not resampled) |
| `--no-realtime` | off | Feed file at max speed (offline mode) |
| `--no-color` | off | Disable ANSI styling |

> **WSL2:** `cpal` requires a PulseAudio bridge on WSL2. Use `--from-file` for all testing on WSL2 until the bridge is set up.

## Model Files

Model weights are git-ignored. Convert from HuggingFace using the built-in `wforge convert` subcommand:

```bash
# Basic: download and convert to Burn format
wforge convert --model-id openai/whisper-tiny.en --output models/tiny_en_converted

# Quantized: INT8 compression (37 MB vs 150 MB for tiny.en)
wforge convert --model-id openai/whisper-tiny.en --output models/tiny_en_quantized --quantize int8

# From local file instead of downloading
wforge convert --local-safetensors /path/to/model.safetensors --output models/tiny_en_converted
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

`wforge` — GPU-accelerated Whisper transcription from the command line.

```
wforge <COMMAND>

Commands:
  transcribe   Transcribe an audio file to text, SRT, or JSON
  convert      Convert a HuggingFace Whisper safetensors model to Burn `.mpk` format
  stream       Realtime streaming transcription from microphone or file

wforge transcribe [OPTIONS]
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
      --models-dir <PATH>          Directory holding `.mpk`/`.cfg` models (or set WF_MODELS_DIR)
      --vad-enabled                Enable voice activity detection
      --vad-threshold <F>          VAD detection threshold (0.0–1.0) [default: 0.5]
      --device <DEVICE>            auto | cpu | wgpu | cuda [default: auto]
      --diarize                    Enable speaker diarization
      --diarize-threshold <F>      Cosine similarity threshold [default: 0.7]
      --encoder-batch-size <N>     Encoder forward-pass batch size

wforge list-models [OPTIONS]
      --models-dir <PATH>          Directory to scan for `.mpk` models (or set WF_MODELS_DIR)

wforge convert [OPTIONS]
      --model-id <ID>              HuggingFace model ID [default: openai/whisper-tiny.en]
      --output <PATH>              Output path without extension (required)
      --local-safetensors <PATH>   Load from local safetensors file instead of downloading
      --quantize <MODE>            none | int8 [default: none]

wforge stream [OPTIONS]
  -m, --model <MODEL>              Model name under models/ (required)
      --models-dir <PATH>          Directory holding `.mpk`/`.cfg` models (or set WF_MODELS_DIR)
      --device <DEVICE>            auto | cpu | wgpu | cuda [default: auto]
      --input-device <NAME>        Named cpal input device (default: system default)
      --list-input-devices         Print available input devices and exit
      --max-window-secs <F>        Hard cap on growing decode buffer in seconds [default: 5.0]
      --stride-secs <F>            Stride between windows in seconds [default: 1.0]
      --vad-threshold <F>          Silero VAD threshold [default: 0.5]
      --silence-secs <F>           Hard end-of-utterance silence [default: 2.0]
      --punct-silence-secs <F>     Soft EOU after terminal punctuation [default: 0.8]
      --prompt-tokens <N>          <|prevtext|> carry-over tokens, 0 = disabled [default: 0]
      --json                       NDJSON output to stdout
      --record-to <PATH>           Tee 16 kHz mono WAV to file
      --transcript-to <PATH>       Append committed lines to text file
      --no-color                   Disable ANSI styling
      --from-file <PATH>           Feed a WAV file instead of microphone
      --no-realtime                Feed file at max speed (offline mode)
```

**Available models:**
- `tiny.en` — English-only, 39M parameters
- `base`, `small`, `medium`, `large-v2`, `large-v3` — Multilingual

All models are converted from OpenAI's [HuggingFace releases](https://huggingface.co/openai) via `wforge convert`.

## Architecture

Four-crate workspace built on **Burn 0.21** (Rust 2024 edition, requires **Rust 1.85+**). All crates published on [crates.io](https://crates.io/):

| Crate | Role |
|-------|------|
| [`whisperforge-core`](https://crates.io/crates/whisperforge-core) | Whisper model, audio pipeline, KV-cached decoding |
| [`whisperforge`](https://crates.io/crates/whisperforge) | `wforge` binary; `transcribe` + `convert` subcommands |
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
