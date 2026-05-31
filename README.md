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
| Single `wforge` CLI: `transcribe` / `pull` / `list` / `stream` | ✅ Complete |
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
let model = load_whisper::<Flex<f32>>("models/tiny.en/model", &device)?;
let tokenizer = Tokenizer::from_file("models/tiny.en/tokenizer.json")?;
let transcriber = WhisperTranscriber::new(model, tokenizer, DecodingConfig::default());

let audio = load_audio_file("speech.wav")?;
let result = transcribe_audio(&transcriber, &audio, &device)?;
println!("{}", result.text);
```

## Quick Start

### Using the CLI

After [installing](#installation), grab a model and transcribe:

```bash
# 1. Pull a model — a friendly alias maps to openai/whisper-<alias> and auto-names the dir
wforge pull tiny.en

# 2. Transcribe (audio is positional; auto-selects WGPU when compiled in)
wforge transcribe audio.wav -m tiny.en

# Browse models, audio devices, and compute backends (bare `list` shows all three)
wforge list models

# SRT with speaker labels
wforge transcribe audio.wav -m tiny.en --format srt --diarize -o output.srt

# JSON output
wforge transcribe audio.wav --format json

# Native CUDA (CubeCL) — requires CUDA toolkit at build time
cargo install whisperforge --features cuda
wforge transcribe audio.wav -m tiny.en --device cuda
```

> Run `wforge transcribe audio.wav` with no `-m` on a terminal and you'll be prompted to
> pick from your installed models. `pull` also accepts a raw HF id (`org/my-model`) or a
> local path to a fine-tuned `.safetensors` file.

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
wforge stream --model tiny.en

# Long continuous monologue — fewer trim seams (28 s window)
wforge stream --model tiny.en --preset dictation

# From file — no microphone needed, runs faster than real time
wforge stream --model tiny.en \
  --from-file audio.wav --no-realtime --json

# Record the captured audio while streaming
wforge stream --model tiny.en --record-to /tmp/session.wav

# Append each committed sentence to a transcript file
wforge stream --model tiny.en --transcript-to /tmp/transcript.txt

# List available input devices
wforge list devices
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model <name>` | picker / `tiny.en` | Converted model name (omit on a TTY to pick interactively) |
| `--preset <conversation\|dictation>` | `conversation` | Window/endpointing bundle; individual flags override |
| `--device <auto\|cpu\|wgpu\|cuda>` | `auto` | Backend selection |
| `--input-device <name>` | system default | Named cpal input device (see `wforge list devices`) |
| `--max-window-secs <f32>` | preset (5.0) | Hard cap on the growing decode buffer; cap-hit triggers a stride-based trim |
| `--stride-secs <f32>` | preset (1.0) | Stride between successive windows |
| `--vad-threshold <f32>` | `0.5` | Silero VAD speech probability threshold |
| `--silence-secs <f32>` | preset (2.0) | Hard end-of-utterance silence |
| `--punct-silence-secs <f32>` | preset (0.8) | Soft EOU after terminal punctuation |
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

Model weights are git-ignored. Fetch/convert with the built-in `wforge pull` subcommand:

```bash
# Friendly alias → openai/whisper-tiny.en, stored as models/tiny.en/
wforge pull tiny.en

# Quantized: INT8 compression (37 MB vs 150 MB for tiny.en)
wforge pull tiny.en --name tiny.en-int8 --quantize int8

# Raw HuggingFace id (community / fine-tuned repos)
wforge pull org/my-whisper-model

# Local fine-tuned model — a .safetensors file (or a dir with model.safetensors + tokenizer.json)
wforge pull /path/to/model.safetensors --name my-finetuned
```

Models land under the resolved models directory: `--models-dir` → `$WF_MODELS_DIR` → `./models/` if present → platform cache dir (`~/.cache/whisperforge/models`).

**`pull` options:**

| Arg/Flag | Default | Description |
|------|---------|-------------|
| `<MODEL>` | required | Friendly alias, raw HF id (`org/model`), or a local `.safetensors` path / dir |
| `--name` | derived from MODEL | Override the directory name the model is stored under |
| `--quantize` | `none` | Quantization: `none` (FP32) or `int8` (~4× compression) |

**Generated files** — each model is a self-contained directory under `models/`:
```
models/
└── <model_name>/
    ├── model.mpk           # Burn model weights
    ├── model.cfg           # Metadata (precision, config)
    └── tokenizer.json      # BPE tokenizer (per-model — multilingual vs .en differ)
```

Both the CLI and library load models from `<models-dir>/<model_name>/`. The library `load_whisper` takes the `model` stem (e.g. `models/tiny.en/model`); the tokenizer lives alongside it.

## CLI Reference

`wforge` — GPU-accelerated Whisper transcription from the command line.

```
wforge [--models-dir <PATH>] <COMMAND>   # --models-dir is global (or set WF_MODELS_DIR)

Commands:
  transcribe   Transcribe an audio file to text, SRT, VTT, or JSON
  pull         Download (or import) a Whisper model and convert it to Burn `.mpk` (alias: convert)
  list         List models, audio devices, or compute backends
  stream       Realtime streaming transcription from microphone or file
  completions  Generate a shell completion script

wforge transcribe <AUDIO> [OPTIONS]
  <AUDIO>                          Input audio file (WAV, MP3, FLAC, OGG, M4A)
  -m, --model <MODEL>              Model name (omit on a TTY to pick interactively; else tiny.en)
  -l, --language <LANG>            Language code, or `auto` [default: en]
      --task <TASK>                transcribe | translate (X→English) [default: transcribe]
  -o, --output <FILE>              Write output to file
  -f, --format <FMT>               text | srt | vtt | json [default: text]   (srt/vtt/json carry timestamps)
      --preset <PRESET>            fast | balanced | accurate [default: balanced]
      --device <DEVICE>            auto | cpu | wgpu | cuda [default: auto]
  (see `--help` for grouped Advanced decoding / VAD / Diarization / Tuning flags)

wforge pull <MODEL> [OPTIONS]
  <MODEL>                          Alias (tiny.en), HF id (org/model), or local .safetensors path/dir
      --name <NAME>                Override the stored directory name
      --quantize <MODE>            none | int8 [default: none]

wforge list [models|devices|backends] [--json]   # default: all

wforge completions <bash|zsh|fish|powershell|elvish>   # e.g. `wforge completions zsh > _wforge`

wforge stream [OPTIONS]
  -m, --model <MODEL>              Model name (omit on a TTY to pick interactively)
      --preset <PRESET>            conversation | dictation [default: conversation]
      --device <DEVICE>            auto | cpu | wgpu | cuda [default: auto]
      --input-device <NAME>        Named cpal input device (see `wforge list devices`)
      --from-file <PATH>           Feed a 16 kHz mono WAV instead of microphone
      --no-realtime                Feed file at max speed (offline mode)
      --json                       NDJSON output to stdout
  (see `--help` for grouped Window/Endpointing/Quality-gate/Output flags)
```

**Available models:** any Whisper checkpoint on HuggingFace. Common aliases:
- `tiny.en`, `base.en`, `small.en`, `medium.en` — English-only
- `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo` — Multilingual

All aliases resolve to OpenAI's [HuggingFace releases](https://huggingface.co/openai) via `wforge pull`.

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
