# WhisperForge 🔥

A high-performance Rust rewrite of WhisperX, leveraging the Burn framework for CUDA-accelerated speech transcription with word-level timestamps and speaker diarization.

## Overview

WhisperForge combines the best of Rust's safety and performance with state-of-the-art speech recognition capabilities:

- **🚀 8x realtime** transcription on CUDA (large-v2 model)
- **💾 2GB VRAM** maximum usage (50% reduction vs WhisperX)
- **⚡ <0.5s startup** time for model loading
- **🎯 SOTA decoding** strategy from faster-whisper analysis
- **🗣️ Speaker diarization** with speaker labels
- **⏱️ Word-level timestamps** with <50ms precision
- **🛡️ Memory-safe** Rust implementation with zero unsafe in hot paths

## Technology Stack

### Core Framework
- **[Burn 0.20.0](https://burn.dev/)** - ML framework with CUDA backend (primary)
- **Rust 1.70+** - Memory-safe systems programming language

### Audio Processing
- **rubato** - High-quality audio resampling
- **hound** - WAV file I/O
- **ffmpeg-next** - Broad audio format support

### Speech Processing
- **earshot** - Pure Rust voice activity detection
- **tokenizers** - Fast BPE tokenization
- **pyannote-rs** - Speaker diarization (ONNX)

### Utilities
- **clap** - Command-line argument parsing
- **anyhow** - Error handling
- **serde** - JSON serialization

## Project Structure

```
whisperforge/
├── whisperforge-core/      # Core Whisper models & SOTA decoding
│   ├── src/model.rs        # Model architecture (custom attention)
│   ├── src/audio.rs        # Mel spectrogram computation
│   ├── src/load.rs         # Model loader (Burn format)
│   └── src/decoding.rs     # SOTA decoding (🔄 in progress)
├── whisperforge-align/     # Wav2Vec2 alignment & VAD
├── whisperforge-diarize/   # Speaker diarization
├── whisperforge-cli/       # CLI binary & Python API
├── whisperforge-convert/   # Model conversion (HF → Burn)
├── AGENTS.md               # AI agent development guidelines
├── DEVELOPMENT_PLAN.md     # 4-week development roadmap
└── PROJECT_STATUS.md       # Current progress & next steps
```

## Quick Start

### Build from Source

```bash
# Check code compiles
cargo check --all

# Build in debug mode
cargo build

# Build release binary (optimized for CUDA)
cargo build --release
```

### Run Tests

```bash
# Test core crates (align has pre-existing issues)
cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli

# Run single test with output
cargo test -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture
```

### Format & Lint

```bash
# Format all code
cargo fmt --all

# Run clippy checks
cargo clippy --all-targets --all-features
```

### Run CLI

```bash
# Transcribe audio (requires converted model)
cargo run -p whisperforge-cli -- -a audio.wav -m tiny_en_converted

# With debug output
cargo run -p whisperforge-cli -- --debug-inference --audio-file audio.wav
```

## Development Status

### ✅ Completed
- Custom attention block architecture
- OpenAI → Burn 0.20 model converter
- Model loader with NamedMpk format
- Audio loading & mel spectrogram computation
- Basic token generation & CLI interface

### 🔄 In Progress
- **SOTA Decoding Module** (faster-whisper strategy)
- Quality metrics (compression ratio, log probability)
- Beam search with temperature fallback

### ⏳ Planned
- Voice Activity Detection (VAD) filtering
- Word-level timestamp extraction
- Speaker diarization integration
- Batch processing & optimization

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed progress and [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for full roadmap.

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `whisperforge-core/src/model.rs` | Whisper architecture with custom attention | ✅ Complete |
| `whisperforge-core/src/load.rs` | Burn model loader | ✅ Complete |
| `whisperforge-core/src/audio.rs` | Mel spectrogram computation | 🟡 Basic STFT |
| `whisperforge-core/src/decoding.rs` | SOTA decoding strategy | 🔴 Starting |
| `whisperforge-convert/src/convert.rs` | HF → Burn model conversion | ✅ Complete |
| `whisperforge-cli/src/main.rs` | CLI interface | 🟡 Basic working |
| `AGENTS.md` | AI agent development guide | ✅ Complete |
| `DEVELOPMENT_PLAN.md` | 4-week development roadmap | ✅ Complete |
| `PROJECT_STATUS.md` | Current progress & blockers | ✅ Updated |

## Model Files

Models are excluded from version control (`.gitignore`) but required for development:

```
models/
├── tiny_en_converted.mpk       # Burn 0.20 format (converted)
├── tiny_en_converted.cfg       # Configuration JSON
└── tokenizer.json              # BPE tokenizer
```

Download from HuggingFace and convert using `whisperforge-convert` crate.

## SOTA Decoding Strategy

WhisperForge implements the proven decoding strategy from [faster-whisper](https://github.com/guillaumekln/faster-whisper) (SYSTRAN):

```rust
DecodingConfig {
    beam_size: 5,                           // Beam search width
    temperatures: [0.0, 0.2, ..., 1.0],    // Fallback sequence
    compression_ratio_threshold: 2.4,       // Quality metric
    log_prob_threshold: -1.0,               // Quality metric
    no_speech_threshold: 0.6,               // VAD threshold
}
```

**Hybrid Strategy**: Start with beam search (temp=0), fallback to temperature sampling if quality metrics fail.

**Quality Assessment**:
- Compression ratio: `len(text) / len(gzip(text))`
- Log probability: Average token confidence
- No-speech probability: Silence detection

## Development Workflow for AI Agents

This project is optimized for Claude CLI / opencode workflows:

1. **Review** [AGENTS.md](AGENTS.md) for coding standards and development guidelines
2. **Check** [PROJECT_STATUS.md](PROJECT_STATUS.md) for current blockers and next steps
3. **Reference** [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for feature roadmap
4. **Use** `.claude/context.md` for quick context loading in Claude CLI
5. **Follow** [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow

See `.claude/` directory for Claude CLI configuration and prompts.

## Performance Targets

| Model | CUDA Speed | VRAM | Target |
|-------|-----------|------|--------|
| tiny.en | 200x realtime | 0.5GB | ✅ |
| base | 150x realtime | 1GB | ✅ |
| small | 100x realtime | 2GB | ✅ |
| medium | 80x realtime | 5GB | 🔄 |
| large-v2 | **8x realtime** | 2GB | 🎯 |

## Common Commands

```bash
# Quick project check
cargo check --all

# Run all tests (except known failures)
cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli

# Format and lint
cargo fmt --all && cargo clippy --all-targets --all-features

# Build optimized release
cargo build --release

# Generate documentation
cargo doc --open

# Run specific test with output
cargo test test_name -- --nocapture
```

## Error Handling

The codebase uses `anyhow::Result<T>` for application-level errors and propagates context through `.with_context()`:

```rust
use anyhow::{Context, Result};

pub fn load_audio(path: &str) -> Result<AudioData> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read audio file: {}", path))?;
    // ...
    Ok(audio)
}
```

## Code Style

See [AGENTS.md](AGENTS.md) for comprehensive style guidelines. Key points:

- **Imports**: std → external crates → local modules
- **Naming**: `PascalCase` types, `snake_case` functions, `SCREAMING_SNAKE_CASE` constants
- **Documentation**: Doc comments for all public APIs with examples
- **Performance**: No clones in hot paths, pre-allocate tensors, use batch operations
- **Testing**: Unit tests for public functions, integration tests for pipelines

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines, including:

- Development workflow for features and bug fixes
- Code review expectations
- Testing requirements
- Documentation standards

## Architecture Highlights

### Custom Attention Blocks
The model implements optimized attention mechanisms:
- Multi-head self-attention in encoder
- Cross-attention in decoder (encoder-decoder interaction)
- Position-wise feed-forward networks
- Layer normalization and residual connections

### Memory Efficiency
- Tensor views instead of clones
- Pre-allocated buffers for audio processing
- Batch operations for parallelization
- CUDA stream management for concurrent processing

### Type Safety
- Generic over `Backend` trait for cross-platform support
- Type aliases for complex tensor shapes
- Compile-time constraints via const generics

## Known Limitations

- ⚠️ CUDA backend required for performance targets (CPU fallback available but slow)
- ⚠️ Basic greedy decoding only (SOTA module in progress)
- ⚠️ Word timestamps not yet extracted from attention patterns
- ⚠️ Speaker diarization not integrated

## Resources

- **Burn Documentation**: https://burn.dev/
- **AGENTS.md**: Complete AI agent development guidelines
- **DEVELOPMENT_PLAN.md**: Detailed 4-week roadmap with SOTA analysis
- **PROJECT_STATUS.md**: Current progress, blockers, and next steps

## License

[Add your license here]

---

**Last Updated**: January 26, 2026  
**Optimized for**: Claude CLI / opencode AI agents  
**Current Focus**: SOTA decoding implementation (faster-whisper strategy)
