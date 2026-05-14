# whisperforge-convert

Convert HuggingFace Whisper safetensors to Burn with INT8 quantization support.

## Quick Links

- **Full Documentation**: [WhisperForge Repository](https://github.com/bevsxyz/WhisperForge#model-files)
- **Installation**: `cargo install whisperforge-convert`

## Quick Start

```bash
# Convert from HuggingFace (FP32)
cargo run --release -p whisperforge-convert -- \
  --model-id openai/whisper-tiny.en \
  --output models/tiny_en_converted

# Convert with INT8 quantization (4× compression)
cargo run --release -p whisperforge-convert -- \
  --model-id openai/whisper-tiny.en \
  --output models/tiny_en_quantized \
  --quantize int8

# Convert from local safetensors file
cargo run --release -p whisperforge-convert -- \
  --local-safetensors /path/to/model.safetensors \
  --output models/custom_model
```

## Options

| Flag | Description |
|------|-------------|
| `--model-id` | HuggingFace model identifier [default: openai/whisper-tiny.en] |
| `--output` | Output path without extension (generates `.mpk` + `.cfg`) |
| `--quantize` | Quantization: `none` (FP32) or `int8` |
| `--local-safetensors` | Load from local file instead of downloading |

## Output

Generates two files:
- `.mpk` — Burn model weights
- `.cfg` — Metadata (precision, config)

For full details, see the [Model Files guide](https://github.com/bevsxyz/WhisperForge#model-files).

## See Also

- [`whisperforge-core`](https://crates.io/crates/whisperforge-core) — Library
- [`whisperforge-cli`](https://crates.io/crates/whisperforge-cli) — CLI binary
- [`whisperforge-align`](https://crates.io/crates/whisperforge-align) — VAD & SRT
- [`whisperforge-diarize`](https://crates.io/crates/whisperforge-diarize) — Speaker diarization
