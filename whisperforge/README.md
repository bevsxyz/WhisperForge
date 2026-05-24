# whisperforge

Fast GPU-accelerated speech-to-text CLI with streaming, quantization, speaker diarization, and multilingual support.

## Quick Links

- **Full Documentation**: [WhisperForge Repository](https://github.com/bevsxyz/WhisperForge)
- **Installation**: `cargo install whisperforge`
- **Usage Guide**: [CLI Reference](https://github.com/bevsxyz/WhisperForge#cli-reference)

## Quick Start

```bash
# Transcribe audio (auto-selects WGPU when compiled in, otherwise CPU)
wforge transcribe -a audio.wav -m tiny_en_converted

# Force CPU backend
wforge transcribe -a audio.wav -m tiny_en_converted --device cpu

# SRT output with speaker diarization
wforge transcribe -a audio.wav -m tiny_en_converted --output-format srt --diarize -o output.srt

# Convert a HuggingFace model and list local models
wforge convert --model-id openai/whisper-tiny.en --output models/tiny_en_converted
wforge list-models
```

## Options

- `-a, --audio-file` — Input audio (WAV, MP3, FLAC, OGG, M4A)
- `-m, --model` — Model name [default: tiny_en_converted]
- `--output-format` — text | srt | json [default: text]
- `--device` — auto | cpu | wgpu | cuda [default: auto]
- `--diarize` — Enable speaker diarization
- `--decoding-preset` — fast | balanced | accurate

For complete options and usage examples, see the [full documentation](https://github.com/bevsxyz/WhisperForge#cli-reference).

## See Also

- [`whisperforge-core`](https://crates.io/crates/whisperforge-core) — Library
- [`whisperforge-align`](https://crates.io/crates/whisperforge-align) — VAD & SRT
- [`whisperforge-diarize`](https://crates.io/crates/whisperforge-diarize) — Speaker diarization
