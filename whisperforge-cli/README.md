# whisperforge-cli

Fast GPU-accelerated speech-to-text CLI with streaming, quantization, speaker diarization, and multilingual support.

## Quick Links

- **Full Documentation**: [WhisperForge Repository](https://github.com/bevsxyz/WhisperForge)
- **Installation**: `cargo install whisperforge-cli`
- **Usage Guide**: [CLI Reference](https://github.com/bevsxyz/WhisperForge#cli-reference)

## Quick Start

```bash
# Transcribe audio (CPU)
wf -a audio.wav -m tiny_en_converted

# GPU transcription (Vulkan/DX12/Metal)
wf -a audio.wav -m tiny_en_converted --wgpu

# SRT output with speaker diarization
wf -a audio.wav -m tiny_en_converted --output-format srt --diarize -o output.srt

# JSON output
wf -a audio.wav --output-format json
```

## Options

- `-a, --audio-file` — Input audio (WAV, MP3, FLAC, OGG, M4A)
- `-m, --model` — Model name [default: tiny_en_converted]
- `--output-format` — text | srt | json [default: text]
- `--wgpu` — Use GPU backend
- `--diarize` — Enable speaker diarization
- `--decoding-preset` — fast | balanced | accurate

For complete options and usage examples, see the [full documentation](https://github.com/bevsxyz/WhisperForge#cli-reference).

## See Also

- [`whisperforge-core`](https://crates.io/crates/whisperforge-core) — Library
- [`whisperforge-convert`](https://crates.io/crates/whisperforge-convert) — Model converter
- [`whisperforge-align`](https://crates.io/crates/whisperforge-align) — VAD & SRT
- [`whisperforge-diarize`](https://crates.io/crates/whisperforge-diarize) — Speaker diarization
