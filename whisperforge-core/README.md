# whisperforge-core

GPU-accelerated Whisper model inference with streaming audio, quantization, and KV-cached decoding.

## Quick Links

- **Full Documentation**: [WhisperForge Repository](https://github.com/bevsxyz/WhisperForge)
- **Installation**: See [Installation Guide](https://github.com/bevsxyz/WhisperForge#installation)
- **Examples**: [Library Usage](https://github.com/bevsxyz/WhisperForge#quick-start)

## Features

- All Whisper model sizes (tiny.en through large-v2/v3)
- GPU acceleration via WGPU (Vulkan/DX12/Metal)
- burn-flex backend: CPU + automatic GPU dispatch
- INT8 quantization (~4× compression)
- Streaming audio pipeline with resampling
- KV-cache O(n) decoder
- Per-token timestamps via cross-attention

## Usage

```rust
use whisperforge_core::{Model, WhisperConfig};
use std::path::Path;

let config = WhisperConfig::tiny_en();
let model = Model::load(Path::new("models/tiny_en_converted/model"))?;
let transcript = model.transcribe(audio_samples, sample_rate)?;
println!("{}", transcript);
```

## See Also

- [`whisperforge`](https://crates.io/crates/whisperforge) — Command-line binary (`wforge`) including `wforge convert` for HuggingFace model conversion
- [`whisperforge-align`](https://crates.io/crates/whisperforge-align) — VAD & SRT output
- [`whisperforge-diarize`](https://crates.io/crates/whisperforge-diarize) — Speaker diarization

For full documentation, visit the [WhisperForge repository](https://github.com/bevsxyz/WhisperForge).
