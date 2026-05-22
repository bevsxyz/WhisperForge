# whisperforge-diarize

Speaker diarization for speech transcription via embedding clustering.

## Quick Links

- **Full Documentation**: [WhisperForge Repository](https://github.com/bevsxyz/WhisperForge)
- **Architecture**: [Overview](https://github.com/bevsxyz/WhisperForge#architecture)

## Features

- Speaker embedding extraction
- Cosine similarity clustering
- `SPEAKER_NN` label assignment
- Configurable similarity threshold
- Works with SRT and JSON output

## Usage

```rust
use whisperforge_diarize::DiarizationConfig;
use whisperforge_core::{Model, WhisperConfig};

let config = WhisperConfig::tiny_en();
let model = Model::load(Path::new("models/tiny_en_converted"))?;

// With CLI: use --diarize flag
// wf -a audio.wav -m tiny_en_converted --diarize
```

## CLI Integration

The CLI automatically applies diarization labels when using the `--diarize` flag:

```bash
wf -a audio.wav -m tiny_en_converted --diarize --output-format srt -o output.srt
```

Output includes speaker labels:
```
1
00:00:00,000 --> 00:00:05,000
SPEAKER_0: Hello, how are you?

2
00:00:05,000 --> 00:00:10,000
SPEAKER_1: I'm doing great, thanks for asking.
```

## See Also

- [`whisperforge-core`](https://crates.io/crates/whisperforge-core) — Library
- [`whisperforge`](https://crates.io/crates/whisperforge) — CLI binary (`wf`)
- [`whisperforge-convert`](https://crates.io/crates/whisperforge-convert) — Model converter
- [`whisperforge-align`](https://crates.io/crates/whisperforge-align) — VAD & SRT

For full documentation, visit the [WhisperForge repository](https://github.com/bevsxyz/WhisperForge).
