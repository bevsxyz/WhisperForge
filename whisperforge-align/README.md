# whisperforge-align

Voice activity detection, audio segmentation, and SRT subtitle generation for speech transcription.

## Quick Links

- **Full Documentation**: [WhisperForge Repository](https://github.com/bevsxyz/WhisperForge)
- **Architecture**: [Overview](https://github.com/bevsxyz/WhisperForge#architecture)

## Features

- Voice activity detection (VAD) — skip silence segments
- Audio segmentation — break long audio into batches
- Batched transcription — efficient multi-segment processing
- SRT subtitle output — ready for video players
- JSON export — timestamps and metadata

## Usage

```rust
use whisperforge_align::BatchedTranscriber;
use whisperforge_core::{Model, WhisperConfig};

let config = WhisperConfig::tiny_en();
let model = Model::load(Path::new("models/tiny_en_converted/model"))?;
let mut transcriber = BatchedTranscriber::new(model)?;

let segments = transcriber.transcribe_file("audio.wav")?;
for segment in segments {
    println!("[{} - {}] {}", segment.start, segment.end, segment.text);
}
```

## See Also

- [`whisperforge-core`](https://crates.io/crates/whisperforge-core) — Library
- [`whisperforge`](https://crates.io/crates/whisperforge) — CLI binary (`wforge`); `wforge convert` ports HuggingFace safetensors
- [`whisperforge-diarize`](https://crates.io/crates/whisperforge-diarize) — Speaker diarization

For full documentation, visit the [WhisperForge repository](https://github.com/bevsxyz/WhisperForge).
