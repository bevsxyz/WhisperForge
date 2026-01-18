# WhisperForge Development Plan

## Overview

WhisperForge is a clean Rust rewrite of WhisperX, leveraging Burn as the primary ML framework for CUDA-accelerated inference. This project aims to provide a high-performance, memory-safe alternative for audio transcription with word-level timestamps and speaker diarization.

## Key Constraints & Technology Stack

### Primary Framework: Burn (burn.dev)
- **CUDA Backend**: Primary target for GPU acceleration
- **Fallback**: wgpu/CPU for broader compatibility
- **Performance**: Competitive with NVIDIA cuBLAS (as of 2025 benchmarks)
- **Advantages**: Rust-native, memory-safe, cross-platform

### Core Dependencies

| Category | Crate | Purpose | Status |
|----------|-------|---------|--------|
| **ML Framework** | `burn` | Primary ML framework with CUDA backend | ✅ Production Ready |
| **Audio Processing** | `rubato` | Audio resampling (16kHz mono) | ✅ Stable v1.0 |
| **Audio I/O** | `hound` | WAV file reading/writing | ✅ Stable |
| **Audio I/O** | `ffmpeg-next` | Broad format support | ✅ Stable |
| **VAD** | `earshot` | Pure Rust VAD (fast) | ✅ MIT License |
| **VAD** | `silero-vad-rust` | Silero VAD ONNX models | ✅ Stable |
| **Diarization** | `pyannote-rs` | Speaker diarization (ONNX) | ✅ MIT License |
| **Diarization** | `sherpa-rs` | Alternative diarization | ✅ Apache-2.0 |
| **Whisper** | `whisper-rs` | whisper.cpp bindings (fallback) | ✅ Unlicense |
| **CLI** | `clap` | Command-line interface | ✅ Stable |
| **Error Handling** | `anyhow` | Error propagation | ✅ Stable |
| **Serialization** | `serde` | JSON/SRT output | ✅ Stable |

## Architecture Plan

### Crate Structure

```
whisperforge/
├── whisperforge-core/     # Core Whisper models (Burn)
├── whisperforge-align/    # Wav2Vec2 alignment & VAD
├── whisperforge-diarize/  # Speaker diarization
├── whisperforge-cli/      # CLI binary & Python API
└── whisperforge-convert/  # Model conversion tools
```

### Detailed Crate Breakdown

| Crate | Purpose | Key Dependencies | Public API |
|-------|---------|------------------|------------|
| **whisperforge-core** | Whisper transcription using Burn | `burn[cuda]`, `tokenizers`, `safetensors` | `WhisperModel`, `TranscriptionResult` |
| **whisperforge-align** | Word-level alignment & VAD | `burn`, `earshot`, `rubato` | `Aligner`, `VadDetector` |
| **whisperforge-diarize** | Speaker diarization | `pyannote-rs`, `sherpa-rs` | `Diarizer`, `SpeakerSegment` |
| **whisperforge-cli** | CLI interface & library API | `clap`, `anyhow`, `serde` | `main()`, `transcribe()` |
| **whisperforge-convert** | HF→Burn model conversion | `candle-core`, `safetensors` | `convert_whisper_model()` |

## Phase-Based Development Plan

### Phase 1: Base Whisper Transcription (Weeks 1-2)

#### Week 1: Burn Whisper Integration
**Goal**: Basic Whisper transcription using Burn

**Tasks**:
- [ ] Set up workspace with multiple crates
- [ ] Implement Whisper model architecture in Burn
- [ ] Create model conversion script (HF→Burn format)
- [ ] Load tiny.en model and test basic transcription
- [ ] Implement basic audio loading (16kHz mono)

**Deliverables**:
- `whisperforge-core` with tiny.en model support
- Model conversion script for all Whisper variants
- Basic CLI: `whisperforge audio.wav --model tiny.en`

**Technical Details**:
```rust
// Core API design
pub struct WhisperModel<B: Backend> {
    model: WhisperEncoder<B>,
    decoder: WhisperDecoder<B>,
}

impl<B: Backend> WhisperModel<B> {
    pub fn new(model_path: &str, device: B::Device) -> Result<Self>;
    pub fn transcribe(&self, audio: Tensor<B, 2>) -> Result<String>;
}
```

#### Week 2: Batching & VAD Integration
**Goal**: Batched inference with VAD preprocessing

**Tasks**:
- [ ] Integrate VAD using `earshot` crate
- [ ] Implement audio segmentation based on VAD
- [ ] Add batched inference support
- [ ] Implement basic SRT output format
- [ ] Add CUDA backend optimization

**Deliverables**:
- VAD-based audio segmentation
- Batched inference (70x realtime target)
- SRT output format support
- CUDA optimization for large-v2 model

**Performance Targets**:
- **tiny.en**: 200x realtime on CUDA
- **base**: 150x realtime on CUDA
- **large-v2**: 70x realtime on CUDA (8GB VRAM)

### Phase 2: Word-Level Alignment (Week 3)

#### Week 3: Wav2Vec2 Forced Alignment
**Goal**: Precise word-level timestamps

**Tasks**:
- [ ] Research Wav2Vec2 model support in Burn
- [ ] Implement phoneme-based alignment
- [ ] Add language-specific alignment models (en/fr/de/es/it)
- [ ] Integrate CTC alignment algorithm
- [ ] Add word/character timestamp generation

**Technical Approach**:
```rust
pub struct Aligner<B: Backend> {
    wav2vec2: Wav2Vec2Model<B>,
    phoneme_map: HashMap<String, Vec<Phoneme>>,
}

impl<B: Backend> Aligner<B> {
    pub fn align(&self, audio: Tensor<B, 2>, transcript: &str) -> Result<AlignmentResult>;
    pub fn align_words(&self, segments: &[TranscriptionSegment]) -> Result<Vec<WordSegment>>;
}
```

**Supported Languages**:
- English (primary)
- French, German, Spanish, Italian
- Extensible for additional languages

### Phase 3: Diarization & Polish (Week 4)

#### Week 4: Speaker Diarization
**Goal**: Multi-speaker identification and assignment

**Tasks**:
- [ ] Integrate `pyannote-rs` for speaker diarization
- [ ] Implement speaker embedding extraction
- [ ] Add speaker clustering algorithms
- [ ] Assign speakers to word segments
- [ ] Add multispeaker SRT output

**Technical Implementation**:
```rust
pub struct Diarizer {
    segmentation_model: SegmentationModel,
    embedding_model: SpeakerEmbeddingModel,
}

impl Diarizer {
    pub fn diarize(&self, audio: &AudioData) -> Result<Vec<SpeakerSegment>>;
    pub fn assign_speakers(&self, words: &[WordSegment]) -> Result<Vec<WordSegment>>;
}
```

**Performance Targets**:
- **1 hour audio**: <1 minute processing time
- **Accuracy**: Comparable to pyannote-audio
- **Memory**: <2GB additional RAM usage

## Advanced Features (Post-MVP)

### Quantization Support
- **Int8 Quantization**: Using Burn's quantization features
- **Memory Reduction**: Target 50% VRAM reduction
- **Performance**: Minimal accuracy loss (<2%)

### WASM Export
- **Browser Support**: Compile to WebAssembly
- **Edge Deployment**: Client-side transcription
- **API**: Same interface as native version

### Python Bindings
- **PyO3 Integration**: Python API for compatibility
- **Drop-in Replacement**: Compatible with whisperx API
- **Performance**: 10x faster than original WhisperX

## Model Management Strategy

### Model Conversion Pipeline
```bash
# Convert HuggingFace model to Burn format
whisperforge-convert \
  --source openai/whisper-large-v2 \
  --target models/whisper-large-v2.burn \
  --backend cuda
```

### Model Caching
- **Local Cache**: `~/.cache/whisperforge/`
- **Auto-download**: Models from HuggingFace Hub
- **Versioning**: Model hash verification
- **Compression**: Optional model compression

### Supported Models
| Model | Size | VRAM (CUDA) | VRAM (CPU) | Speed |
|-------|------|-------------|------------|-------|
| tiny.en | 39MB | 0.5GB | 1GB | 200x |
| base | 74MB | 1GB | 2GB | 150x |
| small | 244MB | 2GB | 4GB | 100x |
| medium | 769MB | 5GB | 8GB | 80x |
| large-v2 | 1550MB | 8GB | 12GB | 70x |
| large-v3 | 1550MB | 8GB | 12GB | 70x |

## Audio Processing Pipeline

### Input Processing
```rust
pub struct AudioProcessor {
    sample_rate: u32,
    channels: u16,
    vad: VoiceActivityDetector,
}

impl AudioProcessor {
    pub fn load_audio(&mut self, path: &str) -> Result<AudioData>;
    pub fn preprocess(&mut self, audio: AudioData) -> Result<Vec<AudioSegment>>;
    pub fn resample(&self, audio: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>>;
}
```

### Pipeline Stages
1. **Loading**: FFmpeg-based audio loading
2. **Preprocessing**: 16kHz mono conversion
3. **VAD**: Voice activity detection
4. **Segmentation**: Chunk-based audio splitting
5. **Transcription**: Batched Whisper inference
6. **Alignment**: Word-level timestamp generation
7. **Diarization**: Speaker identification
8. **Output**: SRT/JSON formatting

## CLI Interface Design

### Basic Usage
```bash
# Simple transcription
whisperforge audio.wav --model large-v2 --output transcript.srt

# Advanced usage
whisperforge video.mp4 \
  --model large-v2 \
  --compute cuda \
  --batch 16 \
  --language auto \
  --align true \
  --diarize true \
  --output-format srt \
  --output-dir ./transcripts \
  --speaker-labels "Speaker A,Speaker B,Speaker C"
```

### Configuration File
```toml
# ~/.config/whisperforge/config.toml
[default]
model = "large-v2"
compute = "cuda"
batch_size = 16
language = "auto"

[alignment]
enabled = true
model = "wav2vec2-base"

[diarization]
enabled = true
model = "pyannote-segmentation"
min_speakers = 1
max_speakers = 10
```

## Performance Benchmarks

### Target Performance (vs WhisperX)

| Metric | WhisperX (Python) | WhisperForge (Rust) | Improvement |
|--------|-------------------|---------------------|-------------|
| **Speed** | 1x realtime | 70x realtime | 70x faster |
| **Memory** | 4GB VRAM | 2GB VRAM | 50% reduction |
| **Accuracy** | Baseline | Baseline | Equal |
| **Startup** | 5s | 0.5s | 10x faster |
| **CPU Usage** | High | Low | 60% reduction |

### Benchmark Suite
```bash
# Run comprehensive benchmarks
whisperforge-benchmark \
  --test-set ted-lium \
  --models tiny.en,base,small,medium,large-v2 \
  --compute cuda \
  --metrics wer,cer,timestamp_accuracy,speed
```

## Testing Strategy

### Unit Tests
- **Model Loading**: Verify all Whisper models load correctly
- **Audio Processing**: Test resampling and VAD
- **Alignment**: Validate word-level timestamps
- **Diarization**: Test speaker identification

### Integration Tests
- **End-to-End**: Full pipeline testing
- **Format Support**: Various audio/video formats
- **Language Support**: Multi-language transcription
- **Performance**: Speed and memory benchmarks

### Test Datasets
- **TED-LIUM**: WER and timestamp accuracy
- **AISHELL**: Chinese language support
- **Common Voice**: Multi-language evaluation
- **Custom**: Domain-specific testing

## Deployment & Distribution

### Binary Distribution
```bash
# Install pre-compiled binary
curl -sSL https://github.com/whisperforge/whisperforge/install.sh | bash

# Or via cargo
cargo install whisperforge-cli
```

### Docker Images
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y ffmpeg
COPY whisperforge /usr/local/bin/
ENTRYPOINT ["whisperforge"]
```

### Python Package
```python
# Drop-in replacement for whisperx
import whisperforge

# Transcribe with alignment and diarization
result = whisperforge.transcribe(
    "audio.wav",
    model="large-v2",
    align=True,
    diarize=True
)
```

## Risk Assessment & Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Burn Wav2Vec2 Support** | Medium | High | Use Candle as fallback, implement custom model |
| **CUDA Performance** | Low | High | Benchmark early, optimize kernels |
| **Model Conversion** | Medium | Medium | Develop robust conversion tools |
| **Memory Usage** | Low | Medium | Implement streaming, quantization |

### Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Development Time** | Medium | Medium | Focus on MVP, use existing crates |
| **Community Adoption** | Low | Medium | Python compatibility, clear docs |
| **Maintenance** | Low | Medium | Automated testing, CI/CD |

## Success Metrics

### Performance Metrics
- **70x realtime** transcription on CUDA (large-v2)
- **<8GB VRAM** usage for large-v2 model
- **<1s startup** time
- **<2% WER** degradation vs WhisperX

### Adoption Metrics
- **1000+ GitHub stars** within 6 months
- **10,000+ downloads** on crates.io
- **Python package** with 1000+ installs
- **Community contributions** from 20+ developers

### Quality Metrics
- **95%+ test coverage**
- **Zero security vulnerabilities**
- **Comprehensive documentation**
- **Multi-platform support** (Linux, macOS, Windows)

## Conclusion

WhisperForge represents a significant advancement in ASR technology, combining:
- **Rust's safety and performance**
- **Burn's cutting-edge ML framework**
- **Comprehensive feature parity with WhisperX**
- **10x performance improvement**

The phased development approach ensures rapid delivery of value while maintaining high quality standards. The modular architecture enables extensibility and community contributions.

With the current maturity of the Rust ecosystem and Burn's competitive performance, WhisperForge is well-positioned to become the preferred choice for high-performance audio transcription.