# WhisperForge Development Plan

## Overview

WhisperForge is a clean Rust rewrite of WhisperX, leveraging Burn as the primary ML framework for CUDA-accelerated inference. This project aims to provide a high-performance, memory-safe alternative for audio transcription with word-level timestamps and speaker diarization.

**🎯 NEW DIRECTION**: Based on analysis of faster-whisper (SYSTRAN), we are implementing their proven SOTA decoding strategy instead of basic hybrid approaches.

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
| **Compression** | `flate2` | For compression ratio metric | � Planned (SOTA decoding) |

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
| **whisperforge-core** | Whisper transcription using Burn | `burn[cuda]`, `tokenizers`, `safetensors`, `flate2` | `WhisperModel`, `DecodingConfig`, `TranscriptionResult` |
| **whisperforge-align** | Word-level alignment & VAD | `burn`, `earshot`, `rubato` | `Aligner`, `VadDetector` |
| **whisperforge-diarize** | Speaker diarization | `pyannote-rs`, `sherpa-rs` | `Diarizer`, `SpeakerSegment` |
| **whisperforge-cli** | CLI interface & library API | `clap`, `anyhow`, `serde` | `main()`, `transcribe()` |
| **whisperforge-convert** | HF→Burn model conversion | `candle-core`, `safetensors` | `convert_whisper_model()` |

## Phase-Based Development Plan

### **🎯 Phase 1: SOTA Decoding Implementation** (Weeks 1-2)

**🔬 Research Complete**: Analyzed faster-whisper source code
- **Strategy**: Hybrid decoding with quality-based fallback
- **Source**: SYSTRAN faster-whisper (production-proven)
- **Performance**: 4x faster than original OpenAI implementation

#### Week 1: Core SOTA Decoding Infrastructure
**Goal**: Implement faster-whisper-style decoding strategy

**Tasks**:
- [ ] Create `whisperforge-core/src/decoding.rs` module
- [ ] Implement `DecodingConfig` with faster-whisper parameters
- [ ] Add quality metrics (compression ratio, log probability)
- [ ] Implement beam search with patience and length penalty
- [ ] Implement temperature sampling with best_of candidates
- [ ] Add `decode_with_fallback()` hybrid switching logic

**Core SOTA Structures**:
```rust
pub struct DecodingConfig {
    pub beam_size: usize = 5,
    pub best_of: usize = 5,
    pub patience: f32 = 1.0,
    pub length_penalty: f32 = 1.0,
    pub repetition_penalty: f32 = 1.0,
    pub no_repeat_ngram_size: usize = 0,
    pub temperatures: Vec<f32> = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    pub compression_ratio_threshold: f32 = 2.4,
    pub log_prob_threshold: f32 = -1.0,
    pub no_speech_threshold: f32 = 0.6,
    pub condition_on_previous_text: bool = true,
    pub prompt_reset_on_temperature: f32 = 0.5,
}

pub enum DecodingPreset {
    Fast { 
        beam_size: 1, 
        temperatures: vec![0.0],
        no_vad_filter: true 
    },
    Balanced { 
        beam_size: 5, 
        temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 
    },
    Accurate { 
        beam_size: 10, 
        temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: 2.2  // Stricter quality
    },
}
```

**Quality Metrics Implementation**:
```rust
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,
}

fn calculate_compression_ratio(text: &str) -> f32 {
    use flate2::write::GzEncoder;
    let text_bytes = text.as_bytes();
    let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(text_bytes).unwrap();
    let compressed = encoder.finish().unwrap();
    text_bytes.len() as f32 / compressed.len() as f32
}
```

**Deliverables**:
- Complete SOTA decoding module with faster-whisper algorithm
- Quality-based automatic fallback switching
- Beam search and sampling implementations
- Unit tests for all quality metrics

#### Week 2: CLI Integration & VAD Support
**Goal**: Integrate SOTA decoding with CLI and add VAD filtering

**Tasks**:
- [ ] Update CLI with decoding preset parameters (fast|balanced|accurate)
- [ ] Add beam size and temperature parameters to CLI
- [ ] Integrate Silero VAD or earshot for speech filtering
- [ ] Implement VAD-based audio chunking
- [ ] Add no-speech probability calculation
- [ ] Update help documentation with SOTA examples

**CLI Integration**:
```rust
#[derive(Parser, Debug)]
struct Args {
    /// Decoding strategy preset (fast|balanced|accurate)
    #[arg(long, default_value = "balanced")]
    decoding_preset: String,
    
    /// Beam size for decoding (only used with temperature=0)
    #[arg(long, default_value = "5")]
    beam_size: usize,
    
    /// Temperature for sampling (can be repeated for fallback)
    #[arg(long, value_delimiter = ',')]
    temperature: Vec<f32>,
    
    /// Disable voice activity detection filtering
    #[arg(long)]
    no_vad_filter: bool,
    
    /// Extract word-level timestamps
    #[arg(long)]
    word_timestamps: bool,
    
    /// Maximum number of new tokens to generate per-chunk
    #[arg(long)]
    max_new_tokens: Option<usize>,
}
```

**Deliverables**:
- CLI with full SOTA decoding parameter support
- VAD filtering integration
- User-configurable decoding presets
- Comprehensive help documentation

### Phase 2: Advanced Features (Week 3)

#### Week 3: Word-Level Alignment & Batch Processing
**Goal**: Add word timestamps and parallel processing

**Tasks**:
- [ ] Implement cross-attention alignment for word timestamps
- [ ] Add dynamic time warping alignment
- [ ] Implement batched inference for multiple audio segments
- [ ] Add word-level timing extraction
- [ ] Support SRT timestamp formatting
- [ ] Use encoder-decoder attention patterns for alignment

**Advanced Features**:
```rust
pub fn extract_word_timestamps(
    attention_weights: &Tensor<B, 4>,
    tokens: &[u32],
    audio_duration: f32,
    model: &WhisperModel<B>,
) -> Result<Vec<WordSegment>> {
    // Cross-attention pattern analysis
    // Dynamic time warping alignment
    // Word boundary detection using attention weights
}

pub fn transcribe_batch(
    model: &WhisperModel<B>,
    audio_segments: Vec<AudioData>,
    config: &DecodingConfig,
) -> Result<Vec<TranscriptionResult>> {
    // Parallel processing with CUDA streams
    // Memory-efficient batch handling
    // VAD-based segment scheduling
}
```

### Phase 3: Diarization & Polish (Week 4)

#### Week 4: Speaker Diarization Integration
**Goal**: Multi-speaker identification and assignment

**Tasks**:
- [ ] Integrate `pyannote-rs` for speaker diarization
- [ ] Implement speaker embedding extraction
- [ ] Add speaker clustering algorithms
- [ ] Assign speakers to word segments
- [ ] Add multi-speaker SRT output
- [ ] Implement speaker label customization

**Diarization Pipeline**:
```rust
pub struct DiarizationConfig {
    pub min_speakers: usize = 1,
    pub max_speakers: usize = 10,
    pub model_type: DiarizationModel,
}

pub enum DiarizationModel {
    PyannoteSegmentation,
    SherpaOffline,
}
```

## Audio Processing Pipeline (Updated with SOTA)

### Input Processing
```rust
pub struct AudioProcessor {
    sample_rate: u32,
    channels: u16,
    vad: VoiceActivityDetector,
    decoding_config: DecodingConfig,
}

impl AudioProcessor {
    pub fn load_audio(&mut self, path: &str) -> Result<AudioData>;
    pub fn preprocess(&mut self, audio: AudioData) -> Result<Vec<AudioSegment>>;
    pub fn resample(&self, audio: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>>;
    pub fn apply_vad_filter(&self, audio: AudioData) -> Result<Vec<SpeechSegment>>;
}
```

### Pipeline Stages (SOTA Enhanced)
1. **Loading**: FFmpeg-based audio loading
2. **Preprocessing**: 16kHz mono conversion
3. **VAD**: Voice activity detection with configurable models
4. **Segmentation**: Chunk-based audio splitting based on VAD
5. **SOTA Transcription**: Hybrid decoding with quality-based fallback
6. **Word Alignment**: Cross-attention timestamp extraction
7. **Diarization**: Speaker identification and assignment
8. **Output**: SRT/JSON formatting with word timestamps

## CLI Interface Design (SOTA Enhanced)

### Basic Usage
```bash
# Simple transcription with balanced preset
whisperforge audio.wav --model tiny_en_converted

# Accurate transcription with word timestamps
whisperforge audio.wav --preset accurate --word-timestamps

# Fast transcription for real-time use
whisperforge audio.wav --preset fast --no-vad-filter
```

### Advanced Usage
```bash
# Full SOTA configuration
whisperforge video.mp4 \
  --model large-v2 \
  --preset balanced \
  --beam-size 5 \
  --temperature 0.0,0.2,0.4,0.6,0.8,1.0 \
  --compression-threshold 2.4 \
  --logprob-threshold -1.0 \
  --word-timestamps \
  --vad-filter \
  --language auto \
  --output-format srt \
  --output-dir ./transcripts \
  --speaker-labels "Speaker A,Speaker B,Speaker C"
```

### Configuration File (SOTA)
```toml
# ~/.config/whisperforge/config.toml
[default]
model = "large-v2"
preset = "balanced"
compute = "cuda"

[decoding]
beam_size = 5
temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
compression_ratio_threshold = 2.4
log_prob_threshold = -1.0
no_speech_threshold = 0.6
condition_on_previous_text = true
prompt_reset_on_temperature = 0.5

[vad]
enabled = true
model = "silero"
min_speech_duration = 0.1
min_silence_duration = 0.16

[alignment]
word_timestamps = false
prepend_punctuations = "\"'\"¿([{-"
append_punctuations = "\"'.。,，!！?？:：\")])}、"
```

## Performance Benchmarks (SOTA Targets)

### Target Performance (vs WhisperX & faster-whisper)

| Metric | WhisperX (Python) | faster-whisper | WhisperForge (Rust) | Target |
|--------|-------------------|---------------|---------------------|--------|
| **Speed** | 1x realtime | 4x realtime | 8x realtime | 8x faster |
| **Memory** | 4GB VRAM | 3GB VRAM | 2GB VRAM | 50% reduction |
| **Accuracy** | Baseline | +2% WER improvement | +4% WER improvement | Better quality |
| **Startup** | 5s | 1s | 0.5s | 10x faster |
| **CPU Usage** | High | Medium | Low | 60% reduction |

### SOTA Decoding Benchmarks
```bash
# Run comprehensive SOTA decoding benchmarks
whisperforge-benchmark \
  --test-set ted-lium \
  --models tiny.en,base,small,medium,large-v2 \
  --presets fast,balanced,accurate \
  --metrics wer,cer,timestamp_accuracy,speed,compression_ratio \
  --compare-with faster-whisper
```

## Testing Strategy (SOTA Enhanced)

### Unit Tests
- **SOTA Decoding**: Test all temperature fallback sequences
- **Quality Metrics**: Validate compression ratio and log prob calculations
- **Beam Search**: Test various beam sizes and patience values
- **VAD Integration**: Test speech filtering accuracy
- **Word Alignment**: Validate timestamp precision

### Integration Tests
- **End-to-End**: Full pipeline with SOTA decoding
- **Format Support**: Various audio/video formats with VAD
- **Language Support**: Multi-language SOTA transcription
- **Performance**: SOTA speed and accuracy benchmarks
- **Quality Comparison**: Compare with faster-whisper on same inputs

### Test Datasets
- **TED-LIUM**: WER and timestamp accuracy for SOTA validation
- **Common Voice**: Multi-language evaluation with SOTA metrics
- **LibriSpeech**: Clean speech benchmarking
- **Custom**: Domain-specific SOTA testing

## Model Management Strategy (Updated)

### Model Conversion Pipeline
```bash
# Convert HuggingFace model to Burn format with SOTA optimizations
whisperforge-convert \
  --source openai/whisper-large-v2 \
  --target models/whisper-large-v2.burn \
  --backend cuda \
  --optimize-for-sota-decoding
```

### Model Caching
- **Local Cache**: `~/.cache/whisperforge/`
- **Auto-download**: Models from HuggingFace Hub with SOTA compatibility info
- **Versioning**: Model hash verification and SOTA decoding notes
- **Compression**: Optional model compression for faster loading

### Supported Models (SOTA Optimized)
| Model | Size | VRAM (CUDA) | VRAM (CPU) | SOTA Speed |
|-------|------|-------------|------------|------------|
| tiny.en | 39MB | 0.5GB | 1GB | 200x realtime |
| base | 74MB | 1GB | 2GB | 150x realtime |
| small | 244MB | 2GB | 4GB | 100x realtime |
| medium | 769MB | 5GB | 8GB | 80x realtime |
| large-v2 | 1550MB | 8GB | 12GB | 70x realtime |
| large-v3 | 1550MB | 8GB | 12GB | 70x realtime |

## SOTA Decoding Research Summary

### Source Analysis: faster-whisper (SYSTRAN)
**File Studied**: `faster-whisper/transcribe.py`
**Key Insights**:
- **Hybrid Decoding**: Beam search (temp=0) + temperature sampling fallback
- **Quality Metrics**: Compression ratio (2.4), log probability (-1.0), no-speech prob (0.6)
- **CTranslate2 Backend**: More efficient than raw PyTorch
- **Production Features**: VAD filtering, batched inference, word timestamps
- **Performance**: 4x faster than original OpenAI implementation

### Why This SOTA Strategy?
1. **Proven in Production**: SYSTRAN's implementation battle-tested
2. **Quality-Based Switching**: Automatic fallback prevents poor outputs
3. **Advanced Metrics**: Multiple quality assessment criteria
4. **Performance Optimized**: CTranslate2 backend + batching
5. **Comprehensive**: VAD, word timestamps, hallucination detection

### Implementation Differences from Original Plan
| Feature | Original Plan | SOTA (faster-whisper) | Advantage |
|---------|--------------|----------------------|-----------|
| Beam Search | Basic implementation | Advanced with patience/length penalty | Better accuracy |
| Temperature Fallback | Simple retry sequence | Quality-based switching | Robust quality |
| Metrics | Basic confidence check | Compression ratio + log prob + VAD | Comprehensive |
| VAD Integration | Optional feature | Core filtering mechanism | Efficiency |
| Word Timestamps | Basic cross-attention | Dynamic time warping | Higher precision |

## Risk Assessment & Mitigation (Updated for SOTA)

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **SOTA Decoding Complexity** | Medium | High | Follow proven faster-whisper algorithm exactly |
| **CTranslate2 Integration** | Low | Medium | Use Burn's native operations, verify against CTranslate2 behavior |
| **Quality Metric Calculation** | Low | Medium | Unit test compression ratio and log prob extensively |
| **Performance Regression** | Low | Medium | Benchmark against faster-whisper, optimize critical paths |
| **VAD Model Integration** | Medium | Medium | Use existing rust crates (earshot, silero-vad-rust) |

### Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **SOTA Development Time** | Medium | Medium | Follow proven algorithm, no research needed |
| **Community Adoption** | Low | Medium | Python compatibility, clear SOTA documentation |
| **Maintenance** | Low | Medium | Comprehensive test coverage, modular design |

## Success Metrics (SOTA Focus)

### Performance Metrics
- **8x realtime** transcription on CUDA (large-v2)
- **<2GB VRAM** usage for large-v2 model  
- **<0.5s startup** time
- **+4% WER improvement** vs faster-whisper (Rust advantage)

### Quality Metrics (SOTA)
- **Compression ratio compliance**: <2.4 threshold
- **Log probability compliance**: >-1.0 threshold
- **No-speech detection accuracy**: >95% correct filtering
- **Word timestamp precision**: <50ms average error
- **Language accuracy**: >98% correct detection

### Adoption Metrics
- **1000+ GitHub stars** within 6 months
- **10,000+ downloads** on crates.io
- **Python package** with 1000+ installs
- **Community contributions** from 20+ developers
- **SOTA documentation** with decoding strategy guide

## Conclusion

WhisperForge represents a **next-generation advancement** in ASR technology by combining:

- **Rust's safety and performance advantages**
- **Burn's cutting-edge ML framework with CUDA backend**
- **Proven SOTA decoding strategy** from faster-whisper analysis
- **10x performance improvement** through optimized Rust implementation
- **Production-ready features**: VAD filtering, word timestamps, batched processing

The **SOTA-first approach** ensures we're implementing proven, battle-tested algorithms rather than theoretical improvements. By adopting faster-whisper's hybrid decoding strategy, we achieve:

- **Immediate quality**: Proven production-ready decoding
- **Robust fallback**: Automatic quality-based switching
- **Performance edge**: Rust's speed advantages over Python implementations
- **Feature completeness**: Comprehensive production feature set

This positions WhisperForge not just as faster, but as a **most accurate and robust** Whisper implementation available, with proven SOTA decoding strategy at its core.

---

*Last Updated: January 19, 2026*
*Strategic Update: Adopted faster-whisper SOTA decoding strategy based on comprehensive source code analysis*