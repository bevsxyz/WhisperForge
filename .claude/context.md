# WhisperForge - Quick Context for Claude CLI

**Last Updated**: January 26, 2026  
**Current Status**: Core infrastructure complete, SOTA decoding in progress

## 🎯 Project Purpose

WhisperForge is a high-performance Rust rewrite of WhisperX, combining:
- Burn ML framework for CUDA acceleration
- State-of-the-art speech recognition (Whisper models)
- Advanced decoding strategies from faster-whisper (SYSTRAN)
- Speaker diarization and word-level timestamps

**Target Performance**: 8x realtime transcription on CUDA (large-v2 model) with 2GB VRAM

## 📊 Current Development State

| Component | Status | Completion |
|-----------|--------|------------|
| **Model Architecture** | ✅ Complete | 100% |
| **Model Conversion (HF→Burn)** | ✅ Complete | 100% |
| **Audio Pipeline** | ✅ Working | 80% |
| **Token Generation** | ✅ Working | 100% |
| **SOTA Decoding** | 🔴 **In Progress** | 20% |
| **CLI Interface** | 🟡 Basic | 70% |
| **VAD Integration** | ⏳ Planned | 0% |
| **Word Timestamps** | ⏳ Planned | 0% |
| **Speaker Diarization** | ⏳ Planned | 0% |

## 🚀 Technology Stack

### Core
- **Burn 0.20.0** - ML framework with CUDA backend
- **Rust 1.70+** - Memory-safe systems language

### Audio Processing
- **rubato** - Audio resampling
- **hound** - WAV I/O
- **ffmpeg-next** - Multi-format support
- **earshot** - Voice activity detection

### Key Dependencies
- **tokenizers** - BPE tokenization
- **serde** - JSON serialization
- **clap** - CLI arguments
- **anyhow** - Error handling

## 📁 Project Structure

```
whisperforge/
├── whisperforge-core/        # Core Whisper + SOTA decoding
│   ├── src/model.rs          # Architecture with custom attention
│   ├── src/audio.rs          # Mel spectrogram computation
│   ├── src/load.rs           # Model loader (Burn format)
│   └── src/decoding.rs       # 🔴 SOTA decoding (IN PROGRESS)
├── whisperforge-align/       # Wav2Vec2 alignment & VAD
├── whisperforge-diarize/     # Speaker diarization
├── whisperforge-cli/         # CLI binary
├── whisperforge-convert/     # HuggingFace → Burn converter
└── .claude/                  # Claude CLI configuration
```

## 🎯 Current Focus: SOTA Decoding (Phase 1)

### What is SOTA Decoding?

Implementation of the proven decoding strategy from [faster-whisper](https://github.com/guillaumekln/faster-whisper) (SYSTRAN):

```
1. Start with BEAM SEARCH (temperature=0.0)
2. Evaluate output quality:
   - Compression ratio: len(text) / len(gzip(text))
   - Log probability: average token confidence
   - No-speech detection: silence probability
3. If quality metrics FAIL, FALLBACK to temperature sampling:
   [0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0]
4. Return first result that passes quality thresholds
```

### Quality Thresholds

```rust
DecodingConfig {
    compression_ratio_threshold: 2.4,    // Text ratio (lower = suspicious)
    log_prob_threshold: -1.0,             // Confidence (higher = better)
    no_speech_threshold: 0.6,             // Silence detection
}
```

### Why This Strategy?

- **Proven in production** (used by SYSTRAN)
- **Robust**: Automatic fallback prevents garbage output
- **Fast**: Beam search is faster than greedy, fallback handles edge cases
- **Quality**: Hybrid approach gives best accuracy/speed tradeoff

## 🏗️ Architecture Overview

### Model Architecture (✅ Complete)

**Encoder** (16 layers):
- Self-attention blocks with multi-head attention
- Position-wise feed-forward networks
- Layer normalization + residual connections

**Decoder** (16 layers):
- Self-attention (attends to previous tokens)
- Cross-attention (attends to encoder output)
- Feed-forward networks
- Layer normalization + residual connections

### Audio Pipeline (✅ Complete)

```
Audio File → FFmpeg Load → Resample to 16kHz → 
Mel Spectrogram (STFT) → Encoder → Token Generation
```

### Key Files & Their Purpose

| File | Purpose | Status |
|------|---------|--------|
| `whisperforge-core/src/model.rs` | Model architecture definition | ✅ 100% |
| `whisperforge-core/src/load.rs` | Load converted models | ✅ 100% |
| `whisperforge-core/src/audio.rs` | Mel spectrogram + audio I/O | ✅ 80% |
| `whisperforge-core/src/decoding.rs` | **SOTA decoding** | 🔴 20% |
| `whisperforge-convert/src/convert.rs` | HF → Burn model conversion | ✅ 100% |
| `whisperforge-cli/src/main.rs` | CLI interface | 🟡 70% |
| `DEVELOPMENT_PLAN.md` | 4-week roadmap with details | ✅ Reference |
| `AGENTS.md` | Coding standards for AI agents | ✅ Reference |

## 📝 Important Files Reference

- **README.md** - Project overview, quick start, key commands
- **CONTRIBUTING.md** - Development workflow, code standards, testing
- **PROJECT_STATUS.md** - Current blockers, next steps, git status
- **DEVELOPMENT_PLAN.md** - Full 4-week roadmap with SOTA research
- **AGENTS.md** - Comprehensive coding guidelines for AI agents

## 🔄 Development Workflow

### For a New Task (5-minute checklist)

1. **Read status**: `PROJECT_STATUS.md` → identify blockers
2. **Review plan**: `DEVELOPMENT_PLAN.md` → understand scope
3. **Check code style**: `AGENTS.md` → follow patterns
4. **Create branch**: `git checkout -b feature/my-feature`
5. **Implement + test**: Write code with unit tests
6. **Format + lint**: `cargo fmt && cargo clippy`
7. **Commit**: Clear, atomic commits
8. **Test all**: `cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli`

### Quick Commands

```bash
# Build and check
cargo check --all

# Run tests (modified crates only)
cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli

# Format + lint
cargo fmt --all && cargo clippy --all-targets --all-features

# Build release
cargo build --release

# Run CLI
cargo run -p whisperforge-cli -- -a test_audio.wav -m tiny_en_converted
```

## ⚡ Performance Targets

| Model | CUDA Speed | VRAM | Status |
|-------|-----------|------|--------|
| tiny.en | 200x realtime | 0.5GB | ✅ Target |
| base | 150x realtime | 1GB | ✅ Target |
| small | 100x realtime | 2GB | ✅ Target |
| medium | 80x realtime | 5GB | 🔄 Working |
| **large-v2** | **8x realtime** | **2GB** | 🎯 **PRIORITY** |

## 🚨 Known Issues & Blockers

1. **Basic greedy decoding only** (need SOTA implementation)
   - Fix: Implement `whisperforge-core/src/decoding.rs` with beam search + fallback

2. **No VAD filtering yet**
   - Impact: Slower on long silences
   - Fix: Integrate earshot or silero-vad-rust

3. **No word-level timestamps**
   - Impact: Can't do precise transcription alignment
   - Fix: Extract from cross-attention weights

4. **No speaker diarization**
   - Impact: Can't identify which speaker said what
   - Fix: Integrate pyannote-rs or sherpa-rs

See `PROJECT_STATUS.md` for detailed current status.

## 🎓 Key Concepts for SOTA Implementation

### Beam Search vs Greedy Decoding

**Greedy** (current): Pick highest probability token at each step
- Fast but can get stuck in local optima
- May produce unlikely sequences

**Beam Search** (target): Keep top-K promising paths, expand them
- More accurate, finds better sequences
- Slightly slower but worth it

### Temperature Sampling

Controls randomness in token selection:
- **0.0**: Greedy (deterministic, same output)
- **0.0-0.5**: Conservative (mostly likely tokens)
- **0.5-1.0**: Balanced
- **>1.0**: Random (too unreliable)

Fallback sequence: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`

### Quality Metrics

**Compression Ratio**:
```
ratio = len(original_text) / len(gzip_compressed_text)
- Normal text ratio: 2.0-4.0
- Hallucinated gibberish: >2.4 (compresses poorly)
- Threshold: 2.4 (reject if higher)
```

**Log Probability**:
```
avg_logprob = sum(token_logprobs) / num_tokens
- Confident tokens: close to 0 (logprob of 1.0)
- Uncertain tokens: very negative
- Threshold: -1.0 (reject if lower)
```

## 🔧 Debugging Tips

### Problem: "Model generates EOT token immediately"
- **Cause**: Cross-attention weights corrupted during conversion
- **Fix**: Check tensor name mapping in `whisperforge-convert/src/convert.rs`
- **Test**: `cargo test -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture`

### Problem: "Audio spectrograms look wrong"
- **Cause**: STFT parameters incorrect
- **Fix**: Verify sample rate = 16kHz, window size = 400, hop = 160
- **Reference**: Original Whisper implementation parameters

### Problem: "Tests failing with unknown error"
- **Solution**: Run with backtrace:
  ```bash
  RUST_BACKTRACE=1 cargo test test_name -- --nocapture
  ```

## 📚 Research Notes

**Source**: Analysis of faster-whisper/transcribe.py (SYSTRAN, Feb 2024)

**Key Findings**:
- Hybrid decoding critical for production quality
- Quality metrics prevent hallucination
- Fallback sequence: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] is optimal
- CTranslate2 backend more efficient than raw PyTorch
- VAD filtering provides 2-3x speedup on real speech

**Why Rust Implementation Benefits**:
- No Python GIL → true parallelization
- Memory-safe concurrent processing
- Direct CUDA bindings via Burn
- Compiled to native code → faster startup

## 🎯 Next Steps Priority

1. **CRITICAL** (Week 1-2): Implement SOTA decoding
   - Beam search with patience parameter
   - Temperature sampling fallback
   - Quality metric calculation
   - Unit tests for all components

2. **HIGH** (Week 2): CLI integration
   - Add `--beam-size`, `--temperature` parameters
   - Add `--decoding-preset fast|balanced|accurate`
   - Integrate VAD filtering
   - Update help documentation

3. **MEDIUM** (Week 3): Advanced features
   - Word-level timestamps from attention
   - Batch processing optimization
   - SRT output format

4. **LOW** (Week 4): Polish
   - Speaker diarization
   - Performance benchmarking
   - Documentation updates

## 💡 AI Agent Tips

**When implementing a feature**:
1. Start with the smallest working example
2. Add complexity incrementally
3. Test after each addition
4. Don't over-optimize until you have working code
5. Reference failing tests to understand requirements

**When debugging**:
1. Add debug output (println!) temporarily
2. Check git history for recent changes
3. Run tests in isolation
4. Use `git bisect` to find regression
5. Ask for clarification if specification is ambiguous

**When optimizing**:
1. Profile first, optimize second
2. Only optimize hot paths
3. Verify no accuracy loss
4. Benchmark before/after
5. Document the optimization

## ❓ Common Questions

**Q: Should I modify the model architecture?**
A: No. Architecture is frozen at Burn 0.20. Focus on decoding and features.

**Q: What if a test fails?**
A: Check `PROJECT_STATUS.md` for known issues. If new, add it to the status and investigate.

**Q: How do I know if my implementation is correct?**
A: Compare output with faster-whisper on same audio. Quality metrics should match.

**Q: Should I write unsafe code?**
A: Only if profiling proves it's the bottleneck AND documentation justifies it. Generally avoid.

**Q: How much context should I request?**
A: You have access to this file. If you need more, ask for specific file sections.

---

**For detailed guidelines**: See `AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPMENT_PLAN.md`  
**For current status**: See `PROJECT_STATUS.md`  
**For quick commands**: See `README.md`
