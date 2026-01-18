# WhisperForge - Agent Development Guidelines

This file contains guidelines and commands for agentic coding agents working on the WhisperForge repository.

## Repository Structure

WhisperForge is organized as a multi-crate workspace:

```
whisperforge/
├── whisperforge-core/     # Core Whisper models (Burn)
├── whisperforge-align/    # Wav2Vec2 alignment & VAD
├── whisperforge-diarize/  # Speaker diarization
├── whisperforge-cli/      # CLI binary & Python API
├── whisperforge-convert/  # Model conversion tools
└── Cargo.toml            # Workspace configuration
```

## Build, Test, and Lint Commands

### Workspace Commands
```bash
# Build all crates in debug mode
cargo build

# Build all crates in release mode
cargo build --release

# Build specific crate
cargo build -p whisperforge-core

# Run tests for all crates
cargo test

# Run tests for specific crate
cargo test -p whisperforge-core

# Run single test
cargo test test_name -- --exact

# Run tests with output
cargo test -- --nocapture

# Run tests in release mode
cargo test --release

# Check code without building
cargo check

# Check specific crate
cargo check -p whisperforge-core
```

### Linting and Formatting
```bash
# Format code (all crates)
cargo fmt

# Format specific crate
cargo fmt -p whisperforge-core

# Run clippy lints
cargo clippy --all-targets --all-features

# Run clippy for specific crate
cargo clippy -p whisperforge-core --all-targets --all-features

# Run clippy with warnings as errors
cargo clippy -- -D warnings
```

### Development Commands
```bash
# Run CLI with development build
cargo run -p whisperforge-cli -- --help

# Run with specific arguments
cargo run -p whisperforge-cli -- audio.wav --model tiny.en

# Run benchmarks (if available)
cargo bench

# Generate documentation
cargo doc --open

# Check for outdated dependencies
cargo outdated
```

## Code Style Guidelines

### General Principles
- **Performance First**: Optimize for CUDA acceleration and memory efficiency
- **Safety**: Leverage Rust's type system and memory safety guarantees
- **Modularity**: Keep crates focused and dependencies minimal
- **Compatibility**: Support CUDA (primary), wgpu, and CPU backends

### Imports and Dependencies
```rust
// Standard library first
use std::path::PathBuf;
use std::sync::Arc;

// External crates (alphabetical)
use anyhow::{Context, Result};
use burn::tensor::Tensor;
use clap::Parser;

// Local modules
use whisperforge_core::model::WhisperModel;
use whisperforge_core::audio::AudioProcessor;
```

**Import Rules:**
- Group imports: std, external crates, local modules
- Use `use` statements sparingly - prefer full paths in impl blocks
- Avoid `*` imports except for test modules
- Use `crate::` for local module references when needed

### Naming Conventions
```rust
// Types: PascalCase
pub struct WhisperModel<B: Backend> {
    // Fields: snake_case
    encoder: WhisperEncoder<B>,
    decoder: WhisperDecoder<B>,
}

// Functions: snake_case
pub fn transcribe_audio(&self, audio: &[f32]) -> Result<String> {}

// Constants: SCREAMING_SNAKE_CASE
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
pub const MAX_BATCH_SIZE: usize = 16;

// Enums: PascalCase, variants: PascalCase
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV2,
    LargeV3,
}
```

### Error Handling
```rust
// Use anyhow for application-level errors
use anyhow::{Context, Result, anyhow};

// Return Result<T> from fallible functions
pub fn load_model(path: &Path) -> Result<WhisperModel<Cuda>> {
    let model_data = std::fs::read(path)
        .with_context(|| format!("Failed to read model file: {}", path.display()))?;
    
    if model_data.is_empty() {
        return Err(anyhow!("Model file is empty"));
    }
    
    // ... rest of implementation
    Ok(model)
}

// Use thiserror for library-specific error types (if needed)
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("Unsupported sample rate: {0}")]
    UnsupportedSampleRate(u32),
    #[error("Audio format not supported")]
    UnsupportedFormat,
}
```

### Type System Usage
```rust
// Use generics for backend abstraction
pub trait TranscriptionBackend: Backend {
    type Model: WhisperModel<Self>;
}

// Use type aliases for complex types
pub type AudioTensor<B> = Tensor<B, 2, Float>;
pub type ModelDevice<B> = <B as Backend>::Device;

// Prefer concrete types over generics when possible
pub struct CudaWhisperModel {
    model: WhisperModel<Cuda>,
    device: CudaDevice,
}
```

### Documentation
```rust
/// High-performance Whisper transcription model using Burn framework.
/// 
/// This struct provides CUDA-accelerated transcription with support for
/// batched inference and multiple model sizes.
/// 
/// # Examples
/// 
/// ```rust
/// use whisperforge_core::WhisperModel;
/// 
/// let model = WhisperModel::new("tiny.en", device)?;
/// let transcript = model.transcribe(&audio)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
/// 
/// # Errors
/// 
/// Returns an error if:
/// - Model file cannot be loaded
/// - CUDA device is not available
/// - Audio format is invalid
/// 
/// # Performance
/// 
/// - tiny.en: ~200x realtime on CUDA
/// - large-v2: ~70x realtime on CUDA
pub struct WhisperModel<B: Backend> {
    // ... fields
}
```

### Testing Guidelines
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_model_loading() -> Result<()> {
        // Arrange
        let model_path = "test_models/tiny.en";
        let device = CudaDevice::default();
        
        // Act
        let model = WhisperModel::new(model_path, device)?;
        
        // Assert
        assert!(model.is_loaded());
        Ok(())
    }

    #[test]
    fn test_transcription_accuracy() -> Result<()> {
        // Test with known audio and expected output
        let audio = load_test_audio("test_audio.wav")?;
        let model = create_test_model()?;
        
        let result = model.transcribe(&audio)?;
        
        assert!(result.contains("expected text"));
        Ok(())
    }

    // Integration tests go in separate files: tests/integration_test.rs
}
```

### Performance Guidelines
```rust
// Use CUDA streams for parallel processing - CRITICAL for speed
pub fn transcribe_batch(&self, audio_batch: &[AudioTensor<Cuda>]) -> Result<Vec<String>> {
    let stream = CudaStream::new();
    
    // Process in parallel using CUDA streams
    let results: Vec<_> = audio_batch
        .par_iter()
        .map(|audio| self.transcribe_single(audio, &stream))
        .collect::<Result<Vec<_>>>()?;
    
    Ok(results)
}

// Pre-allocate tensors to avoid allocations in hot paths
pub struct TranscriptionState<B: Backend> {
    input_buffer: Tensor<B, 2>,
    output_buffer: Tensor<B, 2>,
    // ... other buffers
}

// Use const generics for compile-time optimization
pub struct AudioBuffer<const N: usize> {
    data: [f32; N],
}

// NEVER clone tensors in hot paths - use references or tensor views
// AVOID memory allocations in audio processing loops
// PREFER batch operations over element-wise operations
```

### Backend Abstraction
```rust
// Primary implementation for CUDA
pub type WhisperModelCuda = WhisperModel<Cuda>;

// Fallback implementations
pub type WhisperModelWgpu = WhisperModel<Wgpu>;
pub type WhisperModelCpu = WhisperModel<NdArray>;

// Factory function for backend selection
pub fn create_model(backend: BackendType, model_path: &str) -> Result<Box<dyn WhisperModelTrait>> {
    match backend {
        BackendType::Cuda => Ok(Box::new(WhisperModel::new_cuda(model_path)?)),
        BackendType::Wgpu => Ok(Box::new(WhisperModel::new_wgpu(model_path)?)),
        BackendType::Cpu => Ok(Box::new(WhisperModel::new_cpu(model_path)?)),
    }
}
```

## Development Workflow

1. **Before Coding**: Run `cargo check` to ensure code compiles
2. **While Coding**: Use `cargo clippy` frequently to catch issues early
3. **Before Commit**: Run `cargo test` and `cargo fmt`
4. **Performance Testing**: Use release builds for benchmarks
5. **Documentation**: Update examples and docstrings for public APIs

## Key Dependencies to Understand

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

## Testing Requirements

- All public functions must have unit tests
- Integration tests for end-to-end workflows
- Performance benchmarks for CUDA operations
- Memory usage validation for large models
- Cross-platform compatibility testing

## Performance Targets

- **tiny.en**: 200x realtime on CUDA
- **base**: 150x realtime on CUDA
- **large-v2**: 70x realtime on CUDA (8GB VRAM)
- **Startup**: <1s for model loading
- **Memory**: 50% reduction vs WhisperX

Remember: This is a performance-critical project. Always consider CUDA optimization, memory efficiency, and batched inference in your implementations.