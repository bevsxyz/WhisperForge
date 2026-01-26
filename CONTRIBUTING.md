# Contributing to WhisperForge

Welcome! This document provides guidelines for contributing to WhisperForge, optimized for both human and AI agent workflows.

## Table of Contents

1. [Development Workflow](#development-workflow)
2. [Code Standards](#code-standards)
3. [Testing Requirements](#testing-requirements)
4. [Pull Request Process](#pull-request-process)
5. [Common Development Tasks](#common-development-tasks)
6. [Performance Considerations](#performance-considerations)
7. [Debugging Guide](#debugging-guide)

## Development Workflow

### Before Starting a Task

1. **Read the context files** (takes 5-10 minutes):
   - `README.md` - Project overview
   - `PROJECT_STATUS.md` - Current blockers and progress
   - `DEVELOPMENT_PLAN.md` - Feature roadmap and priorities
   - `AGENTS.md` - Detailed coding guidelines

2. **Check current status**:
   ```bash
   git status
   git log --oneline -5
   ```

3. **Identify your focus**:
   - Bug fix: Reference issue number and reproduction steps
   - Feature: Reference week/phase in DEVELOPMENT_PLAN.md
   - Refactor: Understand performance implications (see AGENTS.md)

### Feature Implementation Flow

```
1. Plan
   ↓
2. Implement
   ↓
3. Test
   ↓
4. Format & Lint
   ↓
5. Commit
   ↓
6. Push & Create PR
```

### Step 1: Plan

For each feature:

1. **Break down into subtasks** (identify blockers early)
   - What data structures are needed?
   - What existing code do I need to modify?
   - What tests are required?

2. **For AI agents**: Use `.claude/prompts/` templates for task decomposition

3. **Check dependencies**:
   ```bash
   cargo tree -p whisperforge-core
   ```

### Step 2: Implement

**File locations by feature**:

| Feature | Primary Crate | Key Files |
|---------|---------------|-----------|
| Model architecture | whisperforge-core | `src/model.rs` |
| Audio processing | whisperforge-core | `src/audio.rs` |
| Decoding strategy | whisperforge-core | `src/decoding.rs` |
| Model conversion | whisperforge-convert | `src/convert.rs` |
| CLI interface | whisperforge-cli | `src/main.rs` |
| VAD/Alignment | whisperforge-align | `src/lib.rs` |
| Diarization | whisperforge-diarize | `src/lib.rs` |

**During implementation**:

- Create a feature branch: `git checkout -b feature/my-feature`
- Commit frequently with descriptive messages
- Keep commits atomic (one logical change per commit)
- Reference related code with line numbers: `src/model.rs:42`

### Step 3: Test

```bash
# Run tests for modified crates
cargo test -p whisperforge-core

# Run specific test with output
cargo test -p whisperforge-core test_name -- --nocapture

# Test all (except known failures)
cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli
```

**Coverage requirements**:
- All public functions must have unit tests
- Integration tests for cross-crate features
- Performance benchmarks for hot paths (audio processing, decoding)

### Step 4: Format & Lint

```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --all-targets --all-features

# Run clippy with warnings as errors
cargo clippy -- -D warnings
```

**Before committing**: Ensure `cargo clippy` passes with no warnings.

### Step 5: Commit

Use clear, descriptive commit messages:

```bash
# Feature commits
git commit -m "feat: Add SOTA decoding with beam search and temperature fallback"

# Bug fix commits
git commit -m "fix: Correct cross-attention weight loading in model converter"

# Refactor commits
git commit -m "refactor: Extract audio preprocessing into separate module"

# Documentation commits
git commit -m "docs: Update decoding strategy explanation in README"

# Test commits
git commit -m "test: Add integration test for full transcription pipeline"
```

**Format**: `<type>: <description>`

Types:
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code reorganization (no logic change)
- `perf` - Performance improvement
- `test` - Test addition/update
- `docs` - Documentation update
- `chore` - Build/tooling changes

### Step 6: Push & Create PR

```bash
git push origin feature/my-feature
# Create PR on GitHub with clear description
```

## Code Standards

### Import Organization

```rust
// 1. Standard library first
use std::path::PathBuf;
use std::sync::Arc;

// 2. External crates (alphabetical)
use anyhow::{Context, Result};
use burn::tensor::Tensor;
use clap::Parser;

// 3. Local modules (crate path first)
use crate::model::WhisperModel;
use crate::audio::AudioProcessor;
```

### Naming Conventions

```rust
// Types: PascalCase
pub struct WhisperModel<B: Backend> { }

// Functions/methods: snake_case
pub fn transcribe_audio(&self, audio: &[f32]) -> Result<String> { }

// Constants: SCREAMING_SNAKE_CASE
pub const MAX_BATCH_SIZE: usize = 16;

// Variables: snake_case
let sample_rate = 16000;
```

### Error Handling

Use `anyhow::Result<T>` for all fallible functions:

```rust
use anyhow::{Context, Result, anyhow};

pub fn load_model(path: &Path) -> Result<WhisperModel<Cuda>> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read model: {}", path.display()))?;
    
    if data.is_empty() {
        return Err(anyhow!("Model file is empty"));
    }
    
    Ok(model)
}
```

**Key patterns**:
- Use `.with_context()` to add context about what operation failed
- Use `anyhow!()` for unrecoverable errors
- Never panic in library code (only in `main()`)
- Log errors with sufficient context for debugging

### Documentation

All public functions require doc comments:

```rust
/// High-performance audio transcription with Burn backend.
///
/// Processes audio through mel-spectrogram encoding and token generation
/// using SOTA hybrid decoding strategy.
///
/// # Arguments
/// * `audio` - Audio samples at 16kHz sample rate
/// * `config` - Decoding configuration with beam size and temperature
///
/// # Returns
/// Transcription result with tokens, text, and quality metrics
///
/// # Errors
/// Returns error if:
/// - Audio format is invalid
/// - Model weights are corrupted
/// - CUDA device is unavailable
///
/// # Example
/// ```rust
/// let result = model.transcribe(&audio, &config)?;
/// println!("Text: {}", result.text);
/// ```
pub fn transcribe(&self, audio: &[f32], config: &DecodingConfig) -> Result<TranscriptionResult> {
```

### Performance Critical Code

For hot paths (audio processing, decoding loops):

```rust
// ✅ GOOD: Pre-allocate buffers
let mut output = Vec::with_capacity(expected_size);

// ✅ GOOD: Use tensor views instead of clones
let slice = tensor.slice([0..batch_size]);

// ❌ AVOID: Clones in loops
for sample in audio {
    let audio_clone = audio.clone();  // Wrong!
}

// ✅ GOOD: Batch operations
let results = inputs.par_iter().map(|x| process(x)).collect();

// ✅ GOOD: Use references
fn process(&self, data: &[f32]) -> Result<Vec<u32>>
```

See [AGENTS.md](AGENTS.md) for detailed performance guidelines.

## Testing Requirements

### Unit Tests

Every public function needs a unit test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_load_model_valid_path() -> Result<()> {
        // Arrange
        let model_path = "test_models/tiny.en";
        
        // Act
        let model = WhisperModel::load(model_path)?;
        
        // Assert
        assert!(model.is_loaded());
        Ok(())
    }

    #[test]
    fn test_load_model_invalid_path() {
        let result = WhisperModel::load("nonexistent.bin");
        assert!(result.is_err());
    }
}
```

### Integration Tests

For cross-crate features, use `tests/` directory:

```
tests/
├── transcription_pipeline.rs
└── model_conversion.rs
```

### Test Naming Convention

- `test_<function>_<scenario>`: Tests for success cases
- `test_<function>_error_<case>`: Tests for error cases

Examples:
- `test_transcribe_short_audio`
- `test_transcribe_empty_audio`
- `test_decode_with_fallback_quality_metric`

### Performance Benchmarks

For performance-critical sections, include benchmarks:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- transcribe
```

Use existing benchmark patterns in the codebase as reference.

## Pull Request Process

### Before Creating PR

1. **All tests pass**:
   ```bash
   cargo test -p whisperforge-core -p whisperforge-convert -p whisperforge-cli
   ```

2. **Code formatted**:
   ```bash
   cargo fmt --all
   ```

3. **No clippy warnings**:
   ```bash
   cargo clippy --all-targets --all-features
   ```

4. **Documentation complete**:
   - Doc comments on public items
   - README.md updated if behavior changes
   - DEVELOPMENT_PLAN.md updated if schedule changes

### PR Description Template

```markdown
## What does this PR do?
Brief description of changes.

## Related Issue/Task
- References DEVELOPMENT_PLAN.md phase/week
- Or references GitHub issue number

## Testing
- [ ] Unit tests added
- [ ] Integration tests pass
- [ ] Manual testing completed

## Performance Impact
- Expected impact on speed/memory (if any)
- Benchmarks if applicable

## Checklist
- [ ] Code formatted (`cargo fmt`)
- [ ] Tests pass (`cargo test`)
- [ ] Clippy passes (`cargo clippy`)
- [ ] Documentation updated
```

### Code Review Expectations

**For AI agents**:
- Verify logic correctness against reference implementation
- Check error handling completeness
- Validate performance optimizations
- Ensure compliance with AGENTS.md standards

**For humans**:
- Architectural consistency
- Design feedback
- Optimization suggestions

## Common Development Tasks

### Adding a New Feature

1. Create feature branch: `git checkout -b feature/my-feature`
2. Implement with tests
3. Update DEVELOPMENT_PLAN.md if schedule changes
4. Run full test suite
5. Format and lint
6. Create PR with clear description

### Fixing a Bug

1. Create bug branch: `git checkout -b fix/bug-description`
2. Write failing test first (TDD style)
3. Fix the bug
4. Verify test passes
5. Run full test suite
6. Create PR with issue reference

### Optimizing Performance

1. Profile to identify bottleneck: `cargo build --release && time cargo run --release`
2. Implement optimization
3. Benchmark before/after
4. Document improvement in commit message
5. Run full test suite to ensure no regressions

### Refactoring Code

1. Ensure all tests pass before starting
2. Make minimal changes (one logical operation)
3. Keep tests passing throughout
4. Use git to revert if needed: `git reset --hard`

### Debugging Tips

```bash
# Run test with output
cargo test test_name -- --nocapture

# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run with debug symbols in release mode
cargo test --release

# Use println! debugging (temporary)
println!("Debug value: {:?}", variable);
cargo test -- --nocapture

# Use RUST_LOG for structured logging (future)
RUST_LOG=debug cargo run
```

## Performance Considerations

WhisperForge has strict performance targets (see [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)):

- **tiny.en**: 200x realtime on CUDA
- **large-v2**: 8x realtime on CUDA
- **Memory**: 2GB max VRAM usage

### When to Optimize

Only optimize if:
- Profiling shows this is the bottleneck
- Change improves user experience measurably
- No loss of accuracy
- Code remains readable

### How to Benchmark

```bash
# For hot paths, use release mode
cargo build --release
time cargo run --release -- audio.wav

# For detailed profiling (requires tools)
cargo install cargo-flamegraph
cargo flamegraph
```

## Debugging Guide

### Common Issues

**Problem**: Tests fail with "cross-attention weights incorrect"
```bash
# Solution: Check model conversion logic in whisperforge-convert/src/convert.rs
# Verify tensor names mapping matches OpenAI format
```

**Problem**: Audio processing produces incorrect spectrograms
```bash
# Solution: Check STFT parameters in src/audio.rs
# Verify sample rate conversion (should be 16kHz)
```

**Problem**: Model generates EOT token immediately
```bash
# Solution: Debug cross-attention loading
# Use tests in load.rs to verify weight corruption
```

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for known issues and their fixes.

## Getting Help

1. **Check existing documentation**:
   - `README.md` - Quick reference
   - `AGENTS.md` - Detailed coding standards
   - `DEVELOPMENT_PLAN.md` - Feature specifications
   - `PROJECT_STATUS.md` - Known issues and blockers

2. **For AI agents** (Claude CLI):
   - Review `.claude/context.md` for quick context
   - Use `.claude/prompts/` templates for task breakdown
   - Ask clarifying questions before implementing

3. **For bugs**: Create reproducible test case first

## Questions or Feedback?

- **For OpenCode/Claude CLI**: Provide clear task description with context
- **For feature requests**: Reference relevant phase in DEVELOPMENT_PLAN.md
- **For bugs**: Include error message, test case, and environment details

---

**Last Updated**: January 26, 2026  
**Optimized for**: Claude CLI / opencode AI agents
