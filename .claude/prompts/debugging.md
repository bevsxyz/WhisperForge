# Debugging Guide

Use this guide when troubleshooting issues in WhisperForge.

## General Debugging Strategy

1. **Reproduce the issue**
   - Create a minimal test case
   - Add to `#[test]` function
   - Run with output: `cargo test -- --nocapture`

2. **Gather information**
   - Error message (full backtrace)
   - What changed recently? `git log --oneline -10`
   - Which crate is affected?
   - Can you isolate the problem?

3. **Hypothesize**
   - What could cause this behavior?
   - What assumptions might be wrong?
   - Reference similar working code

4. **Test hypothesis**
   - Add debug output or assertions
   - Run the failing test
   - Verify hypothesis confirmed/rejected

5. **Fix and verify**
   - Apply fix
   - Run test again
   - Run full test suite
   - Commit with clear message

## Common Issues & Solutions

### Issue: "Model generates EOT token immediately"

**Symptom**: Model should generate text but outputs `<|endoftext|>` as first token

**Probable Cause**: Corrupted cross-attention weights during model conversion

**Debug Steps**:
```bash
# 1. Check conversion log
cargo test -p whisperforge-convert convert::tests::test_convert_tiny_en -- --nocapture

# 2. Inspect tensor names
cargo run -p whisperforge-convert -- inspect models/tiny_en_openai.safetensors | grep "cross_attn"

# 3. Verify weight loading
cargo test -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture
```

**Solution**: Check `whisperforge-convert/src/convert.rs`:
- Verify tensor name mapping: `decoder.layers.X.encoder_attn.*` → `blocks.X.cross_attn.*`
- Ensure weights are correctly transposed if needed
- Check dtype conversion (F16 → F32)

**Reference**: See `PROJECT_STATUS.md` "🔥 MAJOR BREAKTHROUGH: Cross-Attention Fix" section

---

### Issue: "Audio spectrogram looks wrong"

**Symptom**: Mel spectrogram has incorrect shape or values

**Probable Cause**: STFT parameters incorrect or sample rate mismatch

**Debug Steps**:
```bash
# 1. Check audio loading
cargo test -p whisperforge-core audio::tests::test_load_audio -- --nocapture

# 2. Verify STFT parameters
cargo test -p whisperforge-core audio::tests::test_mel_spectrogram -- --nocapture

# 3. Print debug info
println!("Sample rate: {}", audio.sample_rate);
println!("Spectrogram shape: {:?}", spectrogram.shape());
println!("Spectrogram values: {:?}", spectrogram);
```

**Expected Values** (Whisper standard):
- Sample rate: 16kHz (after resampling)
- Window size: 400 samples
- Hop length: 160 samples
- Number of mel bins: 80
- Output shape: [80, T] where T = (num_samples - 400) / 160 + 1

**Solution**: Check `whisperforge-core/src/audio.rs`:
- Verify `sample_rate == 16000`
- Check STFT window and hop parameters
- Compare with original Whisper implementation

---

### Issue: "Test fails with 'cross-attention weights incorrect'"

**Symptom**: Test assertion fails in model loading or inference

**Probable Cause**: Model weights not loaded correctly

**Debug Steps**:
```bash
# 1. Run with backtrace
RUST_BACKTRACE=1 cargo test test_name -- --nocapture

# 2. Check specific tensor
cargo test -p whisperforge-core load::tests::test_load_whisper_model -- --nocapture

# 3. Compare with expected
# Print tensor shape and sample values
println!("Tensor shape: {:?}", weight_tensor.shape());
println!("Sample values: {:?}", weight_tensor);
```

**Solution**:
- Verify safetensors conversion preserved tensor values
- Check dtype conversions (F16/BF16 → F32)
- Ensure no accidental transposition

---

### Issue: "Compilation fails with 'cannot find function'"

**Symptom**: `error[E0425]: cannot find function 'xyz' in this scope`

**Probable Cause**: Wrong import, API change, or typo

**Debug Steps**:
```bash
# 1. Check Burn API version
grep "burn =" Cargo.toml  # Should be 0.20.0

# 2. Find the function
rg "fn xyz" --type rust

# 3. Check if it's a method on a struct
# Look for `impl SomeType { fn xyz() ... }`
```

**Solution**:
- Verify Burn 0.20.0 API (may differ from newer versions)
- Add missing import: `use module::function;`
- Fix typo or method name
- Check if API was refactored in newer Burn

---

### Issue: "Test fails with 'CUDA error' or 'out of memory'"

**Symptom**: Tests fail only on CUDA, work fine on CPU

**Probable Cause**: Memory allocation issue or CUDA-specific bug

**Debug Steps**:
```bash
# 1. Run on CPU backend
cargo test --features "ndarray"  # If supported

# 2. Check memory usage
nvidia-smi  # During test execution

# 3. Reduce batch size or model size
# Modify test to use smaller audio chunks
```

**Solution**:
- Profile memory usage on release build
- Check for tensor accumulation without cleanup
- Verify CUDA stream management
- Consider splitting large operations

---

### Issue: "Performance is slower than expected"

**Symptom**: Transcription takes longer than target (8x realtime for large-v2)

**Probable Cause**: Suboptimal algorithm, missing optimization, or incorrect backend

**Debug Steps**:
```bash
# 1. Measure baseline
time cargo run --release -p whisperforge-cli -- -a test_audio.wav

# 2. Profile with release build
cargo build --release
perf record ./target/release/whisperforge-cli -a test_audio.wav

# 3. Check which backend is active
cargo run --release -- --help | grep backend
```

**Solution**:
- Verify CUDA backend is active (should be default)
- Check for unnecessary tensor clones
- Profile to identify bottleneck
- Reference AGENTS.md performance guidelines
- Consider batching or parallelization

---

### Issue: "Clippy warnings won't go away"

**Symptom**: `cargo clippy` shows warnings that seem correct

**Probable Cause**: Clippy warning is overly strict or false positive

**Debug Steps**:
```bash
# 1. Check specific warning
cargo clippy -- -W clippy::name_of_warning

# 2. Review Clippy docs
# Visit https://rust-lang.github.io/rust-clippy/

# 3. Verify it's legitimate
# Compare with similar patterns in codebase
```

**Solution**:
- Fix legitimate warnings (most are correct)
- Use `#[allow(clippy::name)]` only for false positives
- Add comment explaining why exception is needed
- Reference clippy docs in the comment

---

## Debugging Tools & Techniques

### Print Debugging

```rust
// Temporary debug output
println!("Variable: {:?}", my_var);
println!("Debug info: {:#?}", complex_structure);

// Run test with output
cargo test test_name -- --nocapture
```

### Test-Driven Debugging

```rust
#[test]
fn test_debug_issue() -> Result<()> {
    // Reproduce the issue
    let input = /* problematic input */;
    let result = my_function(input)?;
    
    // Add assertions to understand behavior
    println!("Result: {:?}", result);  // Temporary
    assert_eq!(result, expected);
    
    Ok(())
}

// Run with: cargo test test_debug_issue -- --nocapture
```

### Backtrace Debugging

```bash
# Full backtrace for panics
RUST_BACKTRACE=1 cargo test

# Very detailed backtrace
RUST_BACKTRACE=full cargo test

# With line numbers
RUST_BACKTRACE=full cargo test --release
```

### Git Debugging

```bash
# Find when bug was introduced
git bisect start
git bisect bad HEAD
git bisect good v0.1.0

# Then test each commit
cargo test test_name

# See recent changes
git log --oneline -p -n 5

# Diff specific file
git diff HEAD~1 src/model.rs
```

## Debugging Workflow

### For Model/Inference Issues

1. **Isolate the problem**:
   - Does issue happen in all models or specific ones?
   - Does issue happen in all inputs or specific ones?
   - Which layer/component fails?

2. **Create minimal reproduction**:
   ```rust
   #[test]
   fn test_minimal_reproduction() -> Result<()> {
       let model = create_test_model()?;
       let audio = create_minimal_test_audio();
       let result = model.transcribe(&audio)?;
       println!("Result: {}", result.text);
       assert!(!result.text.is_empty());
       Ok(())
   }
   ```

3. **Debug incrementally**:
   - Test model loading
   - Test encoder separately
   - Test decoder separately
   - Test full pipeline

### For Performance Issues

1. **Establish baseline**:
   ```bash
   # Measure on release build
   cargo build --release
   time ./target/release/whisperforge-cli -a audio.wav
   ```

2. **Profile hot path**:
   ```bash
   # Identify where time is spent
   perf record -g ./target/release/whisperforge-cli -a audio.wav
   perf report
   ```

3. **Optimize and verify**:
   - Change one thing at a time
   - Measure before/after
   - Verify accuracy unchanged

## When to Ask for Help

**Ask immediately if**:
- Error message is cryptic or unclear
- Behavior contradicts documentation
- You've tried >3 approaches without success
- Multiple crates affected
- Unsure how to proceed

**Provide when asking**:
1. Error message (full text)
2. What you've tried
3. Expected vs actual behavior
4. Minimal reproduction case
5. Recent changes (git log)

## Key Resources

- **AGENTS.md**: Debugging patterns and error handling
- **DEVELOPMENT_PLAN.md**: Known issues and blockers
- **PROJECT_STATUS.md**: Recent issues and fixes
- **Burn Documentation**: https://burn.dev/
- **This file**: `.claude/prompts/debugging.md`

---

**Template Version**: 1.0  
**Last Updated**: January 26, 2026  
**Use Case**: Quick reference for troubleshooting WhisperForge issues
