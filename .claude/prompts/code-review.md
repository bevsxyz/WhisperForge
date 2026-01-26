# Code Review Checklist

Use this checklist when reviewing code changes (for AI agents or humans).

## Functional Correctness

- [ ] Code implements the stated requirements
- [ ] Logic is sound and handles edge cases
- [ ] Error cases are properly handled
- [ ] No infinite loops or deadlocks
- [ ] Performance is acceptable (no obvious O(n²) or worse where avoidable)

## Code Quality

- [ ] Follows naming conventions from AGENTS.md
- [ ] Functions are focused and have single responsibility
- [ ] Functions are reasonably sized (<50 lines preferred)
- [ ] Comments explain "why" not "what"
- [ ] No commented-out code left behind
- [ ] No debug println!() statements left in production code

## Error Handling

- [ ] All fallible operations return `Result<T>`
- [ ] Errors use `.with_context()` to add helpful context
- [ ] No unwrap() except in main() or tests
- [ ] No panic!() in library code
- [ ] Error messages are user-friendly

## Memory & Performance

- [ ] No unnecessary clones in hot paths (check AGENTS.md)
- [ ] Tensors use references/views, not copies
- [ ] Pre-allocated buffers where applicable
- [ ] Batch operations used instead of element-wise loops
- [ ] No obvious memory leaks or unbounded allocations

## Testing

- [ ] All public functions have unit tests
- [ ] Tests use descriptive names: `test_<function>_<scenario>`
- [ ] Tests cover happy path and error cases
- [ ] Integration tests verify cross-crate interaction
- [ ] All tests pass: `cargo test`

## Documentation

- [ ] Public functions have doc comments with `///`
- [ ] Doc comments include examples for complex functions
- [ ] No typos or grammatical errors
- [ ] README.md updated if behavior changes
- [ ] DEVELOPMENT_PLAN.md updated if timeline changes

## Code Style

- [ ] Imports organized (std → external → local)
- [ ] All code formatted: `cargo fmt --check`
- [ ] No clippy warnings: `cargo clippy --all-targets --all-features`
- [ ] Code style matches existing codebase
- [ ] Follows patterns from AGENTS.md

## Git & Commits

- [ ] Commits are atomic (one logical change each)
- [ ] Commit messages are clear and descriptive
- [ ] Format: `<type>: <description>` (feat/fix/refactor/test/docs/chore)
- [ ] No merge commits in history (rebase-friendly)
- [ ] Branch name is descriptive: `feature/name` or `fix/issue`

## Architecture

- [ ] Changes fit the overall architecture
- [ ] No unexpected dependencies added
- [ ] New public APIs are well-designed
- [ ] No breaking changes to existing APIs (unless intentional)
- [ ] Comments explain complex architectural decisions

## SOTA Decoding Specific (if applicable)

- [ ] Implementation matches faster-whisper strategy
- [ ] Quality metrics are correctly calculated
- [ ] Compression ratio uses gzip compression
- [ ] Log probability is averaged correctly
- [ ] Fallback sequence is [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- [ ] Threshold values match DEVELOPMENT_PLAN.md

## Performance Specific (if applicable)

- [ ] Profiling was done to identify bottleneck
- [ ] Optimization improves measured performance
- [ ] No accuracy loss from optimization
- [ ] Benchmarks show improvement
- [ ] Code remains readable

## Before Merge

- [ ] All tests passing: `cargo test`
- [ ] No clippy warnings: `cargo clippy`
- [ ] Code formatted: `cargo fmt`
- [ ] Documentation complete
- [ ] Reviewer approval obtained
- [ ] CI/CD checks pass (if configured)

## Common Issues to Check For

**Tensor Operations**:
- ✅ Using `tensor.slice()` or views instead of `.clone()`
- ❌ Avoid `tensor.clone()` in loops

**Error Handling**:
- ✅ `.with_context(|| format!("..."))?`
- ❌ Avoid `.unwrap()` outside main()

**Testing**:
- ✅ `test_transcribe_empty_audio()` → tests error case
- ❌ Only happy path tests

**Performance**:
- ✅ Pre-allocated `Vec::with_capacity()`
- ❌ Repeated allocations in loops

**Code Style**:
- ✅ `pub fn transcribe_audio(&self) -> Result<String>`
- ❌ `pub fn transcribeAudio()`

## Questions to Ask

1. **Does this change align with DEVELOPMENT_PLAN.md?**
   - If no, discussion needed

2. **Are performance targets maintained?**
   - Target: 8x realtime for large-v2
   - If slower, needs optimization or justification

3. **Does this introduce new dependencies?**
   - If yes, is it justified? Check workspace dependencies.

4. **Will this affect other crates?**
   - Check if API changes are breaking

5. **Is error handling comprehensive?**
   - Are all error paths covered?

## Sign-Off

- [ ] Code reviewed
- [ ] No blocking issues
- [ ] Approved for merge
- [ ] Reviewer: [name]
- [ ] Date: [date]

---

**Template Version**: 1.0  
**Use Case**: Ensuring code quality for SOTA decoding and performance-critical path
