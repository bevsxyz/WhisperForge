# Task Breakdown Template

Use this template when starting a new development task to clarify requirements and identify blockers.

## Task Information

**Task**: [Clear task name]  
**Phase**: [From DEVELOPMENT_PLAN.md]  
**Priority**: [Critical/High/Medium/Low]  
**Estimated Effort**: [1-2 days / 2-5 days / etc.]

## Objective

What are we trying to accomplish? (1-2 sentences)

## Requirements

### Functional Requirements
- [ ] Requirement 1
- [ ] Requirement 2
- [ ] Requirement 3

### Quality Requirements
- [ ] No clippy warnings
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Performance targets met (if applicable)

## Implementation Plan

### Step 1: Design
What data structures/modules do we need?

```rust
// Sketch the API
pub struct MyNewType { }
pub fn my_function() -> Result<T> { }
```

### Step 2: Core Implementation
Which files need modification?
- `file1.rs` - description
- `file2.rs` - description

### Step 3: Testing
What test cases are needed?
- [ ] Happy path: normal input
- [ ] Error case: invalid input
- [ ] Edge case: boundary condition
- [ ] Integration: cross-crate interaction

### Step 4: Verification
How will we verify it works?
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing (how?)
- [ ] Performance benchmarks (if applicable)

## Potential Blockers

What could go wrong?
1. **Blocker**: Impact [High/Medium/Low] - Mitigation
2. **Blocker**: Impact [High/Medium/Low] - Mitigation

## File References

Key files to understand:
- `file1.rs:line` - What it does
- `file2.rs:line` - What it does

## Success Criteria

Task is complete when:
- [ ] All requirements met
- [ ] All tests passing
- [ ] Clippy passes with no warnings
- [ ] Documentation updated
- [ ] Performance targets met (if applicable)
- [ ] Code reviewed and approved

## Rollback Plan

If something goes wrong:
1. Revert to previous commit: `git reset --hard HEAD~1`
2. Or start fresh: `git checkout -b <new-branch>`
3. Or ask for help if truly stuck

## Notes

Any additional context or research findings?

---

**Template Version**: 1.0  
**Use Case**: Breaking down DEVELOPMENT_PLAN.md tasks into actionable steps
