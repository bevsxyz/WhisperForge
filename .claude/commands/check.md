Run the full quality gate (see CLAUDE.md § Commands):

```bash
cargo fmt --all -- --check && cargo clippy --all-targets --all-features && cargo check --all
```

Report which step passed or failed. Show first 30 lines of output on failure.