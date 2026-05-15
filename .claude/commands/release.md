Release checklist for WhisperForge. Stop and report on first failure.

1. Confirm working directory is clean: `git status --porcelain` — stop if dirty.

2. Run `mise run release-check` (compile + smoke tests) — stop if any step fails.

3. Determine the version bump type:
   - `patch` — bug fixes only
   - `minor` — new features, backwards-compatible
   - `major` — breaking changes
   Ask the user which type if not specified.

4. Dry-run to preview: `cargo release <type> --no-confirm --dry-run`
   Show the output so the user can confirm the version and CHANGELOG preview.

5. On confirmation: `cargo release <type> --no-confirm`
   This bumps Cargo.toml version, regenerates CHANGELOG via git-cliff, commits both, tags with v{{version}}, and pushes.
   The GitHub Actions release workflow triggers automatically on the pushed tag.
