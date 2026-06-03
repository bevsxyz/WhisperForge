# Distribution

How WhisperForge is built, released, and published across package managers.
Tracked in [issue #4](https://github.com/bevsxyz/WhisperForge/issues/4).

The foundation is **[`dist`](https://opensource.axo.dev/cargo-dist/)** (cargo-dist):
one config (`[workspace.metadata.dist]` in [Cargo.toml](Cargo.toml)) builds every
target, emits SHA-256 checksums, and generates installers + a GitHub Release. Almost
every downstream package manager just consumes those release assets.

## Release artifacts (produced by `dist` on every `vX.Y.Z` tag)

| Target | Asset |
|--------|-------|
| Linux x86_64 | `whisperforge-x86_64-unknown-linux-gnu.tar.xz` |
| Linux aarch64 | `whisperforge-aarch64-unknown-linux-gnu.tar.xz` |
| macOS x86_64 | `whisperforge-x86_64-apple-darwin.tar.xz` |
| macOS aarch64 | `whisperforge-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `whisperforge-x86_64-pc-windows-msvc.zip` + `.msi` |
| (all) | sibling `<asset>.sha256` + combined `sha256.sum` |
| Installers | `whisperforge-installer.sh`, `whisperforge-installer.ps1`, `whisperforge.rb` (Homebrew), npm package |

Archives are flat (binaries at root) and bundle `wforge`, the `wf` alias shim,
`README.md`, `LICENSE`, `CHANGELOG.md`. The dev-only `stream_bench` is an `[[example]]`,
so it is **not** shipped.

> **Build portability:** release builds use a stock toolchain on `ubuntu-22.04` (broad
> glibc compatibility). The sccache wrapper and mold linker are intentionally **not** in
> `.cargo/config.toml` — they're set per-environment (CI in `ci.yml`, local dev in
> `mise.toml [env]`) so bare release runners and fresh clones build without them.

## Cutting a release (0.5.0 and beyond)

1. `git checkout main && git pull`
2. `cargo release minor` (0.4 → 0.5; use `patch`/`major` as appropriate). This bumps the
   workspace version, **upgrades the inter-crate `version =` requirements** (0.4 → 0.5),
   regenerates `CHANGELOG.md` via git-cliff, commits, tags `v0.5.0`, and pushes.
3. Pushing the tag fires two workflows in parallel:
   - **`release.yml`** (dist) → builds all targets, GitHub Release, pushes the Homebrew
     formula, publishes the npm package.
   - **`publish-crates.yml`** → publishes the 4 crates to crates.io via OIDC.
4. Verify: `dist plan` locally first; after the run, check the GitHub Release assets and
   `crates.io/crates/whisperforge`.

---

## One-time manual setup (🧑 — needs accounts/secrets/repos Claude can't create)

### crates.io — Trusted Publishing (replaces the revoked token)
For **each** crate — `whisperforge-core`, `whisperforge-diarize`, `whisperforge-align`,
`whisperforge` — go to crates.io → the crate → **Settings → Trusted Publishing → Add**:
- Repository owner: `bevsxyz`  ·  Repository name: `WhisperForge`
- Workflow filename: `publish-crates.yml`  ·  Environment: `release`

Then in the GitHub repo, create an **Environment named `release`** (Settings →
Environments). No secret needed — auth is OIDC.

### GitHub repo secrets (Settings → Secrets and variables → Actions)
| Secret | Used by | Purpose |
|--------|---------|---------|
| `HOMEBREW_TAP_TOKEN` | `release.yml` | PAT (repo scope) with write access to the tap repo, so dist can push the formula |
| `NPM_TOKEN` | `release.yml` | npm automation token to publish the npm package (npm OIDC is a future option) |

### New repos to create
| Repo | Why |
|------|-----|
| `bevsxyz/homebrew-tap` | Generic personal Homebrew tap; dist auto-pushes `whisperforge.rb` here. Future tools can publish here too. Install: `brew install bevsxyz/tap/whisperforge` |
| `bevsxyz/scoop-bucket` | Scoop bucket; copy [`bucket/whisperforge.json`](bucket/whisperforge.json) into its `bucket/`. Install: `scoop bucket add whisperforge https://github.com/bevsxyz/scoop-bucket; scoop install whisperforge` |

### Per-channel accounts
| Channel | Account / action |
|---------|------------------|
| PyPI (pip) | PyPI account → configure a **Trusted Publisher** (repo + the pip publish workflow + environment). See [packaging/pypi/](packaging/pypi/) |
| npm | npm account/org + `NPM_TOKEN` (above) |
| AUR ×2 | AUR account + registered SSH key; initial push of [packaging/aur/whisperforge-bin](packaging/aur/whisperforge-bin/) and [whisperforge-git](packaging/aur/whisperforge-git/) to `ssh://aur@aur.archlinux.org/<pkg>.git` |
| winget | Fork `microsoft/winget-pkgs` + a PAT for `komac`; first submission goes through Microsoft moderation. See [packaging/winget/](packaging/winget/) |
| Chocolatey | chocolatey.org account + `CHOCO_API_KEY`; first submission is moderated. See [packaging/chocolatey/](packaging/chocolatey/) |
| aqua | Open a PR to `aquaproj/aqua-registry` with [packaging/aqua/registry.yaml](packaging/aqua/registry.yaml) (best regenerated with `aqua gr bevsxyz/WhisperForge`). Once merged, **mise / ubi / eget** work for free |

---

## Per-release templating (🧑/CI — checksums that can't be known until the build exists)

Some manifests embed a checksum that only exists after the release builds. Fill these
from the dist-emitted `<asset>.sha256` files (the marker is `<SHA256_FILLED_AT_RELEASE>`):

- **Chocolatey**: set `$checksum64` in [chocolateyinstall.ps1](packaging/chocolatey/tools/chocolateyinstall.ps1), then `choco pack` + `choco push`.
- **winget**: `komac` computes the `InstallerSha256` automatically when you run it against the release; or fill it manually in [the installer manifest](packaging/winget/bevsxyz.WhisperForge.installer.yaml).
- **AUR**: run `updpkgsums` in each package dir to replace `SKIP`, regenerate `.SRCINFO`
  (`makepkg --printsrcinfo > .SRCINFO`), then push.
- **Scoop / aqua / pip / Homebrew**: no manual checksum step — they fetch the digest from
  the `.sha256` URL (Scoop/aqua/pip) or are generated by dist (Homebrew).

> Most of these can be automated later (e.g. `komac update --submit`, the
> `KSXGitHub/github-actions-deploy-aur` action) once the channels are live.

## Channel status

| Channel | Mechanism | Scaffolded | Live after |
|---------|-----------|:---------:|------------|
| crates.io | `publish-crates.yml` (OIDC) | ✅ | trusted publishers configured |
| GitHub Release + installers | dist `release.yml` | ✅ | first `v*` tag |
| cargo-binstall | dist metadata | ✅ | first release (verify `cargo binstall whisperforge`) |
| Homebrew | dist → tap | ✅ | tap repo + `HOMEBREW_TAP_TOKEN` |
| npm (wrapper) | dist npm installer | ✅ | `NPM_TOKEN` |
| Scoop | [`bucket/whisperforge.json`](bucket/whisperforge.json) | ✅ | bucket repo |
| AUR (`-bin`, `-git`) | [packaging/aur/](packaging/aur/) | ✅ | AUR account + push |
| aqua (→ mise/ubi/eget) | [packaging/aqua/](packaging/aqua/) | ✅ | aqua-registry PR |
| pip (wrapper) | [packaging/pypi/](packaging/pypi/) | ✅ | PyPI trusted publisher + publish workflow |
| winget | [packaging/winget/](packaging/winget/) | ✅ | winget-pkgs PR (moderated) |
| Chocolatey | [packaging/chocolatey/](packaging/chocolatey/) | ✅ | choco account + push (moderated) |
| Docker/GHCR, Nix flake, deb/rpm | — | ⏳ tracked | future (issue #4) |
| Native PyO3/maturin + napi-rs libs | — | ⏳ tracked | follow-up issues |
