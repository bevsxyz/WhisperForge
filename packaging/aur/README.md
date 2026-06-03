# AUR Packages for WhisperForge

This directory contains packaging scaffolding for Arch User Repository (AUR) submission.

## Package Overview

Two complementary packages are provided:

- **whisperforge-bin** — prebuilt release binaries (recommended for end users)
  - Fetches x86_64 or aarch64 tarballs from GitHub releases
  - Fast installation, no compilation required
  
- **whisperforge-git** — VCS package for development/latest main branch
  - Clones from the main repository and builds from source
  - Useful for testing unreleased features or contributing to development

Both packages provide the `whisperforge` virtual name (install either one) and install the `wforge` binary to `/usr/bin/wforge`, plus a `wf` shim (in binary package only).

## Prerequisites for AUR Publishing

- An Arch User Repository account (register at https://aur.archlinux.org/)
- SSH key registered and configured for AUR (e.g., `~/.ssh/aur`)
- Git configured with your AUR account email

## Initial Setup & Push

### For whisperforge-bin

```bash
git clone ssh://aur@aur.archlinux.org/whisperforge-bin.git
cd whisperforge-bin
cp /path/to/WhisperForge/packaging/aur/whisperforge-bin/PKGBUILD .
cp /path/to/WhisperForge/packaging/aur/whisperforge-bin/.SRCINFO .
git add PKGBUILD .SRCINFO
git commit -m "Initial commit: whisperforge-bin 0.5.0"
git push
```

### For whisperforge-git

```bash
git clone ssh://aur@aur.archlinux.org/whisperforge-git.git
cd whisperforge-git
cp /path/to/WhisperForge/packaging/aur/whisperforge-git/PKGBUILD .
cp /path/to/WhisperForge/packaging/aur/whisperforge-git/.SRCINFO .
git add PKGBUILD .SRCINFO
git commit -m "Initial commit: whisperforge-git"
git push
```

## Maintenance

### Updating SHA256 Checksums (whisperforge-bin only)

Before each release, populate the checksums in `whisperforge-bin/PKGBUILD`:

```bash
cd whisperforge-bin
updpkgsums
git add PKGBUILD .SRCINFO
git commit -m "chore: update checksums for v0.5.0"
git push
```

Alternatively, checksums can be computed offline:

```bash
sha256sum whisperforge-x86_64-unknown-linux-gnu.tar.xz
sha256sum whisperforge-aarch64-unknown-linux-gnu.tar.xz
```

And pasted into the `PKGBUILD` `sha256sums_x86_64` and `sha256sums_aarch64` arrays.

### Version Bumps

For **whisperforge-bin** (release track):
1. Bump `pkgver` in PKGBUILD (e.g., 0.5.0 → 0.6.0)
2. Reset `pkgrel=1`
3. Update checksums via `updpkgsums`
4. Regenerate `.SRCINFO`: `makepkg --printsrcinfo > .SRCINFO`
5. Commit and push

For **whisperforge-git** (VCS track):
- No manual version bump needed; `pkgver()` function auto-detects from git tags
- Only `pkgrel` is incremented if PKGBUILD changes (e.g., new dependencies, build flags)
- Regenerate `.SRCINFO` and push

## CI/CD Integration (Optional)

To automate AUR pushes on release, consider using [KSXGitHub/github-actions-deploy-aur](https://github.com/KSXGitHub/github-actions-deploy-aur):

```yaml
# .github/workflows/publish-aur.yml (example)
on:
  push:
    tags:
      - 'v*'

jobs:
  publish-aur:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: KSXGitHub/github-actions-deploy-aur@master
        with:
          pkgname: whisperforge-bin
          pkgbuild: packaging/aur/whisperforge-bin/PKGBUILD
          ssh_private_key: ${{ secrets.AUR_SSH_PRIVATE_KEY }}
          ssh_known_hosts: ${{ secrets.AUR_SSH_KNOWN_HOSTS }}
          git_username: bevsxyz
          git_email: bevan.stanely@accenture.com
          commit_message: "Release v${{ github.ref_name }}"
```

Store the AUR SSH private key in GitHub Actions Secrets as `AUR_SSH_PRIVATE_KEY`.

## Testing Locally

To test a package build before submission:

```bash
cd whisperforge-bin  # or whisperforge-git
makepkg -si  # Installs and tests the package
```

## Troubleshooting

- **PGP Key Errors**: Ensure the `validpgpkeys` array in PKGBUILD is set (or omitted for trust-all).
- **Build Fails in -git**: Ensure Rust/Cargo are installed and the repo builds locally with `cargo build --release --locked -p whisperforge --features gpu`.
- **Network Issues on Push**: Verify SSH key is configured: `ssh -T aur@aur.archlinux.org` should succeed.

## Further Reading

- [AUR Submission Guidelines](https://wiki.archlinux.org/title/AUR_submission_guidelines)
- [AUR User Guidelines](https://wiki.archlinux.org/title/AUR_user_guidelines)
- [PKGBUILD Reference](https://wiki.archlinux.org/title/PKGBUILD)
