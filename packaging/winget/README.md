# Windows Package Manager (winget) Manifest

This directory contains the winget manifest scaffolding for WhisperForge 0.5.0, following the Microsoft winget-pkgs 1.6 manifest schema.

## Files

- `bevsxyz.WhisperForge.yaml` — version manifest (PackageVersion, DefaultLocale)
- `bevsxyz.WhisperForge.locale.en-US.yaml` — default locale manifest (description, publisher, tags)
- `bevsxyz.WhisperForge.installer.yaml` — installer manifest (download URLs, nested portable ZIP, SHA256)

## Before Submission

1. **Fill in the SHA256 placeholder**: Replace `<SHA256_FILLED_AT_RELEASE>` in `bevsxyz.WhisperForge.installer.yaml` with the actual SHA256 of the Windows x64 ZIP release asset (`whisperforge-x86_64-pc-windows-msvc.zip`).
   - Download the `.sha256` file from the GitHub release
   - Or compute: `sha256sum whisperforge-x86_64-pc-windows-msvc.zip`

2. **Validate the manifests** locally (optional):
   ```bash
   winget validate .
   ```

## Submission

Submit to [microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs) using one of these methods:

### Option A: Manual PR (Git)

1. Fork https://github.com/microsoft/winget-pkgs
2. Create a branch `add-whisperforge-0.5.0`
3. Copy the three manifest files to `manifests/b/bevsxyz/WhisperForge/0.5.0/`
4. Commit and push
5. Open a PR; Microsoft will auto-validate and merge within 1–2 weeks

### Option B: komac (automated, requires Rust)

```bash
cargo install komac
komac new \
  --package-identifier "bevsxyz.WhisperForge" \
  --package-version "0.5.0" \
  --urls "https://github.com/bevsxyz/WhisperForge/releases/download/v0.5.0/whisperforge-x86_64-pc-windows-msvc.zip" \
  --installer-type "zip" \
  --nested-installer-type "portable" \
  --nested-installer-files "wforge.exe" \
  --portable-command-alias "wforge"
```

(komac will compute the SHA256 and file paths automatically.)

### Option C: wingetcreate (automated, requires .NET)

```bash
wingetcreate new bevsxyz.WhisperForge \
  --urls "https://github.com/bevsxyz/WhisperForge/releases/download/v0.5.0/whisperforge-x86_64-pc-windows-msvc.zip"
```

## Installation (after merge)

Once the manifests are merged into microsoft/winget-pkgs, users can install via:

```bash
winget install bevsxyz.WhisperForge
wforge --help
```

## Schema Reference

- Manifest version: 1.6.0
- Installer type: zip (portable)
  - Binary `wforge.exe` is copied directly into PATH (no traditional NSIS/MSI installation)
  - Uninstall is manual (delete the binary) — typical for portable installs in winget
- SHA256 verification: enabled

For more details, see [Microsoft winget docs](https://docs.microsoft.com/en-us/windows/package-manager/package/).
