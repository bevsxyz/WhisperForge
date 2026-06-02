# Chocolatey Package for WhisperForge

This directory contains the Chocolatey package definition and build scripts for WhisperForge.

## Building the Package

Navigate to this directory and run:

```powershell
choco pack whisperforge.nuspec
```

This generates a `.nupkg` file ready for publication.

## Publishing to Chocolatey.org

To push the package to the official Chocolatey community repository:

1. Ensure you have a Chocolatey.org account and have configured your API key:
   ```powershell
   choco apikey -k $env:CHOCO_API_KEY -source https://push.chocolatey.org/
   ```

2. Push the `.nupkg` file:
   ```powershell
   choco push whisperforge.0.5.0.nupkg --source https://push.chocolatey.org/
   ```

## Release Process Notes

- **Checksum Templating**: The `tools/chocolateyinstall.ps1` script contains a placeholder `<SHA256_FILLED_AT_RELEASE>` for the Windows x86_64 zip artifact's SHA256 checksum. At release time, this must be replaced with the actual checksum from the GitHub Release assets (e.g., from `whisperforge-x86_64-pc-windows-msvc.zip.sha256`).

- **First Submission**: The first release submission to Chocolatey.org goes through moderation, typically within 24 hours. Subsequent updates are published immediately.

- **Environment Setup**: Publishing requires the environment variable `CHOCO_API_KEY` to be set with your Chocolatey.org API token.
