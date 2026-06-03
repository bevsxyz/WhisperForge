# whisperforge (PyPI binary wrapper)

Fast GPU-accelerated speech-to-text CLI with streaming, quantization, speaker diarization, and multilingual support.

## Installation

```bash
pip install whisperforge
```

This package downloads the prebuilt `wforge` binary for your platform on first run (no compilation required).

## Usage

```bash
wforge --help
wforge transcribe audio.wav -m tiny.en
wforge stream --model tiny.en
wforge pull base
wforge list models
```

For detailed usage, see the [main WhisperForge documentation](https://github.com/bevsxyz/WhisperForge).

## What's inside

This PyPI package is a **thin binary wrapper** that:
1. Detects your OS and CPU architecture
2. Downloads the corresponding prebuilt binary from the GitHub release on first run
3. Caches it in your user cache directory (`~/.cache/whisperforge/` on Linux/macOS, `%APPDATA%/whisperforge/cache/` on Windows)
4. Forwards all CLI arguments to the binary

No Rust compiler or build tools are required — the binary runs immediately after download.

## Future: PyO3 library

A native Python library (PyO3/maturin) for importing and calling WhisperForge's inference engine from Python is planned separately. This wrapper focuses on CLI distribution for now.

## Publishing to PyPI

This package uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) via GitHub Actions OIDC tokens. The CI/CD pipeline handles uploads automatically on release.

## License

MIT

## Links

- GitHub: https://github.com/bevsxyz/WhisperForge
- Issues: https://github.com/bevsxyz/WhisperForge/issues
