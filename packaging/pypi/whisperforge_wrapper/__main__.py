"""
whisperforge binary wrapper: downloads and executes the prebuilt wforge binary.

On first run, this script:
1. Detects your platform and architecture
2. Downloads the corresponding prebuilt binary from the GitHub release
3. Verifies its SHA256 checksum
4. Extracts it to a per-user cache directory
5. Executes it with all forwarded arguments

Subsequent runs reuse the cached binary.
"""

import hashlib
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


__version__ = "0.5.0"

# Map Python's platform identifiers to Rust target triples used in cargo-dist
PLATFORM_MAP = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu.tar.xz",
    ("Linux", "aarch64"): "aarch64-unknown-linux-gnu.tar.xz",
    ("Linux", "arm64"): "aarch64-unknown-linux-gnu.tar.xz",
    ("Darwin", "x86_64"): "x86_64-apple-darwin.tar.xz",
    ("Darwin", "arm64"): "aarch64-apple-darwin.tar.xz",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc.zip",
    ("Windows", "x86_64"): "x86_64-pc-windows-msvc.zip",
}

RELEASE_URL_BASE = f"https://github.com/bevsxyz/WhisperForge/releases/download/v{__version__}"
BINARY_NAME = "wforge.exe" if sys.platform == "win32" else "wforge"


def get_cache_dir() -> Path:
    """
    Return the per-user cache directory for WhisperForge binaries.

    Uses platform-appropriate defaults:
    - Linux/macOS: ~/.cache/whisperforge/
    - Windows: %APPDATA%/whisperforge/cache/
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "whisperforge" / "cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "whisperforge"


def get_platform_asset() -> str:
    """
    Determine the correct asset filename for this platform and architecture.

    Raises:
        RuntimeError: If the platform/architecture combination is not supported.
    """
    system = platform.system()
    machine = platform.machine()

    key = (system, machine)
    if key not in PLATFORM_MAP:
        raise RuntimeError(
            f"Unsupported platform: {system} {machine}. "
            f"Supported combinations: {', '.join(f'{s} {m}' for s, m in PLATFORM_MAP.keys())}"
        )

    return PLATFORM_MAP[key]


def download_file(url: str) -> bytes:
    """
    Download a file from a URL and return its contents as bytes.

    Raises:
        urllib.error.URLError: If the download fails.
    """
    with urllib.request.urlopen(url) as response:
        return response.read()


def verify_sha256(data: bytes, expected_hex: str) -> None:
    """
    Verify that data matches the expected SHA256 checksum.

    Raises:
        RuntimeError: If the checksum does not match.
    """
    actual_hex = hashlib.sha256(data).hexdigest()
    # dist writes checksums in `sha256sum` format: "<hexdigest>  <filename>".
    # Take just the first whitespace-delimited token (the digest).
    expected_hex = expected_hex.split()[0] if expected_hex.split() else expected_hex
    if actual_hex != expected_hex.strip().lower():
        raise RuntimeError(
            f"SHA256 checksum mismatch for downloaded binary. "
            f"Expected: {expected_hex}, got: {actual_hex}"
        )


def extract_binary(archive_data: bytes, asset_name: str) -> bytes:
    """
    Extract the wforge binary from a tar.xz or zip archive.

    Returns:
        The raw binary data.

    Raises:
        RuntimeError: If extraction fails or the binary is not found.
    """
    if asset_name.endswith(".tar.xz"):
        # tar.xz (Linux, macOS)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r|xz") as tar:
                for member in tar:
                    if member.name.endswith(BINARY_NAME) or member.name == BINARY_NAME:
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            raise RuntimeError(f"Failed to extract {BINARY_NAME} from tar.xz")
                        return extracted.read()
        except Exception as e:
            raise RuntimeError(f"Failed to extract tar.xz: {e}")
        raise RuntimeError(f"Binary {BINARY_NAME} not found in tar.xz archive")

    elif asset_name.endswith(".zip"):
        # zip (Windows)
        try:
            with zipfile.ZipFile(io.BytesIO(archive_data)) as zf:
                for name in zf.namelist():
                    if name.endswith(BINARY_NAME) or name == BINARY_NAME:
                        return zf.read(name)
        except Exception as e:
            raise RuntimeError(f"Failed to extract zip: {e}")
        raise RuntimeError(f"Binary {BINARY_NAME} not found in zip archive")

    else:
        raise RuntimeError(f"Unsupported archive format: {asset_name}")


def ensure_binary_cached() -> Path:
    """
    Ensure the wforge binary is present in the cache directory.

    If not present, downloads it from the GitHub release, verifies the checksum,
    and extracts it.

    Returns:
        Path to the cached binary.

    Raises:
        RuntimeError: If download, verification, or extraction fails.
    """
    asset_name = get_platform_asset()
    cache_dir = get_cache_dir()
    binary_path = cache_dir / BINARY_NAME

    # If binary already cached, return it
    if binary_path.exists() and binary_path.is_file():
        return binary_path

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download the binary archive
    asset_url = f"{RELEASE_URL_BASE}/{asset_name}"
    sha256_url = f"{RELEASE_URL_BASE}/{asset_name}.sha256"

    try:
        print(f"Downloading {asset_name}...", file=sys.stderr)
        archive_data = download_file(asset_url)

        print(f"Verifying SHA256 checksum...", file=sys.stderr)
        sha256_data = download_file(sha256_url).decode("utf-8")
        verify_sha256(archive_data, sha256_data)

        print(f"Extracting binary...", file=sys.stderr)
        binary_data = extract_binary(archive_data, asset_name)

        # Write to cache
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tmp:
            tmp.write(binary_data)
            tmp_path = Path(tmp.name)

        # Atomic rename
        tmp_path.replace(binary_path)
        binary_path.chmod(0o755)

        print(f"Cached at {binary_path}", file=sys.stderr)

    except Exception as e:
        raise RuntimeError(f"Failed to download and cache binary: {e}")

    return binary_path


def main() -> None:
    """
    Main entry point: ensure the binary is cached, then execute it with all arguments.
    """
    try:
        binary_path = ensure_binary_cached()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Forward all arguments and execute
    try:
        result = subprocess.run([str(binary_path)] + sys.argv[1:])
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: Failed to execute {binary_path}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
