# This script is a release-templated scaffold. The checksum variable below is filled at release time per the actual artifact.

$ErrorActionPreference = 'Stop'

$url64bit = "https://github.com/bevsxyz/WhisperForge/releases/download/v0.5.0/whisperforge-x86_64-pc-windows-msvc.zip"
$checksum64 = '<SHA256_FILLED_AT_RELEASE>'
$checksumType64 = 'sha256'

# Install the zip package into the Chocolatey tools directory.
# Chocolatey will automatically create shims for any .exe files found in $toolsDir.
Install-ChocolateyZipPackage `
  -PackageName 'whisperforge' `
  -Url64bit $url64bit `
  -Checksum64 $checksum64 `
  -ChecksumType64 $checksumType64 `
  -UnzipLocation $(Get-ToolsLocation)
