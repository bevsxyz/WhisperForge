//! CubeCL GPU STFT kernel for Whisper mel spectrogram computation.
//!
//! Implements a naive DFT (N=400 points) as a `#[cube]` kernel, mapping:
//! - One cube per STFT frame
//! - One unit (thread) per output frequency bin (N_FFT/2+1 = 201 threads)
//!
//! Threads cooperatively load windowed samples into shared memory, then each
//! computes its DFT bin independently. The result is the power spectrum
//! |X[k]|² — identical to what `compute_stft_magnitudes` produces via rustfft.
//!
//! # When to use
//!
//! Enable the `cubecl-stft` feature and call `compute_stft_power_gpu` with a
//! `ComputeClient<R>` from the active runtime. The returned `Vec<f32>` is a flat
//! `[n_frames, n_freqs]` power spectrum (row-major), equivalent to transposing the
//! `[n_freqs][n_frames]` output of `compute_stft_magnitudes` from `audio.rs`.
//!
//! # Dispatch wiring
//!
//! The GPU STFT is not yet wired into `compute_mel_spectrogram` because the CLI's
//! `Wgpu` backend is `Fusion<CubeBackend<WgpuRuntime,...>>` — the Fusion wrapper
//! prevents a generic `B: Backend` bound from seeing the inner Runtime. Full
//! wiring (switching CLI to bare `CubeBackend`) is deferred to Phase D.

use cubecl::prelude::*;

// ──────────────────────────────────────────────────────────────────────────────
// Kernel
// ──────────────────────────────────────────────────────────────────────────────

/// CubeCL DFT kernel: one cube per STFT frame, one unit per frequency bin.
///
/// `n_fft_smem` must equal `n_fft` at the call site — it is the comptime size
/// used to allocate the per-cube shared memory array. The runtime parameters
/// `n_fft`, `hop_length`, and `n_freqs` drive loop bounds and indexing so that
/// the generated shader contains loops (not unrolled code).
#[cube(launch_unchecked)]
fn stft_power_kernel<F: Float>(
    samples: &Array<F>,
    output: &mut Array<F>,
    n_fft: u32,
    hop_length: u32,
    n_freqs: u32,
    #[comptime] n_fft_smem: usize,
) {
    let freq_k = UNIT_POS;
    let frame_idx = CUBE_POS_X;
    let frame_start = frame_idx * hop_length;

    // Cooperatively load Hann-windowed samples into shared memory.
    // Each of the n_freqs threads loads ceil(n_fft / n_freqs) samples.
    let mut smem = SharedMemory::<F>::new(n_fft_smem);
    let two_pi = F::new(6.283185307179586_f32);
    let n_fft_f = F::cast_from(n_fft);

    let mut n = UNIT_POS;
    while n < n_fft {
        let sample = samples[(frame_start + n) as usize];
        // Hann window: w(n) = 0.5 * (1 - cos(2π·n/N))
        let angle = two_pi * F::cast_from(n) / n_fft_f;
        let window = (F::new(1.0_f32) - angle.cos()) * F::new(0.5_f32);
        smem[n as usize] = sample * window;
        n += n_freqs;
    }
    sync_cube();

    // DFT: X[k] = Σ_m smem[m] · e^{-2πi·k·m/N}  →  power = re² + im²
    let two_pi_k = two_pi * F::cast_from(freq_k) / n_fft_f;
    let mut re = F::new(0.0_f32);
    let mut im = F::new(0.0_f32);

    let mut m = 0u32;
    while m < n_fft {
        let phase = two_pi_k * F::cast_from(m);
        re += smem[m as usize] * phase.cos();
        im -= smem[m as usize] * phase.sin();
        m += 1u32;
    }

    output[(frame_idx * n_freqs + freq_k) as usize] = re * re + im * im;
}

// ──────────────────────────────────────────────────────────────────────────────
// Launcher
// ──────────────────────────────────────────────────────────────────────────────

/// Run the GPU STFT on `padded_samples` and return the power spectrum as a flat
/// `Vec<f32>` of shape `[n_frames * n_freqs]` (row-major, frame-major order).
///
/// `padded_samples` must already include the center=True reflect padding
/// (`n_fft/2` samples each side), exactly as produced by `compute_mel_spectrogram`.
///
/// The result is numerically equivalent (within floating-point rounding) to:
/// ```text
/// compute_stft_magnitudes(padded, n_fft, hop)[freq][frame]
///   == result[frame * n_freqs + freq]
/// ```
pub fn compute_stft_power_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    padded_samples: &[f32],
    n_fft: usize,
    hop_length: usize,
    n_frames: usize,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    // Upload padded audio samples to device.
    let samples_bytes: Vec<u8> = padded_samples
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let samples_handle = client.create_from_slice(&samples_bytes);

    // Allocate output buffer: n_frames × n_freqs f32 values.
    let n_output = n_frames * n_freqs;
    let output_handle = client.empty(n_output * std::mem::size_of::<f32>());

    let cube_dim = CubeDim::new_1d(n_freqs as u32);
    let cube_count = CubeCount::Static(n_frames as u32, 1, 1);

    unsafe {
        stft_power_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&samples_handle, padded_samples.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, n_output, 1),
            ScalarArg::new(n_fft as u32),
            ScalarArg::new(hop_length as u32),
            ScalarArg::new(n_freqs as u32),
            n_fft, // comptime n_fft_smem
        )
        .expect("STFT kernel launch failed");
    }

    // Read result back to host.
    let raw = client.read_one(output_handle);
    raw.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
