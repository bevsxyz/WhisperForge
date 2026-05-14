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
//! Wired into `compute_mel_spectrogram_wgpu` in `audio.rs` via
//! `CubeBackend<WgpuRuntime,f32,i32,u32>` in the CLI. The Fusion-wrapped `Wgpu`
//! alias cannot be used here because the Fusion wrapper hides the inner Runtime.

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
    let two_pi = F::new(std::f32::consts::TAU);
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
            ArrayArg::from_raw_parts(samples_handle, padded_samples.len()),
            ArrayArg::from_raw_parts(output_handle.clone(), n_output),
            n_fft as u32,
            hop_length as u32,
            n_freqs as u32,
            n_fft, // comptime n_fft_smem
        );
    }

    // Read result back to host.
    let raw = client.read_one_unchecked(output_handle);
    raw.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_wgpu::WgpuRuntime;
    use cubecl::prelude::Runtime;
    use rustfft::{FftPlanner, num_complex::Complex};
    use std::f32::consts::PI;

    /// Compute CPU DFT power spectrum matching the GPU kernel exactly.
    /// Returns flat [n_frames, n_freqs] (frame-major) to match GPU layout.
    fn cpu_stft_power(
        samples: &[f32],
        n_fft: usize,
        hop_length: usize,
        n_frames: usize,
    ) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let mut out = vec![0.0f32; n_frames * n_freqs];
        let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
        for frame in 0..n_frames {
            let start = frame * hop_length;
            for i in 0..n_fft {
                buf[i] = Complex::new(samples[start + i] * window[i], 0.0);
            }
            fft.process(&mut buf);
            for k in 0..n_freqs {
                out[frame * n_freqs + k] = buf[k].norm_sqr();
            }
        }
        out
    }

    #[test]
    fn test_gpu_stft_matches_cpu() {
        let device = burn_wgpu::WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        // 1 second of 440 Hz sine at 16 kHz, pre-padded (simulate center=True padding).
        let n_samples = 16000 + 400; // 400 = n_fft reflect-padding already applied
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let n_fft = 400;
        let hop = 160;
        let n_freqs = n_fft / 2 + 1;
        let n_frames = (samples.len() - n_fft) / hop + 1;

        let gpu = compute_stft_power_gpu(&client, &samples, n_fft, hop, n_frames);
        let cpu = cpu_stft_power(&samples, n_fft, hop, n_frames);

        assert_eq!(gpu.len(), cpu.len(), "output length mismatch");

        let mut max_abs_err = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            let err = (g - c).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }
        // Floating-point DFT vs FFT may differ slightly; 1e-3 is generous.
        assert!(
            max_abs_err < 1e-3,
            "GPU/CPU STFT max abs error {max_abs_err} exceeds 1e-3 — outputs disagree"
        );

        // Verify frame-major layout: check a known bin for the 440 Hz sine.
        // At 16 kHz, 440 Hz ≈ bin 11 (440 * 400 / 16000 = 11).
        let bin_440 = (440.0 * n_fft as f32 / 16000.0).round() as usize;
        let frame_mid = n_frames / 2;
        let power_at_440 = gpu[frame_mid * n_freqs + bin_440];
        assert!(
            power_at_440 > 1.0,
            "Expected significant power at 440 Hz bin {bin_440}, got {power_at_440}"
        );
    }
}
