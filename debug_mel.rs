use anyhow::Result;
use burn_ndarray::NdArrayDevice;
use whisperforge_core::{AudioData, audio};

fn main() -> Result<()> {
    let device = NdArrayDevice::default();

    // Create simple sine wave
    let sample_rate = 16000;
    let duration = 1.0;
    let freq = 440.0;
    let n_samples = (sample_rate as f32 * duration) as usize;

    let samples: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    let audio = AudioData::new(samples, sample_rate, 1);

    println!("Audio stats:");
    println!("  Samples: {}", audio.samples.len());
    println!(
        "  Mean: {:.4}",
        audio.samples.iter().sum::<f32>() / audio.samples.len() as f32
    );
    println!(
        "  Min: {:.4}",
        audio.samples.iter().cloned().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max: {:.4}",
        audio
            .samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Compute mel spectrogram
    let mel_spec = audio::compute_mel_spectrogram(
        &audio, 400, // n_fft
        160, // hop_length
        80,  // n_mels
        &device,
    )?;

    let mel_data = mel_spec.to_data();
    let mel_values: Vec<f32> = mel_data.to_vec().unwrap();

    println!("\nMel spectrogram stats:");
    println!("  Shape: {:?}", mel_spec.dims());
    println!("  Total values: {}", mel_values.len());

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0;

    for &val in &mel_values {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
        sum += val;
    }

    let mean = sum / mel_values.len() as f32;

    println!("  Min: {:.4}", min_val);
    println!("  Max: {:.4}", max_val);
    println!("  Mean: {:.4}", mean);
    println!(
        "  Std: {:.4}",
        (mel_values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / mel_values.len() as f32)
            .sqrt()
    );

    // Check for NaN or inf values
    let mut nan_count = 0;
    let mut inf_count = 0;

    for &val in &mel_values {
        if val.is_nan() {
            nan_count += 1;
        }
        if val.is_infinite() {
            inf_count += 1;
        }
    }

    if nan_count > 0 {
        println!("  NaN values: {}", nan_count);
    }
    if inf_count > 0 {
        println!("  Inf values: {}", inf_count);
    }

    Ok(())
}
