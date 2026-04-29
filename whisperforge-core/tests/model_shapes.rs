/// Verify that base and small model architectures initialise correctly and
/// produce the expected output shapes from a synthetic mel spectrogram.
///
/// These tests do NOT load real weights — they only exercise the model
/// graph and confirm that the config dimensions are self-consistent.
/// Actual weight-loading tests for base/small are gated on model files
/// being present and run with --ignored (they take several minutes on CPU).
///
/// medium and large-v2 are excluded from CI (too large for automated infra).
use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_ndarray::NdArrayDevice;
use whisperforge_core::WhisperConfig;

type Backend = NdArray<f32>;

fn encoder_output_shape(config: &WhisperConfig, device: &NdArrayDevice) -> [usize; 3] {
    let model = config.init::<Backend>(device);
    let mel = Tensor::<Backend, 3>::zeros([1, 80, 3000], device);
    model.forward_encoder(mel).dims()
}

#[test]
fn test_tiny_en_encoder_output_shape() {
    let device = NdArrayDevice::default();
    let config = WhisperConfig::tiny_en();
    let out = encoder_output_shape(&config, &device);
    // [batch, n_audio_ctx, n_audio_state] — conv stem halves time from 3000→1500
    assert_eq!(out, [1, 1500, 384]);
}

#[test]
#[ignore = "slow: initialises 6-layer 512-dim base model on NdArray CPU (~5 min)"]
fn test_base_encoder_output_shape() {
    let device = NdArrayDevice::default();
    let config = WhisperConfig::base();
    let out = encoder_output_shape(&config, &device);
    assert_eq!(out, [1, 1500, 512]);
}

#[test]
#[ignore = "slow: initialises 12-layer 768-dim small model on NdArray CPU (~20 min)"]
fn test_small_encoder_output_shape() {
    let device = NdArrayDevice::default();
    let config = WhisperConfig::small();
    let out = encoder_output_shape(&config, &device);
    assert_eq!(out, [1, 1500, 768]);
}

#[test]
fn test_model_configs_have_distinct_state_dims() {
    let tiny = WhisperConfig::tiny_en();
    let base = WhisperConfig::base();
    let small = WhisperConfig::small();
    assert_ne!(
        tiny.audio_encoder_config.n_audio_state,
        base.audio_encoder_config.n_audio_state
    );
    assert_ne!(
        base.audio_encoder_config.n_audio_state,
        small.audio_encoder_config.n_audio_state
    );
}
