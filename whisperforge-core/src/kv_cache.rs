use anyhow::{Context, Result};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend, module::embedding};

use crate::model::{Whisper, qkv_attention};

/// Precomputed and growing KV cache for O(n) per-step decoder inference.
///
/// `cross_kv` holds the encoder cross-attention K,V projections for every decoder
/// layer. These are constant for the entire decoding of one audio chunk and are
/// computed once by [`KvCache::new`] before the decode loop.
///
/// `self_kv` holds the growing decoder self-attention K,V cache; one new row is
/// appended per [`forward_decoder_cached`] call.
pub struct KvCache<B: Backend> {
    /// Per-layer static cross-attention K,V from the encoder output.
    cross_kv: Vec<(Tensor<B, 3>, Tensor<B, 3>)>,
    /// Per-layer growing self-attention K,V; `None` until the first decode step.
    self_kv: Vec<Option<(Tensor<B, 3>, Tensor<B, 3>)>>,
    /// Number of tokens decoded so far (indexes into positional embedding).
    pub step: usize,
}

impl<B: Backend> KvCache<B> {
    /// Build a cache pre-populated with cross-attention K,V from `encoder_output`.
    ///
    /// The encoder K,V projections (`n_layers × 2` linear ops) are computed once
    /// here and reused at every subsequent decode step instead of being recomputed
    /// each time `forward_decoder` is called from scratch.
    ///
    /// # Performance
    ///
    /// Call this once per audio chunk. The cost is `n_decoder_layers × 2`
    /// matrix multiplications of shape `[n_encoder_frames × n_text_state]`.
    pub fn new(model: &Whisper<B>, encoder_output: Tensor<B, 3>) -> Self {
        let cross_kv = model
            .decoder
            .blocks
            .iter()
            .map(|block| {
                let ca = &block.cross_attn;
                let k = ca.key.forward(encoder_output.clone());
                let v = ca.value.forward(encoder_output.clone());
                (k, v)
            })
            .collect::<Vec<_>>();

        let n_layers = cross_kv.len();
        Self {
            cross_kv,
            self_kv: vec![None; n_layers],
            step: 0,
        }
    }
}

/// Decode one new token using the KV cache, updating the cache in-place.
///
/// Returns vocabulary logits for the new position as a flat `Vec<f32>` of
/// length `n_vocab`.
///
/// # Why this is faster than `forward_decoder`
///
/// `forward_decoder` processes the **full growing token sequence** at every step
/// (O(n) work per step → O(n²) total). This function processes only the **single
/// new token** and reads cached K,V for past positions (O(1) per step → O(n)
/// total). The cross-attention K,V are also cached across steps (constant).
///
/// # Correctness
///
/// The decoder causal mask is not needed when Q has sequence length 1 — a single
/// query position can attend to all cached K,V without violating causality.
pub fn forward_decoder_cached<B: Backend>(
    model: &Whisper<B>,
    token: u32,
    cache: &mut KvCache<B>,
    device: &B::Device,
) -> Result<Vec<f32>> {
    let decoder = &model.decoder;
    let step = cache.step;

    // Embed the single new token and add its positional encoding. [1, 1, n_text_state]
    let token_tensor: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![token as i32], [1, 1]), device);
    let mut x = embedding(decoder.token_embedding.val(), token_tensor)
        + decoder
            .positional_embedding
            .val()
            .slice([step..(step + 1)])
            .unsqueeze::<3>();

    for (layer_idx, block) in decoder.blocks.iter().enumerate() {
        // --- Self-attention with growing KV cache ---
        let x_norm = block.attn_ln.forward(x.clone());
        let sa = &block.attn;

        // Project only the new token into Q, K, V. [1, 1, n_text_state]
        let q = sa.query.forward(x_norm.clone());
        let k_new = sa.key.forward(x_norm.clone());
        let v_new = sa.value.forward(x_norm);

        // Concatenate new K,V with accumulated cache. [1, step+1, n_text_state]
        let (k_full, v_full) = match cache.self_kv[layer_idx].take() {
            Some((k_prev, v_prev)) => (
                Tensor::cat(vec![k_prev, k_new], 1),
                Tensor::cat(vec![v_prev, v_new], 1),
            ),
            None => (k_new, v_new),
        };
        cache.self_kv[layer_idx] = Some((k_full.clone(), v_full.clone()));

        // Attention: Q[1,1,D] × K[1,step+1,D] — no mask needed (Q len = 1).
        let sa_out = qkv_attention(q, k_full, v_full, None, sa.n_head);
        x = x + sa.out.forward(sa_out);

        // --- Cross-attention: reuse static K,V precomputed from encoder output ---
        let x_norm = block.cross_attn_ln.forward(x.clone());
        let ca = &block.cross_attn;
        let q = ca.query.forward(x_norm);
        let (k_cross, v_cross) = &cache.cross_kv[layer_idx];
        let ca_out = qkv_attention(q, k_cross.clone(), v_cross.clone(), None, ca.n_head);
        x = x + ca.out.forward(ca_out);

        // --- Feed-forward ---
        x = x.clone() + block.mlp.forward(block.mlp_ln.forward(x));
    }

    cache.step += 1;

    // Final layer norm + project to vocabulary. [1, 1, n_vocab] → [n_vocab]
    let x = decoder.ln.forward(x);
    let logits = x.matmul(decoder.token_embedding.val().transpose().unsqueeze::<3>());

    let [_, _, vocab_size] = logits.dims();
    logits
        .squeeze::<1>()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("logit extraction failed: {:?}", e))
        .with_context(|| format!("forward_decoder_cached step {step}, vocab_size={vocab_size}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Int, TensorData};
    use burn_flex::Flex;
    use burn_flex::FlexDevice;

    fn tiny_en_random() -> (crate::model::Whisper<Flex<f32>>, FlexDevice) {
        let device = FlexDevice;
        let config = crate::model::WhisperConfig::tiny_en();
        let model = config.init::<Flex<f32>>(&device);
        (model, device)
    }

    #[test]
    fn test_kv_cache_step_counter() {
        let (model, device) = tiny_en_random();
        let encoder_out = Tensor::<Flex<f32>, 3>::zeros([1, 1500, 384], &device);
        let mut cache = KvCache::new(&model, encoder_out);
        assert_eq!(cache.step, 0);
        forward_decoder_cached(&model, 50258u32, &mut cache, &device).unwrap();
        assert_eq!(cache.step, 1);
        forward_decoder_cached(&model, 50259u32, &mut cache, &device).unwrap();
        assert_eq!(cache.step, 2);
    }

    #[test]
    fn test_kv_cache_logit_shape() {
        let (model, device) = tiny_en_random();
        let encoder_out = Tensor::<Flex<f32>, 3>::zeros([1, 1500, 384], &device);
        let mut cache = KvCache::new(&model, encoder_out);
        let logits = forward_decoder_cached(&model, 50258u32, &mut cache, &device).unwrap();
        assert_eq!(logits.len(), 51864);
    }

    /// Verify that the KV-cached decoder produces numerically identical logits to
    /// `forward_decoder` for the same token sequence and encoder output.
    ///
    /// Uses random weights so no model files are needed. Tolerance 1e-4 covers
    /// float32 rounding from different operation order.
    #[test]
    fn test_kv_cache_matches_forward_decoder() {
        let (model, device) = tiny_en_random();
        let encoder_out =
            Tensor::<Flex<f32>, 3>::random([1, 1500, 384], Distribution::Normal(0.0, 0.1), &device);

        // Typical initial context: sot, en, transcribe, no_timestamps
        let init: [u32; 4] = [50258, 50259, 50359, 50363];

        // --- Original forward_decoder: full sequence in one call ---
        let token_tensor: Tensor<Flex<f32>, 2, Int> = Tensor::from_data(
            TensorData::new(init.iter().map(|&t| t as i32).collect::<Vec<_>>(), [1, 4]),
            &device,
        );
        let logits_full = model.forward_decoder(token_tensor, encoder_out.clone());
        let [b, seq, vocab] = logits_full.dims();
        let orig: Vec<f32> = logits_full
            .slice([0..b, (seq - 1)..seq, 0..vocab])
            .squeeze::<1>()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // --- KV-cached path: one token at a time ---
        let mut cache = KvCache::new(&model, encoder_out);
        let mut cached = Vec::new();
        for &tok in &init {
            cached = forward_decoder_cached(&model, tok, &mut cache, &device).unwrap();
        }

        assert_eq!(orig.len(), cached.len());
        let max_diff = orig
            .iter()
            .zip(cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "KV-cached logits diverge from forward_decoder by {max_diff:.2e} (expected < 1e-4)"
        );
    }
}
