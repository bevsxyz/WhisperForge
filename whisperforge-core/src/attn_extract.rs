use burn::tensor::{Int, Tensor, activation::softmax, backend::Backend, module::embedding};

use crate::model::Whisper;

/// Runs the decoder forward pass identically to `Whisper::forward_decoder` but
/// additionally captures cross-attention weights for the **last query position**,
/// averaged over all decoder layers and heads.
///
/// Returns `(logits [1, seq, vocab], frame_weights [n_encoder_frames])`.
///
/// Convert to an audio timestamp:
/// ```text
/// timestamp_s = argmax(frame_weights) * 2 * hop_length / sample_rate
/// ```
/// The factor of 2 accounts for the encoder's stride-2 Conv1d that halves the
/// 3000-frame mel spectrogram to 1500 encoder frames.
pub fn forward_decoder_with_cross_attn<B: Backend>(
    model: &Whisper<B>,
    tokens: Tensor<B, 2, Int>,
    encoder_output: Tensor<B, 3>,
) -> (Tensor<B, 3>, Vec<f32>) {
    let decoder = &model.decoder;
    let [_, seq_len] = tokens.dims();
    let n_encoder_frames = encoder_output.dims()[1];

    #[allow(clippy::single_range_in_vec_init)]
    let mut x = embedding(decoder.token_embedding.val(), tokens)
        + decoder
            .positional_embedding
            .val()
            .slice([0..seq_len])
            .unsqueeze::<3>();

    let mask = decoder.mask.val();
    let mut layer_frame_weights: Vec<Vec<f32>> = Vec::with_capacity(decoder.blocks.len());

    for block in &decoder.blocks {
        // Self-attention (mirrors ResidualDecoderAttentionBlock::forward)
        let self_out = block
            .attn
            .forward(block.attn_ln.forward(x.clone()), Some(mask.clone()));
        x = x + self_out;

        // Cross-attention — replicated to capture the softmax weight matrix
        let x_norm = block.cross_attn_ln.forward(x.clone());
        let ca = &block.cross_attn;
        let q = ca.query.forward(x_norm);
        let k = ca.key.forward(encoder_output.clone());
        let v = ca.value.forward(encoder_output.clone());

        let (ca_out, w) = cross_attn_softmax_and_out(q, k, v, ca.n_head);

        // w: [1, n_head, q_len, n_encoder_frames]
        // Average over heads; keep only the last query position.
        let [_, n_head, q_len, _] = w.dims();
        let w_slice: Vec<f32> = w
            .slice([0..1, 0..n_head, (q_len - 1)..q_len, 0..n_encoder_frames])
            .into_data()
            .to_vec()
            .unwrap_or_else(|_| vec![0.0; n_head * n_encoder_frames]);

        let mut layer_avg = vec![0.0f32; n_encoder_frames];
        for h in 0..n_head {
            for f in 0..n_encoder_frames {
                layer_avg[f] += w_slice[h * n_encoder_frames + f];
            }
        }
        for val in &mut layer_avg {
            *val /= n_head as f32;
        }
        layer_frame_weights.push(layer_avg);

        x = x + ca_out;

        // MLP
        let mlp_out = block.mlp.forward(block.mlp_ln.forward(x.clone()));
        x = x + mlp_out;
    }

    // Final layer norm + project to vocabulary logits
    let x = decoder.ln.forward(x);
    let logits = x.matmul(decoder.token_embedding.val().transpose().unsqueeze::<3>());

    // Average cross-attention weights over all layers
    let n_layers = layer_frame_weights.len();
    let mut avg_weights = vec![0.0f32; n_encoder_frames];
    for layer_avg in &layer_frame_weights {
        for (f, &w) in layer_avg.iter().enumerate() {
            avg_weights[f] += w;
        }
    }
    if n_layers > 0 {
        for val in &mut avg_weights {
            *val /= n_layers as f32;
        }
    }

    (logits, avg_weights)
}

/// Compute scaled dot-product cross-attention, returning both the context output
/// and the full softmax weight matrix.
///
/// Inputs: q [batch, q_len, state], k/v [batch, kv_len, state].
/// Returns: (output [batch, q_len, state], weights [batch, n_head, q_len, kv_len]).
fn cross_attn_softmax_and_out<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    n_head: usize,
) -> (Tensor<B, 3>, Tensor<B, 4>) {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_kv, _] = k.dims();
    let n_hstate = n_state / n_head;
    let scale = (n_state as f64 / n_head as f64).powf(-0.25);

    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .swap_dims(1, 2)
        * scale;
    let k = k
        .reshape([n_batch, n_kv, n_head, n_hstate])
        .swap_dims(1, 2)
        .transpose()
        * scale;
    let v = v.reshape([n_batch, n_kv, n_head, n_hstate]).swap_dims(1, 2);

    let qk = q.matmul(k); // [batch, n_head, q_len, kv_len]
    let w = softmax(qk, 3); // [batch, n_head, q_len, kv_len]
    let o = w.clone().matmul(v).swap_dims(1, 2).flatten(2, 3); // [batch, q_len, state]

    (o, w)
}
