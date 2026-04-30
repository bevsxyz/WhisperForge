use anyhow::Result;
use burn::tensor::{Tensor, backend::Backend};

/// Extract a speaker embedding from a Whisper encoder output tensor.
///
/// Mean-pools the time dimension of `encoder_output` (`[1, n_frames, d_model]`)
/// to produce a single `[d_model]` vector, then L2-normalises it. The result
/// is suitable as input to cosine-similarity-based speaker clustering.
///
/// # Why this works
///
/// The Whisper encoder is trained to produce frame-level representations that
/// capture acoustic content, including speaker characteristics. Mean-pooling
/// discards fine temporal structure but retains the global acoustic "fingerprint"
/// of the segment, which correlates with speaker identity well enough for
/// agglomerative clustering at the segment level.
pub fn extract_speaker_embedding<B: Backend>(encoder_output: Tensor<B, 3>) -> Result<Vec<f32>> {
    let [_, n_frames, d_model] = encoder_output.dims();

    let flat: Vec<f32> = encoder_output
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("extract_speaker_embedding: tensor read failed: {:?}", e))?;

    // Mean-pool over the time dimension.
    let mut embedding = vec![0.0f32; d_model];
    for frame in 0..n_frames {
        for dim in 0..d_model {
            embedding[dim] += flat[frame * d_model + dim];
        }
    }
    let scale = n_frames as f32;
    for v in &mut embedding {
        *v /= scale;
    }

    // L2-normalise so cosine similarity equals the dot product.
    let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in &mut embedding {
            *v /= norm;
        }
    }

    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArrayDevice;

    #[test]
    fn test_embedding_is_unit_norm() {
        let device = NdArrayDevice::default();
        let enc: Tensor<NdArray<f32>, 3> =
            Tensor::from_data(TensorData::new(vec![1.0f32; 1 * 8 * 4], [1, 8, 4]), &device);
        let emb = extract_speaker_embedding(enc).unwrap();
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "embedding norm={norm}");
    }

    #[test]
    fn test_embedding_dimension_matches_d_model() {
        let device = NdArrayDevice::default();
        let d_model = 384usize;
        let enc: Tensor<NdArray<f32>, 3> = Tensor::zeros([1, 1500, d_model], &device);
        let emb = extract_speaker_embedding(enc).unwrap();
        assert_eq!(emb.len(), d_model);
    }

    #[test]
    fn test_identical_encoder_outputs_produce_identical_embeddings() {
        let device = NdArrayDevice::default();
        let data = (0..384).map(|i| i as f32).collect::<Vec<_>>();
        let flat: Vec<f32> = data.iter().cycle().take(1500 * 384).copied().collect();
        let enc: Tensor<NdArray<f32>, 3> =
            Tensor::from_data(TensorData::new(flat, [1, 1500, 384]), &device);
        let enc2: Tensor<NdArray<f32>, 3> = enc.clone();
        let e1 = extract_speaker_embedding(enc).unwrap();
        let e2 = extract_speaker_embedding(enc2).unwrap();
        let max_diff = e1
            .iter()
            .zip(e2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6);
    }
}
