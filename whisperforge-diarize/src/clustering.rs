/// Cosine similarity between two vectors.
///
/// For L2-normalised inputs this equals the dot product. Inputs need not be
/// normalised — the function computes the general cosine similarity formula.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Agglomerative single-linkage clustering over speaker embeddings.
///
/// Each embedding starts as its own cluster. At each step the pair of active
/// clusters with the highest cosine similarity is merged (if it exceeds
/// `threshold`). The similarity of a merged cluster to any third cluster is
/// the *maximum* of the two constituent clusters' similarities (single-linkage).
///
/// Returns a `Vec<usize>` of cluster labels, one per embedding, numbered in
/// order of first appearance (the embedding that appears earliest in the slice
/// gets label 0, and so on).
pub fn cluster_embeddings(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return vec![];
    }

    // parent[i] == i  ⟺  i is an active cluster representative.
    // After merging j into i, parent[j] = i.
    let mut parent: Vec<usize> = (0..n).collect();

    // Flat upper-triangle similarity matrix (indexed [i*n + j]).
    let mut sim = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = cosine_similarity(&embeddings[i], &embeddings[j]);
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }

    loop {
        let active: Vec<usize> = (0..n).filter(|&i| parent[i] == i).collect();
        if active.len() <= 1 {
            break;
        }

        // Find active pair with maximum similarity.
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;
        for &i in &active {
            for &j in &active {
                if i < j && sim[i * n + j] > best_sim {
                    best_sim = sim[i * n + j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_sim < threshold {
            break;
        }

        // Merge best_j into best_i (single-linkage: take max similarity).
        for &k in &active {
            if k == best_i || k == best_j {
                continue;
            }
            let merged = sim[best_i * n + k].max(sim[best_j * n + k]);
            sim[best_i * n + k] = merged;
            sim[k * n + best_i] = merged;
        }
        parent[best_j] = best_i;
    }

    // Resolve canonical representative via path compression.
    let canonical: Vec<usize> = (0..n)
        .map(|mut i| {
            while parent[i] != i {
                i = parent[i];
            }
            i
        })
        .collect();

    // Renumber labels in order of first occurrence.
    let mut label_map = std::collections::HashMap::<usize, usize>::new();
    let mut next = 0usize;
    canonical
        .iter()
        .map(|&c| {
            *label_map.entry(c).or_insert_with(|| {
                let l = next;
                next += 1;
                l
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_single_embedding_returns_label_zero() {
        let labels = cluster_embeddings(&[unit(vec![1.0, 0.0])], 0.7);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_identical_embeddings_cluster_together() {
        let a = unit(vec![1.0, 0.0]);
        let embeddings = vec![a.clone(), a.clone(), a.clone()];
        let labels = cluster_embeddings(&embeddings, 0.7);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_orthogonal_embeddings_stay_separate() {
        // Cosine similarity of orthogonal vectors is 0 — below any positive threshold.
        let a = unit(vec![1.0, 0.0]);
        let b = unit(vec![0.0, 1.0]);
        let labels = cluster_embeddings(&[a, b], 0.5);
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn test_alternating_two_speakers() {
        // A, B, A, B — A and B are orthogonal so never merge with each other.
        let a = unit(vec![1.0, 0.0]);
        let b = unit(vec![0.0, 1.0]);
        let embeddings = vec![a.clone(), b.clone(), a.clone(), b.clone()];
        let labels = cluster_embeddings(&embeddings, 0.7);
        assert_eq!(labels[0], labels[2], "both A segments should share a label");
        assert_eq!(labels[1], labels[3], "both B segments should share a label");
        assert_ne!(labels[0], labels[1], "A and B should have different labels");
    }

    #[test]
    fn test_labels_start_at_zero_in_first_occurrence_order() {
        let a = unit(vec![1.0, 0.0]);
        let b = unit(vec![0.0, 1.0]);
        // First occurrence: A at index 0 → label 0, B at index 1 → label 1.
        let labels = cluster_embeddings(&[a, b], 0.5);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
    }

    #[test]
    fn test_cosine_similarity_unit_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }
}
