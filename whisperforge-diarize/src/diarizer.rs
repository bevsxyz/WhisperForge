use crate::clustering::cluster_embeddings;

/// Assigns speaker labels to a sequence of pre-computed speaker embeddings.
///
/// Each embedding represents one transcription segment (typically a 30 s chunk
/// or a VAD-bounded voice span). Labels are strings of the form `"SPEAKER_00"`,
/// numbered in order of first appearance in the segment sequence.
///
/// Segments with zero-length or zero-norm embeddings (too short to embed) are
/// labelled `"SPEAKER_UNKNOWN"` and excluded from clustering.
pub struct SpeakerDiarizer {
    /// Cosine similarity threshold above which two segments are merged into
    /// the same speaker cluster. Typical range: 0.5 – 0.85. Lower values
    /// merge more aggressively; higher values split more.
    pub similarity_threshold: f32,
}

impl Default for SpeakerDiarizer {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
        }
    }
}

impl SpeakerDiarizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Assign a speaker label string to each embedding.
    ///
    /// `embeddings` must have the same length as the number of segments the
    /// caller intends to label. An empty embedding (`vec![]`) or an all-zero
    /// embedding is treated as "too short to classify" and receives
    /// `"SPEAKER_UNKNOWN"`.
    pub fn assign_labels(&self, embeddings: &[Vec<f32>]) -> Vec<String> {
        if embeddings.is_empty() {
            return vec![];
        }

        // Separate valid embeddings from unknowns.
        let valid_indices: Vec<usize> = embeddings
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.is_empty() && e.iter().any(|&v| v.abs() > 1e-8))
            .map(|(i, _)| i)
            .collect();

        let valid_embeddings: Vec<Vec<f32>> = valid_indices
            .iter()
            .map(|&i| embeddings[i].clone())
            .collect();

        let cluster_labels = cluster_embeddings(&valid_embeddings, self.similarity_threshold);

        let mut labels = vec!["SPEAKER_UNKNOWN".to_string(); embeddings.len()];
        for (pos, &orig_idx) in valid_indices.iter().enumerate() {
            labels[orig_idx] = format!("SPEAKER_{:02}", cluster_labels[pos]);
        }
        labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_single_segment_gets_speaker_00() {
        let d = SpeakerDiarizer::new();
        let labels = d.assign_labels(&[unit(vec![1.0, 0.0])]);
        assert_eq!(labels, vec!["SPEAKER_00"]);
    }

    #[test]
    fn test_empty_embedding_gets_unknown() {
        let d = SpeakerDiarizer::new();
        let labels = d.assign_labels(&[vec![]]);
        assert_eq!(labels, vec!["SPEAKER_UNKNOWN"]);
    }

    #[test]
    fn test_two_speakers_get_distinct_labels() {
        let d = SpeakerDiarizer::new();
        let a = unit(vec![1.0, 0.0]);
        let b = unit(vec![0.0, 1.0]);
        let labels = d.assign_labels(&[a, b]);
        assert_ne!(labels[0], labels[1]);
        assert!(!labels[0].contains("UNKNOWN"));
        assert!(!labels[1].contains("UNKNOWN"));
    }

    #[test]
    fn test_same_speaker_repeated_gets_same_label() {
        let d = SpeakerDiarizer::new();
        let a = unit(vec![1.0, 0.0]);
        let labels = d.assign_labels(&[a.clone(), a]);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn test_label_format_is_zero_padded() {
        assert_eq!(format!("SPEAKER_{:02}", 0), "SPEAKER_00");
        assert_eq!(format!("SPEAKER_{:02}", 9), "SPEAKER_09");
        assert_eq!(format!("SPEAKER_{:02}", 10), "SPEAKER_10");
    }
}
