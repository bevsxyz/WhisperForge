use anyhow::Result;
use whisperforge_align::{VoiceActivityDetector, VoiceSegment};
use whisperforge_core::audio::AudioData;

/// Audio segmentation using VAD for batched processing
pub struct AudioSegmenter {
    vad: VoiceActivityDetector,
    min_segment_length: f64,
    max_segment_length: f64,
    pad_duration: f64,
}

impl AudioSegmenter {
    /// Create a new audio segmenter with default parameters
    pub fn new(sample_rate: u32) -> Self {
        Self {
            vad: VoiceActivityDetector::new(sample_rate),
            min_segment_length: 1.0, // 1 second minimum
            max_segment_length: 30.0, // 30 second maximum
            pad_duration: 0.1, // 100ms padding
        }
    }

    /// Set minimum segment length in seconds
    pub fn with_min_segment_length(mut self, length: f64) -> Self {
        self.min_segment_length = length;
        self
    }

    /// Set maximum segment length in seconds
    pub fn with_max_segment_length(mut self, length: f64) -> Self {
        self.max_segment_length = length;
        self
    }

    /// Set padding duration in seconds to add around speech segments
    pub fn with_pad_duration(mut self, duration: f64) -> Self {
        self.pad_duration = duration;
        self
    }

    /// Set VAD threshold
    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.vad = self.vad.with_threshold(threshold);
        self
    }

    /// Segment audio based on voice activity
    pub fn segment(&self, audio: &AudioData) -> Result<Vec<AudioSegment>> {
        let voice_segments = self.vad.detect(&audio.samples)?;
        let mut segments = Vec::new();

        for voice_seg in voice_segments {
            // Skip segments that are too short
            if !voice_seg.is_transcribable(self.min_segment_length) {
                continue;
            }

            // Add padding
            let padded_start = (voice_seg.start_time - self.pad_duration).max(0.0);
            let padded_end = (voice_seg.end_time + self.pad_duration)
                .min(audio.duration());

            // Convert to sample indices
            let start_sample = (padded_start * audio.sample_rate as f64) as usize;
            let end_sample = (padded_end * audio.sample_rate as f64) as usize;

            // Split long segments
            let segment_duration = padded_end - padded_start;
            if segment_duration > self.max_segment_length {
                self.split_long_segment(audio, start_sample, end_sample, &mut segments);
            } else {
                segments.push(AudioSegment {
                    start_sample,
                    end_sample,
                    start_time: padded_start,
                    end_time: padded_end,
                    samples: audio.samples[start_sample..=end_sample].to_vec(),
                });
            }
        }

        Ok(segments)
    }

    /// Split long segments into smaller chunks
    fn split_long_segment(
        &self,
        audio: &AudioData,
        start_sample: usize,
        end_sample: usize,
        segments: &mut Vec<AudioSegment>,
    ) {
        let max_samples = (self.max_segment_length * audio.sample_rate as f64) as usize;
        let mut current_start = start_sample;

        while current_start < end_sample {
            let current_end = (current_start + max_samples).min(end_sample);
            let start_time = current_start as f64 / audio.sample_rate as f64;
            let end_time = current_end as f64 / audio.sample_rate as f64;

            segments.push(AudioSegment {
                start_sample: current_start,
                end_sample: current_end - 1,
                start_time,
                end_time,
                samples: audio.samples[current_start..current_end].to_vec(),
            });

            current_start = current_end;
        }
    }

    /// Merge segments that are close together
    pub fn merge_segments(&self, segments: &[AudioSegment], max_gap: f64) -> Result<Vec<AudioSegment>> {
        let mut merged = Vec::new();
        let mut current: Option<AudioSegment> = None;

        for segment in segments {
            match current.as_mut() {
                Some(curr) => {
                    let gap = segment.start_time - curr.end_time;
                    if gap <= max_gap {
                        // Merge segments
                        curr.end_sample = segment.end_sample;
                        curr.end_time = segment.end_time;
                        curr.samples.extend(&segment.samples);
                    } else {
                        merged.push(current.take().unwrap());
                        current = Some(segment.clone());
                    }
                }
                None => current = Some(segment.clone()),
            }
        }

        if let Some(curr) = current {
            merged.push(curr);
        }

        Ok(merged)
    }
}

/// Audio segment with samples and metadata
#[derive(Debug, Clone)]
pub struct AudioSegment {
    /// Start sample index
    pub start_sample: usize,
    /// End sample index (inclusive)
    pub end_sample: usize,
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Audio samples for this segment
    pub samples: Vec<f32>,
}

impl AudioSegment {
    /// Get duration of the segment in seconds
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Get number of samples in the segment
    pub fn sample_count(&self) -> usize {
        self.end_sample - self.start_sample + 1
    }

    /// Check if segment is long enough for transcription
    pub fn is_long_enough(&self, min_duration: f64) -> bool {
        self.duration() >= min_duration
    }

    /// Convert to AudioData
    pub fn to_audio_data(&self, sample_rate: u32) -> AudioData {
        AudioData {
            samples: self.samples.clone(),
            sample_rate,
            channels: 1, // Always mono after preprocessing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmenter_creation() {
        let segmenter = AudioSegmenter::new(16000);
        assert_eq!(segmenter.min_segment_length, 1.0);
        assert_eq!(segmenter.max_segment_length, 30.0);
        assert_eq!(segmenter.pad_duration, 0.1);
    }

    #[test]
    fn test_segmenter_custom_parameters() {
        let segmenter = AudioSegmenter::new(16000)
            .with_min_segment_length(2.0)
            .with_max_segment_length(60.0)
            .with_pad_duration(0.2)
            .with_vad_threshold(0.8);

        assert_eq!(segmenter.min_segment_length, 2.0);
        assert_eq!(segmenter.max_segment_length, 60.0);
        assert_eq!(segmenter.pad_duration, 0.2);
    }

    #[test]
    fn test_audio_segment() {
        let segment = AudioSegment {
            start_sample: 0,
            end_sample: 15999, // 1 second at 16kHz
            start_time: 0.0,
            end_time: 1.0,
            samples: vec![0.0; 16000],
        };

        assert_eq!(segment.duration(), 1.0);
        assert_eq!(segment.sample_count(), 16000);
        assert!(segment.is_long_enough(0.5));
        assert!(!segment.is_long_enough(1.5));

        let audio_data = segment.to_audio_data(16000);
        assert_eq!(audio_data.sample_rate, 16000);
        assert_eq!(audio_data.channels, 1);
        assert_eq!(audio_data.samples.len(), 16000);
    }

    #[test]
    fn test_merge_segments() -> Result<()> {
        let segmenter = AudioSegmenter::new(16000);
        
        let segments = vec![
            AudioSegment {
                start_sample: 0,
                end_sample: 7999,
                start_time: 0.0,
                end_time: 0.5,
                samples: vec![0.0; 8000],
            },
            AudioSegment {
                start_sample: 8000,
                end_sample: 15999,
                start_time: 0.5,
                end_time: 1.0,
                samples: vec![0.0; 8000],
            },
        ];

        // Should merge since gap is 0.0
        let merged = segmenter.merge_segments(&segments, 0.1)?;
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].duration(), 1.0);

        // Should not merge with larger gap threshold
        let merged = segmenter.merge_segments(&segments, 0.0)?;
        assert_eq!(merged.len(), 2);

        Ok(())
    }
}