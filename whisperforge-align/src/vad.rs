use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Voice Activity Detector for audio segmentation
pub struct VoiceActivityDetector {
    sample_rate: u32,
    frame_duration: Duration,
    vad_threshold: f32,
}

impl VoiceActivityDetector {
    /// Create a new VAD with default parameters
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            frame_duration: Duration::from_millis(30), // 30ms frames
            vad_threshold: 0.5,
        }
    }

    /// Set VAD threshold (0.0 - 1.0, higher = more strict)
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set frame duration for analysis
    pub fn with_frame_duration(mut self, duration: Duration) -> Self {
        self.frame_duration = duration;
        self
    }

    /// Detect voice activity in audio data
    pub fn detect(&self, audio: &[f32]) -> Result<Vec<VoiceSegment>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let frame_samples = (self.sample_rate as f32 * self.frame_duration.as_secs_f32()) as usize;
        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut speech_start = 0;

        for (i, frame) in audio.chunks(frame_samples).enumerate() {
            let is_speech = self.is_voice_active(frame)?;
            let frame_start = i * frame_samples;

            if is_speech && !in_speech {
                // Speech starts
                in_speech = true;
                speech_start = frame_start;
            } else if !is_speech && in_speech {
                // Speech ends
                in_speech = false;
                let speech_end = (i * frame_samples).saturating_sub(1);
                segments.push(VoiceSegment {
                    start_sample: speech_start,
                    end_sample: speech_end,
                    start_time: speech_start as f64 / self.sample_rate as f64,
                    end_time: speech_end as f64 / self.sample_rate as f64,
                    confidence: self.calculate_frame_confidence(&audio[speech_start..=speech_end]),
                });
            }
        }

        // Handle case where audio ends while in speech
        if in_speech {
            let speech_end = audio.len() - 1;
            segments.push(VoiceSegment {
                start_sample: speech_start,
                end_sample: speech_end,
                start_time: speech_start as f64 / self.sample_rate as f64,
                end_time: speech_end as f64 / self.sample_rate as f64,
                confidence: self.calculate_frame_confidence(&audio[speech_start..=speech_end]),
            });
        }

        Ok(segments)
    }

    /// Determine if a frame contains voice activity
    fn is_voice_active(&self, frame: &[f32]) -> Result<bool> {
        if frame.is_empty() {
            return Ok(false);
        }

        // Simple energy-based VAD
        let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32;
        let zcr = self.zero_crossing_rate(frame);

        // Combined heuristic: energy + zero crossing rate
        let voice_probability = self.energy_to_probability(energy) * (1.0 - zcr);

        Ok(voice_probability > self.vad_threshold)
    }

    /// Calculate zero crossing rate
    fn zero_crossing_rate(&self, frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        let crossings = frame
            .windows(2)
            .filter(|&pair| pair[0] * pair[1] < 0.0)
            .count();

        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Convert energy to probability using exponential curve
    fn energy_to_probability(&self, energy: f32) -> f32 {
        1.0 - (-energy * 250.0).exp()
    }

    /// Calculate confidence score for a segment
    fn calculate_frame_confidence(&self, segment: &[f32]) -> f32 {
        if segment.is_empty() {
            return 0.0;
        }

        let avg_energy = segment.iter().map(|&x| x * x).sum::<f32>() / segment.len() as f32;
        self.energy_to_probability(avg_energy)
    }
}

/// Represents a segment containing voice activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSegment {
    /// Start sample index
    pub start_sample: usize,
    /// End sample index (inclusive)
    pub end_sample: usize,
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

impl VoiceSegment {
    /// Get duration of the segment in seconds
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Get number of samples in the segment
    pub fn sample_count(&self) -> usize {
        self.end_sample - self.start_sample + 1
    }

    /// Check if segment is long enough for transcription
    pub fn is_transcribable(&self, min_duration: f64) -> bool {
        self.duration() >= min_duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let vad = VoiceActivityDetector::new(16000);
        assert_eq!(vad.sample_rate, 16000);
        assert_eq!(vad.frame_duration, Duration::from_millis(30));
        assert_eq!(vad.vad_threshold, 0.5);
    }

    #[test]
    fn test_vad_custom_threshold() {
        let vad = VoiceActivityDetector::new(16000).with_threshold(0.8);
        assert_eq!(vad.vad_threshold, 0.8);
    }

    #[test]
    fn test_vad_empty_audio() -> Result<()> {
        let vad = VoiceActivityDetector::new(16000);
        let segments = vad.detect(&[])?;
        assert!(segments.is_empty());
        Ok(())
    }

    #[test]
    fn test_zero_crossing_rate() {
        let vad = VoiceActivityDetector::new(16000);

        // All positive samples - zero crossings
        let signal = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(vad.zero_crossing_rate(&signal), 0.0);

        // Alternating signal - high zero crossings
        let signal = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let expected_zcr = 5.0 / 5.0; // 5 crossings out of 5 possible
        assert!((vad.zero_crossing_rate(&signal) - expected_zcr).abs() < 0.001);
    }

    #[test]
    fn test_energy_to_probability() {
        let vad = VoiceActivityDetector::new(16000);

        // Zero energy should give low probability
        let prob = vad.energy_to_probability(0.0);
        assert!(prob < 0.1);

        // High energy should give high probability
        let prob = vad.energy_to_probability(0.01);
        assert!(prob > 0.9);
    }

    #[test]
    fn test_voice_segment() {
        let segment = VoiceSegment {
            start_sample: 0,
            end_sample: 1599, // 0.1s at 16kHz
            start_time: 0.0,
            end_time: 0.1,
            confidence: 0.8,
        };

        assert_eq!(segment.duration(), 0.1);
        assert_eq!(segment.sample_count(), 1600);
        assert!(segment.is_transcribable(0.05));
        assert!(!segment.is_transcribable(0.15));
    }
}
