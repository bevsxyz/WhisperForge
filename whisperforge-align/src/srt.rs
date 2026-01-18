use anyhow::Result;
use serde::{Deserialize, Serialize};

/// SubRip format (.srt) subtitle entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrtEntry {
    /// Sequence number (1-based)
    pub sequence: usize,
    /// Start time in format "HH:MM:SS,mmm"
    pub start_time: String,
    /// End time in format "HH:MM:SS,mmm"
    pub end_time: String,
    /// Subtitle text
    pub text: String,
}

impl SrtEntry {
    /// Create a new SRT entry
    pub fn new(sequence: usize, start_time: f64, end_time: f64, text: String) -> Self {
        Self {
            sequence,
            start_time: format_time(start_time),
            end_time: format_time(end_time),
            text,
        }
    }

    /// Get duration of this entry
    pub fn duration(&self) -> f64 {
        parse_time(&self.end_time) - parse_time(&self.start_time)
    }
}

/// Format time as SRT timestamp (HH:MM:SS,mmm)
fn format_time(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let remaining_seconds = seconds % 60.0;
    let secs = remaining_seconds as u32;
    let millis = ((remaining_seconds - secs as f64) * 1000.0) as u32;

    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

/// Parse SRT timestamp back to seconds
fn parse_time(timestamp: &str) -> f64 {
    let parts: Vec<&str> = timestamp.split(',').collect();
    if parts.len() != 3 {
        return 0.0;
    }

    let time_part = parts[0];
    let time_parts: Vec<&str> = time_part.split(':').collect();
    if time_parts.len() != 3 {
        return 0.0;
    }

    let hours: f64 = time_parts[0].parse().unwrap_or(0.0);
    let minutes: f64 = time_parts[1].parse().unwrap_or(0.0);
    let seconds: f64 = time_parts[2].parse().unwrap_or(0.0);
    let millis: f64 = if parts.len() > 2 {
        parts[2].parse().unwrap_or(0.0)
    } else {
        0.0
    };

    hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0
}

/// SRT writer for transcription output
pub struct SrtWriter {
    entries: Vec<SrtEntry>,
}

impl SrtWriter {
    /// Create a new SRT writer
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an entry to the SRT
    pub fn add_entry(&mut self, entry: SrtEntry) {
        self.entries.push(entry);
    }

    /// Convert all entries to SRT format string
    pub fn to_string(&self) -> String {
        let mut srt = String::new();
        
        for (i, entry) in self.entries.iter().enumerate() {
            srt.push_str(&format!(
                "{}\n{} --> {}\n{}\n\n",
                i + 1,
                entry.start_time,
                entry.end_time,
                entry.text
            ));
        }

        srt
    }

    /// Write SRT content to a file
    pub fn write_to_file(&self, path: &str) -> Result<()> {
        let content = self.to_string();
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create SRT from segments with word timestamps
    pub fn from_segments(segments: &[TranscribedSegment]) -> Self {
        let mut writer = Self::new();
        let mut sequence = 1;

        for segment in segments {
            // Split segment into words for subtitle timing
            let words_per_line = 10; // Approximate words per subtitle line
            for (i, chunk) in segment.words.chunks(words_per_line).enumerate() {
                let chunk_start = if i == 0 {
                    segment.start_time
                } else {
                    segment.start_time + (i * words_per_line * segment.word_duration)
                };

                let chunk_end = chunk_start + (chunk.len() * segment.word_duration);

                let text = chunk.join(" ");
                writer.add_entry(SrtEntry::new(
                    sequence,
                    chunk_start,
                    chunk_end,
                    text,
                ));
                sequence += 1;
            }
        }

        writer
    }
}

/// Transcribed segment with word-level timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribedSegment {
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Duration of each word (seconds)
    pub word_duration: f64,
    /// Transcribed words in order
    pub words: Vec<String>,
}

impl TranscribedSegment {
    /// Create a new transcribed segment
    pub fn new(start_time: f64, end_time: f64, words: Vec<String>) -> Self {
        let duration = end_time - start_time;
        let word_duration = if words.is_empty() {
            0.0
        } else {
            duration / words.len() as f64
        };

        Self {
            start_time,
            end_time,
            word_duration,
            words,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_entry_creation() {
        let entry = SrtEntry::new(1, 10.5, 15.75, "Hello world".to_string());
        assert_eq!(entry.sequence, 1);
        assert_eq!(entry.start_time, "00:00:10,500");
        assert_eq!(entry.end_time, "00:00:15,750");
        assert_eq!(entry.text, "Hello world");
    }

    #[test]
    fn test_time_formatting() {
        assert_eq!(format_time(0.0), "00:00:00,000");
        assert_eq!(format_time(65.5), "00:01:05,500");
        assert_eq!(format_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_time_parsing() {
        assert_eq!(parse_time("00:00:10,500"), 10.5);
        assert_eq!(parse_time("01:05:15,750"), 3915.75);
        assert_eq!(parse_time("invalid"), 0.0);
    }

    #[test]
    fn test_srt_writer() -> Result<()> {
        let mut writer = SrtWriter::new();
        writer.add_entry(SrtEntry::new(1, 0.0, 3.5, "First subtitle".to_string()));
        writer.add_entry(SrtEntry::new(2, 4.0, 7.5, "Second subtitle".to_string()));

        let expected = "1\n00:00:00,000 --> 00:00:03,500\nFirst subtitle\n\n2\n00:00:04,000 --> 00:00:07,500\nSecond subtitle\n\n";
        assert_eq!(writer.to_string(), expected);

        Ok(())
    }

    #[test]
    fn test_transcribed_segment() {
        let words = vec!["Hello".to_string(), "world".to_string()];
        let segment = TranscribedSegment::new(10.0, 15.0, words);
        
        assert_eq!(segment.start_time, 10.0);
        assert_eq!(segment.end_time, 15.0);
        assert_eq!(segment.word_duration, 2.5); // 5 seconds / 2 words
        assert_eq!(segment.words, words);
    }
}