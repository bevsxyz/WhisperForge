//! Streaming audio decoder for processing large files without pre-loading.
//!
//! This module provides `AudioChunkIterator`, a pull-based iterator that decodes
//! and resamples audio on-demand, holding at most one 30-second chunk in memory.
//!
//! # Memory efficiency
//!
//! Instead of loading entire files before processing:
//! - Peak RAM ≈ 2 MB per chunk (16 kHz mono, 30s = 480k samples)
//! - Processes 1-hour files with <10 MB working set (model weights excluded)
//! - Streaming resampler avoids full-file copies

use anyhow::{Result, anyhow};
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, WindowFunction};
use std::path::Path;
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    formats::{FormatOptions, FormatReader},
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

/// A decoded chunk of audio at 16 kHz mono.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// 16 kHz mono samples.
    pub samples: Vec<f32>,
    /// Start time in seconds (relative to file).
    pub start_sec: f32,
    /// End time in seconds (relative to file).
    pub end_sec: f32,
}

/// Streaming iterator over audio file chunks.
///
/// Decodes and resamples on-demand, holding only one packet + resampler state + overlap.
/// Yields chunks with automatic 1-second overlap for alignment across boundaries.
pub struct AudioChunkIterator {
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    sample_rate: u32,
    channels: u16,

    // Rubato streaming resampler state (survives between packets)
    resampler: Option<Async<f32>>,

    // Overlap buffer from previous chunk
    overlap_buf: Vec<f32>,

    // Configuration
    chunk_samples: usize,   // e.g., 480_000 for 30s @ 16 kHz
    overlap_samples: usize, // e.g., 16_000 for 1s @ 16 kHz
    target_rate: u32,

    // Position tracking
    samples_out: usize, // Total 16 kHz samples emitted so far
    done: bool,
    sample_buffer: Option<SampleBuffer<f32>>,
}

impl AudioChunkIterator {
    /// Create a streaming iterator from an audio file.
    ///
    /// # Arguments
    /// * `path` - Path to audio file (WAV, MP3, FLAC, OGG, AAC, MKV, etc.)
    /// * `chunk_sec` - Chunk duration in seconds
    /// * `overlap_sec` - Overlap between chunks in seconds
    pub fn new<P: AsRef<Path>>(path: P, chunk_sec: f32, overlap_sec: f32) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow!("Failed to open audio file '{}': {}", path.display(), e))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| anyhow!("Unsupported audio format '{}': {}", path.display(), e))?;

        let format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow!("No audio tracks found in '{}'", path.display()))?;

        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| anyhow!("Unknown sample rate in '{}'", path.display()))?;
        let channels = track
            .codec_params
            .channels
            .ok_or_else(|| anyhow!("Unknown channel count in '{}'", path.display()))?
            .count() as u16;

        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| anyhow!("Failed to create decoder for '{}': {}", path.display(), e))?;

        let target_rate = 16000;
        let chunk_samples = (chunk_sec * target_rate as f32) as usize;
        let overlap_samples = (overlap_sec * target_rate as f32) as usize;

        // Create resampler if needed (always mono output: 1 channel)
        let resampler = if sample_rate != target_rate {
            let f_ratio = target_rate as f64 / sample_rate as f64;
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: rubato::SincInterpolationType::Cubic,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            let chunk_size = 1024;
            let resampler = Async::<f32>::new_sinc(
                f_ratio,
                2.0,
                &params,
                chunk_size,
                1, // Always mono output
                FixedAsync::Input,
            )
            .map_err(|e| anyhow!("Failed to create resampler: {}", e))?;
            Some(resampler)
        } else {
            None
        };

        Ok(Self {
            reader: format,
            decoder,
            track_id,
            sample_rate,
            channels,
            resampler,
            overlap_buf: Vec::new(),
            chunk_samples,
            overlap_samples,
            target_rate,
            samples_out: 0,
            done: false,
            sample_buffer: None,
        })
    }

    /// Create a 30s-chunk iterator with 1s overlap (Whisper default).
    pub fn default_whisper<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::new(path, 30.0, 1.0)
    }

    /// Decode and accumulate the next chunk of audio.
    fn next_chunk(&mut self) -> Result<Option<AudioChunk>> {
        let target_samples = self.chunk_samples;
        let mut samples = Vec::with_capacity(target_samples);

        // Prepend overlap from previous chunk
        samples.extend_from_slice(&self.overlap_buf);
        let overlap_len = self.overlap_buf.len();

        // Accumulate decoded packets until we have enough samples
        loop {
            if samples.len() >= target_samples {
                break;
            }

            let packet = match self.reader.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.done = true;
                    break;
                }
                Err(symphonia::core::errors::Error::ResetRequired) => {
                    self.sample_buffer = None;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Error reading packet: {}", e));
                }
            };

            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(d) => d,
                Err(symphonia::core::errors::Error::IoError(_)) => continue,
                Err(e) => {
                    return Err(anyhow!("Decode error: {}", e));
                }
            };

            let buf = self.sample_buffer.get_or_insert_with(|| {
                SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec())
            });
            buf.copy_interleaved_ref(decoded);
            let packet_samples = buf.samples().to_vec();

            // Resample or copy samples
            if self.resampler.is_some() {
                let mut resampler = self.resampler.take().unwrap();
                self.resample_packet_into_buffer(&packet_samples, &mut resampler, &mut samples)?;
                self.resampler = Some(resampler);
            } else {
                // Already at target rate; convert to mono if needed
                samples.extend_from_slice(&packet_samples);
            }
        }

        // Convert to mono if needed
        if self.channels > 1 && self.resampler.is_none() {
            samples = self.to_mono(&samples);
        }

        // Trim to target size
        if samples.len() > target_samples {
            samples.truncate(target_samples);
        }

        // If we have nothing, we're done
        if samples.is_empty() {
            return Ok(None);
        }

        // Save overlap for next chunk
        let overlap_start = if samples.len() >= self.overlap_samples {
            samples.len() - self.overlap_samples
        } else {
            0
        };
        self.overlap_buf = samples[overlap_start..].to_vec();

        // Calculate timestamps
        let start_sec = self.samples_out as f32 / self.target_rate as f32;
        let end_sec = (self.samples_out + samples.len()) as f32 / self.target_rate as f32;
        self.samples_out += samples.len() - overlap_len;

        Ok(Some(AudioChunk {
            samples,
            start_sec,
            end_sec,
        }))
    }

    /// Resample a packet of audio into the samples buffer.
    fn resample_packet_into_buffer(
        &mut self,
        packet_samples: &[f32],
        resampler: &mut Async<f32>,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if packet_samples.is_empty() {
            return Ok(());
        }

        // Deinterleave samples into per-channel vectors
        let frames_per_channel = packet_samples.len() / self.channels as usize;
        let mut input_channels: Vec<Vec<f32>> =
            vec![Vec::with_capacity(frames_per_channel); self.channels as usize];

        for (i, &sample) in packet_samples.iter().enumerate() {
            let channel = i % self.channels as usize;
            input_channels[channel].push(sample);
        }

        // Convert to mono by averaging channels
        if self.channels > 1 {
            input_channels[0] = (0..frames_per_channel)
                .map(|f| input_channels.iter().map(|ch| ch[f]).sum::<f32>() / self.channels as f32)
                .collect();
            input_channels.truncate(1);
        }

        // Prepare adapters for rubato
        let input_adapter = SequentialSliceOfVecs::new(&input_channels, 1, frames_per_channel)
            .map_err(|e| anyhow!("Failed to create input adapter: {}", e))?;

        // Estimate output size
        let f_ratio = self.target_rate as f64 / self.sample_rate as f64;
        let estimated_output_frames = (frames_per_channel as f64 * f_ratio) as usize + 10; // +10 for safety

        let mut output_channels: Vec<Vec<f32>> = vec![vec![0.0f32; estimated_output_frames]; 1];
        let mut output_adapter =
            SequentialSliceOfVecs::new_mut(&mut output_channels, 1, estimated_output_frames)
                .map_err(|e| anyhow!("Failed to create output adapter: {}", e))?;

        let mut indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            active_channels_mask: None,
            partial_len: None,
        };

        let mut input_frames_left = frames_per_channel;
        let mut input_frames_next = resampler.input_frames_next();

        // Process full chunks from the resampler
        while input_frames_left >= input_frames_next {
            let (frames_read, frames_written) = resampler
                .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
                .map_err(|e| anyhow!("Resampling failed: {}", e))?;

            indexing.input_offset += frames_read;
            indexing.output_offset += frames_written;
            input_frames_left -= frames_read;
            input_frames_next = resampler.input_frames_next();
        }

        // Remaining frames less than chunk size are buffered internally by the resampler
        // and will be output on the next packet. No need to force-process them here.

        output.extend_from_slice(&output_channels[0][..indexing.output_offset]);
        Ok(())
    }

    /// Convert interleaved samples to mono by averaging channels.
    fn to_mono(&self, samples: &[f32]) -> Vec<f32> {
        if self.channels == 1 {
            return samples.to_vec();
        }
        samples
            .chunks(self.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / self.channels as f32)
            .collect()
    }
}

impl Iterator for AudioChunkIterator {
    type Item = Result<AudioChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done && self.overlap_buf.is_empty() {
            return None;
        }
        match self.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_chunk_iterator_creation() -> Result<()> {
        // Just test that we can construct without a file (will fail gracefully)
        match AudioChunkIterator::default_whisper("/nonexistent/file.wav") {
            Err(e) => {
                assert!(e.to_string().contains("Failed to open audio file"));
                Ok(())
            }
            Ok(_) => Err(anyhow!("Should have failed to open nonexistent file")),
        }
    }
}
