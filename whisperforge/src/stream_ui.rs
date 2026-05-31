use std::io::{self, Write};

use serde_json::json;

use anyhow::Result;
use crossterm::{
    cursor::MoveToColumn,
    execute,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{Clear, ClearType},
};

pub trait StreamSink {
    fn on_partial(&mut self, committed: &str, tentative: &str) -> Result<()>;
    fn on_commit(&mut self, new_committed_text: &str, at_secs: f32) -> Result<()>;
    fn on_endpoint(&mut self, full_utterance: &str, start_secs: f32, end_secs: f32) -> Result<()>;
    /// Per-window decode telemetry. Default impl is a no-op; sinks that care about
    /// quality observability (currently `JsonSink`) override this. Called once per
    /// window decode, after `decode_window` returns and before `committer.ingest`.
    fn on_decode_metrics(
        &mut self,
        _avg_logprob: f32,
        _min_logprob: f32,
        _n_tokens: usize,
        _cap_hit: bool,
    ) -> Result<()> {
        Ok(())
    }
    fn close(&mut self) -> Result<()>;
}

fn fmt_secs(s: f32) -> String {
    let total_tenths = (s * 10.0).round() as u32;
    let tenths = total_tenths % 10;
    let total_secs = total_tenths / 10;
    let mins = total_secs / 60;
    let secs_part = total_secs % 60;
    format!("{mins:02}:{secs_part:02}.{tenths}")
}

pub struct TerminalSink {
    color_enabled: bool,
    stdout: io::Stdout,
    has_partial_line: bool,
}

impl TerminalSink {
    pub fn new(color_enabled: bool) -> Self {
        Self {
            color_enabled,
            stdout: io::stdout(),
            has_partial_line: false,
        }
    }
}

impl StreamSink for TerminalSink {
    fn on_partial(&mut self, committed: &str, tentative: &str) -> Result<()> {
        if self.color_enabled {
            execute!(
                self.stdout,
                MoveToColumn(0),
                Clear(ClearType::UntilNewLine),
                Print(committed),
                SetForegroundColor(Color::Cyan),
                SetAttribute(Attribute::Dim),
                Print(tentative),
                ResetColor,
            )?;
        } else {
            print!("\r{}{}", committed, tentative);
            self.stdout.flush()?;
        }
        self.has_partial_line = true;
        Ok(())
    }

    fn on_commit(&mut self, _new_committed_text: &str, _at_secs: f32) -> Result<()> {
        // on_partial redraws the full line; no extra action needed here.
        Ok(())
    }

    fn on_endpoint(&mut self, full_utterance: &str, start_secs: f32, end_secs: f32) -> Result<()> {
        let ts = format!("[{}–{}]", fmt_secs(start_secs), fmt_secs(end_secs));
        if self.color_enabled {
            execute!(
                self.stdout,
                MoveToColumn(0),
                Clear(ClearType::UntilNewLine),
                Print(format!("{ts} {full_utterance}\n")),
            )?;
        } else {
            println!("\r{ts} {full_utterance}");
        }
        self.has_partial_line = false;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if self.has_partial_line {
            writeln!(self.stdout)?;
        }
        self.stdout.flush()?;
        Ok(())
    }
}

/// NDJSON output sink. One JSON object per line, all carrying `t` (seconds since sink construction)
/// and `type`. Event types:
/// - `partial`: `{committed, tentative}` — current best-effort transcript snapshot per stride
/// - `commit`: `{text, at_secs}` — newly stable text since the previous commit
/// - `endpoint`: `{text, start_secs, end_secs}` — completed utterance at EOU
/// - `decode_metrics`: `{avg_logprob, min_logprob, n_tokens, cap_hit}` — per-window decoder telemetry,
///   emitted once per `decode_window` call. `avg_logprob`/`min_logprob` are `null` when no content
///   tokens were decoded (e.g. no-speech-gated windows).
/// - `shutdown`: `{}` — emitted by `close()`.
pub struct JsonSink<W: Write> {
    pub writer: W,
    start: std::time::Instant,
}

impl<W: Write> JsonSink<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            start: std::time::Instant::now(),
        }
    }

    fn elapsed_t(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl<W: Write> StreamSink for JsonSink<W> {
    fn on_partial(&mut self, committed: &str, tentative: &str) -> Result<()> {
        let t = self.elapsed_t();
        writeln!(
            self.writer,
            "{}",
            json!({"t": t, "type": "partial", "committed": committed, "tentative": tentative})
        )?;
        Ok(())
    }

    fn on_commit(&mut self, new_committed_text: &str, at_secs: f32) -> Result<()> {
        let t = self.elapsed_t();
        writeln!(
            self.writer,
            "{}",
            json!({"t": t, "type": "commit", "text": new_committed_text, "at_secs": at_secs})
        )?;
        Ok(())
    }

    fn on_endpoint(&mut self, full_utterance: &str, start_secs: f32, end_secs: f32) -> Result<()> {
        let t = self.elapsed_t();
        writeln!(
            self.writer,
            "{}",
            json!({"t": t, "type": "endpoint", "text": full_utterance, "start_secs": start_secs, "end_secs": end_secs})
        )?;
        Ok(())
    }

    fn on_decode_metrics(
        &mut self,
        avg_logprob: f32,
        min_logprob: f32,
        n_tokens: usize,
        cap_hit: bool,
    ) -> Result<()> {
        let t = self.elapsed_t();
        let avg = if avg_logprob.is_finite() {
            json!(avg_logprob)
        } else {
            json!(null)
        };
        let min = if min_logprob.is_finite() {
            json!(min_logprob)
        } else {
            json!(null)
        };
        writeln!(
            self.writer,
            "{}",
            json!({"t": t, "type": "decode_metrics", "avg_logprob": avg, "min_logprob": min, "n_tokens": n_tokens, "cap_hit": cap_hit})
        )?;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        let t = self.elapsed_t();
        writeln!(self.writer, "{}", json!({"t": t, "type": "shutdown"}))?;
        self.writer.flush()?;
        Ok(())
    }
}

/// Appends committed utterances to a transcript file — body implemented in C11.
pub struct FileTranscriptSink {
    pub file: std::fs::File,
}

impl FileTranscriptSink {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { file })
    }
}

impl StreamSink for FileTranscriptSink {
    fn on_partial(&mut self, _committed: &str, _tentative: &str) -> Result<()> {
        Ok(())
    }
    fn on_commit(&mut self, _new_committed_text: &str, _at_secs: f32) -> Result<()> {
        Ok(())
    }
    fn on_endpoint(&mut self, full_utterance: &str, start_secs: f32, end_secs: f32) -> Result<()> {
        writeln!(
            self.file,
            "[{}–{}] {}",
            fmt_secs(start_secs),
            fmt_secs(end_secs),
            full_utterance
        )?;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

/// Fan-out sink: forwards every event to all inner sinks in order.
pub struct MultiSink {
    pub sinks: Vec<Box<dyn StreamSink>>,
}

impl MultiSink {
    pub fn new(sinks: Vec<Box<dyn StreamSink>>) -> Self {
        Self { sinks }
    }
}

impl StreamSink for MultiSink {
    fn on_partial(&mut self, committed: &str, tentative: &str) -> Result<()> {
        for s in &mut self.sinks {
            s.on_partial(committed, tentative)?;
        }
        Ok(())
    }
    fn on_commit(&mut self, new_committed_text: &str, at_secs: f32) -> Result<()> {
        for s in &mut self.sinks {
            s.on_commit(new_committed_text, at_secs)?;
        }
        Ok(())
    }
    fn on_endpoint(&mut self, full_utterance: &str, start_secs: f32, end_secs: f32) -> Result<()> {
        for s in &mut self.sinks {
            s.on_endpoint(full_utterance, start_secs, end_secs)?;
        }
        Ok(())
    }
    fn on_decode_metrics(
        &mut self,
        avg_logprob: f32,
        min_logprob: f32,
        n_tokens: usize,
        cap_hit: bool,
    ) -> Result<()> {
        for s in &mut self.sinks {
            s.on_decode_metrics(avg_logprob, min_logprob, n_tokens, cap_hit)?;
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        for s in &mut self.sinks {
            s.close()?;
        }
        Ok(())
    }
}
