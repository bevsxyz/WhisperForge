use std::io::{self, Write};

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

/// NDJSON output sink — body implemented in C11.
pub struct JsonSink<W: Write> {
    pub writer: W,
}

impl<W: Write> JsonSink<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> StreamSink for JsonSink<W> {
    fn on_partial(&mut self, _committed: &str, _tentative: &str) -> Result<()> {
        Ok(())
    }
    fn on_commit(&mut self, _new_committed_text: &str, _at_secs: f32) -> Result<()> {
        Ok(())
    }
    fn on_endpoint(&mut self, _full_utterance: &str, _start_secs: f32, _end_secs: f32) -> Result<()> {
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
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
    fn on_endpoint(&mut self, _full_utterance: &str, _start_secs: f32, _end_secs: f32) -> Result<()> {
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
    fn close(&mut self) -> Result<()> {
        for s in &mut self.sinks {
            s.close()?;
        }
        Ok(())
    }
}
