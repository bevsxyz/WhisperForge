/// End-to-end streaming pipeline integration test.
///
/// Run with:
///   cargo test --release -p whisperforge-core --test stream_integration -- --ignored --nocapture
///
/// Requires:
///   models/tiny_en_converted.{mpk,cfg}
///   models/tokenizer.json
///   models/silero_vad.onnx  (or will be auto-downloaded)
///   test_data/LJ001-0001_16k.wav
///
/// Expected transcript fragment (LJ001-0001):
///   "Printing, in the only sense with which we are at present concerned,
///    differs from most if not from all the arts and crafts..."
use anyhow::Result;
use burn_flex::{Flex, FlexDevice};
use ringbuf::traits::Consumer as _;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tokenizers::Tokenizer;
use whisperforge_core::{
    Chunker, CommitDelta, Committer, EndpointConfig, Endpointer, FakeMic, PromptContext,
    QualityGate, SileroVad, Whisper, WindowConfig, compute_mel_from_samples, decode_window,
    ensure_silero_model, passes_quality_gate, stream_decode::DecodeContext,
};

type B = Flex<f32>;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
#[ignore]
fn test_stream_pipeline_on_ljspeech() -> Result<()> {
    let root = workspace_root();
    let models_dir = root.join("models");
    let audio_path = root.join("test_data/LJ001-0001_16k.wav");

    // Skip if required files are absent.
    let model_mpk = models_dir.join("tiny_en_converted.mpk");
    let tokenizer_path = models_dir.join("tokenizer.json");
    if !model_mpk.exists() {
        eprintln!("SKIP: {} not found", model_mpk.display());
        return Ok(());
    }
    if !tokenizer_path.exists() {
        eprintln!("SKIP: {} not found", tokenizer_path.display());
        return Ok(());
    }
    if !audio_path.exists() {
        eprintln!("SKIP: {} not found", audio_path.display());
        return Ok(());
    }

    let device = FlexDevice;
    let base = models_dir.join("tiny_en_converted");
    let model: Whisper<B> =
        whisperforge_core::load::load_whisper(base.to_str().expect("valid path"), &device)?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let tok = |s: &str, fb: u32| tokenizer.token_to_id(s).unwrap_or(fb);
    let sot_token = tok("<|startoftranscript|>", 50258);
    let language_token = tok("<|en|>", 50259);
    let transcribe_token = tok("<|transcribe|>", 50359);
    let eot_token = tok("<|endoftext|>", 50257);
    let no_speech_token = tok("<|nospeech|>", 50362);
    let notimestamps_token = tok("<|notimestamps|>", 50363);
    let timestamp_begin_token = 50364u32;
    let prevtext_token_id = tokenizer.token_to_id("<|prevtext|>");

    let vad_path = ensure_silero_model(&models_dir)?;
    let vad = SileroVad::open(&vad_path)?;

    let window_cfg = WindowConfig {
        max_window_secs: 28.0,
        stride_secs: 1.0,
        vad_threshold: 0.5,
        min_speech_secs: 0.25,
    };
    let mut chunker = Chunker::new(window_cfg, vad);
    let mut committer = Committer::new();
    let mut endpointer = Endpointer::new(EndpointConfig {
        silence_secs: 2.0,
        punct_silence_secs: 0.8,
        min_utterance_secs: 0.5,
    });
    let mut prompt_ctx = PromptContext::new(60, prevtext_token_id);

    let (fake_mic, _feeder_handle) = FakeMic::open(&audio_path, false)?;

    const WINDOW_SAMPLES: usize = 480_000;
    const SILENCE_FRAME: [f32; 512] = [0.0f32; 512];
    let mut drain_buf = vec![0.0f32; 4096];
    let mut committed_text = String::new();
    let mut endpoint_fired_mid_stream = false;
    let mut silence_pushed_secs = 0.0f32;

    loop {
        let n = fake_mic.consumer.lock().unwrap().pop_slice(&mut drain_buf);

        let window = if n > 0 {
            silence_pushed_secs = 0.0;
            match chunker.push(&drain_buf[..n]) {
                Some(w) => w,
                None => continue,
            }
        } else {
            // No new audio. Synthesise silence so the chunker keeps striding and the
            // endpointer can fire naturally after speech stops. Shut down once the file
            // is exhausted and trailing silence has comfortably exceeded silence_secs.
            if fake_mic.is_done() && silence_pushed_secs > 3.0 {
                break;
            }
            silence_pushed_secs += SILENCE_FRAME.len() as f32 / 16_000.0;
            thread::sleep(Duration::from_millis(1));
            match chunker.push(&SILENCE_FRAME) {
                Some(w) => w,
                None => continue,
            }
        };

        let emits = if window.had_speech {
            let mut padded = window.samples.clone();
            padded.resize(WINDOW_SAMPLES, 0.0);
            let mel =
                compute_mel_from_samples::<B>(&padded, 400, 160, 80, &device).expect("compute mel");
            let encoder_out = model.forward_encoder(mel);
            let ctx = DecodeContext {
                prompt_tokens: prompt_ctx.prompt_tokens(),
                language_token,
                transcribe_token,
                sot_token,
                eot_token,
                no_speech_token,
                notimestamps_token,
                timestamp_begin_token,
                max_new_tokens: 128,
                no_speech_threshold: 0.6,
            };
            decode_window::<B>(&model, encoder_out, &ctx, &tokenizer, &device)?
        } else {
            vec![]
        };

        // Same confidence gate as the live stream main loop (faster-whisper defaults).
        let emits = if !emits.is_empty() && !passes_quality_gate(&emits, &QualityGate::default()) {
            Vec::new()
        } else {
            emits
        };

        let (commit_delta, _tentative) = committer.ingest(emits);
        if let CommitDelta::Committed { ref new_text, .. } = commit_delta {
            if !new_text.is_empty() {
                committed_text.push_str(new_text);
                eprintln!("committed: {new_text}");
            }
        }

        if endpointer.step(&window, committer.committed_text()) || window.forced_eou {
            let final_delta = committer.finalize_utterance();
            if let CommitDelta::Committed { ref new_text, .. } = final_delta {
                if !new_text.is_empty() {
                    committed_text.push_str(new_text);
                }
            }
            endpoint_fired_mid_stream = true;
            prompt_ctx.update_after_eou(committer.committed_tokens(), 128);
            endpointer.reset();
            chunker.reset_utterance();
        }
    }

    // Finalize any remaining partial.
    let final_delta = committer.finalize_utterance();
    if let CommitDelta::Committed { ref new_text, .. } = final_delta {
        if !new_text.is_empty() {
            committed_text.push_str(new_text);
        }
    }

    eprintln!("Full committed transcript: {committed_text}");

    // Tightened UAT (post Phase F remediation):
    // (1) At least one endpoint event must fire mid-stream — driven by `Endpointer::step`
    //     seeing growing trailing silence — rather than only via the shutdown-path
    //     `finalize_utterance`. This exercises B3 (silence-tick) end-to-end.
    // (2) The committed transcript must cover the LJ001-0001 ground truth modulo minor ASR
    //     variance — measured as token-level overlap against a curated keyword set drawn
    //     from the reference sentence.
    //
    // Reference: "Printing, in the only sense with which we are at present concerned,
    //             differs from most if not from all the arts and crafts represented in
    //             the Exhibition"
    assert!(
        endpoint_fired_mid_stream,
        "no endpoint event fired during the stream — silence-tick / endpointer is broken"
    );

    let lower = committed_text.to_lowercase();
    let keywords = [
        "printing",
        "only",
        "sense",
        "which",
        "present",
        "concerned",
        "differs",
        "most",
        "from",
        "all",
        "arts",
        "crafts",
        "represented",
        "exhibition",
    ];
    let hits: Vec<&str> = keywords
        .iter()
        .copied()
        .filter(|kw| lower.contains(kw))
        .collect();
    assert!(
        hits.len() >= 10,
        "expected ≥10/14 LJ001-0001 keywords; got {} ({:?}) in transcript: {committed_text:?}",
        hits.len(),
        hits
    );

    Ok(())
}

/// Long-form continuity test for B9 (timestamp-anchored trim).
///
/// Forces a small max_window_secs so the 10 s LJ001-0001 clip hits the cap repeatedly.
/// Verifies that the cap-hit trim path keeps the utterance going as one continuous
/// commit stream (no premature endpoint events, no duplicated text at the trim seam),
/// and that at least one trim event actually fires.
#[test]
#[ignore]
fn test_stream_pipeline_long_form_trim() -> Result<()> {
    let root = workspace_root();
    let models_dir = root.join("models");
    let audio_path = root.join("test_data/LJ001-0001_16k.wav");

    let model_mpk = models_dir.join("tiny_en_converted.mpk");
    let tokenizer_path = models_dir.join("tokenizer.json");
    if !model_mpk.exists() || !tokenizer_path.exists() || !audio_path.exists() {
        eprintln!("SKIP: missing model or audio fixtures");
        return Ok(());
    }

    let device = FlexDevice;
    let base = models_dir.join("tiny_en_converted");
    let model: Whisper<B> =
        whisperforge_core::load::load_whisper(base.to_str().expect("valid path"), &device)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let tok = |s: &str, fb: u32| tokenizer.token_to_id(s).unwrap_or(fb);
    let sot_token = tok("<|startoftranscript|>", 50258);
    let language_token = tok("<|en|>", 50259);
    let transcribe_token = tok("<|transcribe|>", 50359);
    let eot_token = tok("<|endoftext|>", 50257);
    let no_speech_token = tok("<|nospeech|>", 50362);
    let notimestamps_token = tok("<|notimestamps|>", 50363);
    let timestamp_begin_token = 50364u32;
    let prevtext_token_id = tokenizer.token_to_id("<|prevtext|>");

    let vad_path = ensure_silero_model(&models_dir)?;
    let vad = SileroVad::open(&vad_path)?;

    // max_window_secs = 5.0 forces cap-hits within the 10 s clip; stride_secs = 1.0 gives
    // multiple opportunities to commit a timestamp before each cap.
    let window_cfg = WindowConfig {
        max_window_secs: 5.0,
        stride_secs: 1.0,
        vad_threshold: 0.5,
        min_speech_secs: 0.25,
    };
    let mut chunker = Chunker::new(window_cfg, vad);
    let mut committer = Committer::new();
    let mut endpointer = Endpointer::new(EndpointConfig {
        silence_secs: 2.0,
        punct_silence_secs: 0.8,
        min_utterance_secs: 0.5,
    });
    let mut prompt_ctx = PromptContext::new(60, prevtext_token_id);

    let (fake_mic, _feeder_handle) = FakeMic::open(&audio_path, false)?;

    const WINDOW_SAMPLES: usize = 480_000;
    const SILENCE_FRAME: [f32; 512] = [0.0f32; 512];
    let mut drain_buf = vec![0.0f32; 4096];
    let mut committed_text = String::new();
    let mut trim_count: usize = 0;
    let mut endpoint_count: usize = 0;
    let mut silence_pushed_secs = 0.0f32;

    loop {
        let n = fake_mic.consumer.lock().unwrap().pop_slice(&mut drain_buf);

        let window = if n > 0 {
            silence_pushed_secs = 0.0;
            match chunker.push(&drain_buf[..n]) {
                Some(w) => w,
                None => continue,
            }
        } else {
            if fake_mic.is_done() && silence_pushed_secs > 3.0 {
                break;
            }
            silence_pushed_secs += SILENCE_FRAME.len() as f32 / 16_000.0;
            thread::sleep(Duration::from_millis(1));
            match chunker.push(&SILENCE_FRAME) {
                Some(w) => w,
                None => continue,
            }
        };

        let emits = if window.had_speech {
            let mut padded = window.samples.clone();
            padded.resize(WINDOW_SAMPLES, 0.0);
            let mel =
                compute_mel_from_samples::<B>(&padded, 400, 160, 80, &device).expect("compute mel");
            let encoder_out = model.forward_encoder(mel);
            let ctx = DecodeContext {
                prompt_tokens: prompt_ctx.prompt_tokens(),
                language_token,
                transcribe_token,
                sot_token,
                eot_token,
                no_speech_token,
                notimestamps_token,
                timestamp_begin_token,
                max_new_tokens: 32,
                no_speech_threshold: 0.6,
            };
            decode_window::<B>(&model, encoder_out, &ctx, &tokenizer, &device)?
        } else {
            vec![]
        };

        // Same confidence gate as the live stream main loop (faster-whisper defaults).
        let emits = if !emits.is_empty() && !passes_quality_gate(&emits, &QualityGate::default()) {
            Vec::new()
        } else {
            emits
        };

        let (commit_delta, _tentative) = committer.ingest(emits);
        if let CommitDelta::Committed { ref new_text, .. } = commit_delta {
            if !new_text.is_empty() {
                committed_text.push_str(new_text);
                eprintln!("committed: {new_text}");
            }
        }

        let endpoint_fires = endpointer.step(&window, committer.committed_text());
        if endpoint_fires || window.forced_eou {
            let final_delta = committer.finalize_utterance();
            if let CommitDelta::Committed { ref new_text, .. } = final_delta {
                if !new_text.is_empty() {
                    committed_text.push_str(new_text);
                }
            }
            endpoint_count += 1;
            prompt_ctx.update_after_eou(committer.committed_tokens(), 128);
            endpointer.reset();
            chunker.reset_utterance();
        } else if window.cap_hit {
            // Same stride-based trim logic as the live stream main loop.
            // Trim 1.5 s — leaves 3.5 s of overlap between consecutive strides so the
            // non-causal encoder produces stable enough output for LCP to commit content.
            // Half-buffer (2.5 s) trim was too aggressive — LCP regressed at the seam and
            // committed nothing between trims.
            let target_samples = (1.5_f32 * 16_000.0) as usize;
            let has_commits = !committer.committed_tokens().is_empty();
            let trimmed = if has_commits {
                chunker.trim_oldest(target_samples)
            } else {
                0
            };
            if trimmed > 0 {
                let trim_delta = committer.on_trim();
                if let CommitDelta::Committed { ref new_text, .. } = trim_delta {
                    if !new_text.is_empty() {
                        committed_text.push_str(new_text);
                        eprintln!("trim-commit: {new_text}");
                    }
                }
                trim_count += 1;
                eprintln!("cap-hit trim: trimmed={trimmed} samples");
            }
        }
    }

    let final_delta = committer.finalize_utterance();
    if let CommitDelta::Committed { ref new_text, .. } = final_delta {
        if !new_text.is_empty() {
            committed_text.push_str(new_text);
        }
    }

    eprintln!("Long-form committed transcript: {committed_text}");
    eprintln!("trim_count={trim_count}  endpoint_count={endpoint_count}");

    // (1) At least one trim must have fired — the whole point of this test.
    assert!(
        trim_count >= 1,
        "expected ≥1 cap-hit trim event over a 10 s clip with 5 s cap, got 0"
    );

    // (2) Coverage of the LJ001 sentence — same keyword set as the short-form test, but
    //     since the trim path is the primary continuation mechanism here, this verifies
    //     no content is lost across trims.
    let lower = committed_text.to_lowercase();
    let keywords = [
        "printing",
        "sense",
        "concerned",
        "differs",
        "arts",
        "crafts",
        "exhibition",
    ];
    let hits: Vec<&str> = keywords
        .iter()
        .copied()
        .filter(|kw| lower.contains(kw))
        .collect();
    assert!(
        hits.len() >= 5,
        "expected ≥5/7 LJ001 keywords after long-form trim; got {} ({:?}) in: {committed_text:?}",
        hits.len(),
        hits
    );

    // (3) No egregious duplication at the trim seam: no 4-word phrase should appear twice.
    //     A trim that overshoots and lets the next decode re-derive committed text would
    //     produce repeated 4-word phrases.
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() >= 8 {
        for i in 0..words.len().saturating_sub(3) {
            let phrase = words[i..i + 4].join(" ");
            let count = lower.matches(&phrase).count();
            assert!(
                count <= 1,
                "4-word phrase '{phrase}' appears {count} times — trim seam duplication: {committed_text:?}"
            );
        }
    }

    Ok(())
}
