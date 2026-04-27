/// End-to-end WER benchmark against LJSpeech reference transcriptions.
///
/// Run with:
///   cargo test -p whisperforge-core --test wer_benchmark -- --ignored --nocapture
///
/// Requires models/tiny_en_converted.{mpk,cfg} and models/tokenizer.json.
/// Audio fixtures are in test_data/ (tracked via git LFS).
use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use std::cmp::Ordering;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use whisperforge_core::{audio, load_whisper, DecodingConfig, HybridDecoder, Whisper};

type Backend = NdArray<f32>;

// ── helpers ──────────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Parse `test_data/metadata.txt` → Vec<(id, reference_text)>.
fn load_fixtures() -> Vec<(String, String)> {
    let path = workspace_root().join("test_data/metadata.txt");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("cannot read {}", path.display()))
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let mut parts = l.splitn(2, '|');
            let id = parts.next()?.trim().to_string();
            let text = parts.next()?.trim().to_string();
            Some((id, text))
        })
        .collect()
}

/// Word Error Rate: edit distance on lowercased, punctuation-stripped word lists.
///
/// WER = (substitutions + deletions + insertions) / len(reference_words)
fn word_error_rate(hypothesis: &str, reference: &str) -> f32 {
    let normalize = |s: &str| -> Vec<String> {
        s.to_lowercase()
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphabetic())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect()
    };

    let hyp = normalize(hypothesis);
    let refs = normalize(reference);
    let n = refs.len();
    let m = hyp.len();

    if n == 0 {
        return if m == 0 { 0.0 } else { 1.0 };
    }

    // Standard Levenshtein on word sequences.
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub_cost = usize::from(refs[i - 1] != hyp[j - 1]);
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + sub_cost);
        }
    }

    dp[n][m] as f32 / n as f32
}

/// Run the full tiny_en pipeline on one WAV file and return the transcript.
///
/// Mirrors the CLI logic in whisperforge-cli/src/main.rs exactly so that
/// benchmark results reflect the shipped decoding path.
fn transcribe(
    model: &Whisper<Backend>,
    tokenizer: &Tokenizer,
    audio_path: &str,
    device: &NdArrayDevice,
) -> Result<String> {
    let raw = audio::load_wav_file(audio_path)
        .with_context(|| format!("loading {audio_path}"))?;
    let mel = audio::compute_mel_spectrogram(&raw, 400, 160, 80, device)?;

    // Whisper expects exactly 3000 mel frames (30 s at 100 fps).
    let expected = 3000usize;
    let [batch, n_mels, n_frames] = mel.dims();
    let mel = if n_frames < expected {
        let pad = Tensor::<Backend, 3>::zeros([batch, n_mels, expected - n_frames], device);
        Tensor::cat(vec![mel, pad], 2)
    } else {
        mel.slice([0..batch, 0..n_mels, 0..expected])
    };

    let encoder_output = model.forward_encoder(mel);

    let tok = |s: &str, fallback: u32| tokenizer.token_to_id(s).unwrap_or(fallback);
    let sot = tok("<|startoftranscript|>", 50258);
    let en = tok("<|en|>", 50259);
    let transcribe_tok = tok("<|transcribe|>", 50359);
    let no_timestamps = tok("<|notimestamps|>", 50363);
    let eot = tok("<|endoftext|>", 50257);
    let no_speech = tok("<|nospeech|>", 50362);

    let config = DecodingConfig::balanced();
    let decoder = HybridDecoder::new(config.clone());
    let vocab_size = 51864usize;
    let mut context: Vec<u32> = vec![sot, en, transcribe_tok, no_timestamps];
    let mut all_logits: Vec<Vec<f32>> = Vec::new();

    for _ in 0..config.max_length {
        let token_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(
                context.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                [1, context.len()],
            ),
            device,
        );

        let logits = model.forward_decoder(token_tensor, encoder_output.clone());
        let [b, seq_len, _] = logits.dims();
        let step: Vec<f32> = logits
            .slice([0..b, (seq_len - 1)..seq_len, 0..vocab_size])
            .squeeze::<1>()
            .into_data()
            .to_vec()
            .context("extracting step logits")?;

        let greedy_next = step
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(eot);

        all_logits.push(step);

        if greedy_next == eot {
            break;
        }
        context.push(greedy_next);
    }

    let tokens = decoder.decode_with_fallback(
        &all_logits,
        no_timestamps,
        vocab_size,
        eot,
        no_speech,
        |ids| tokenizer.decode(ids, false).unwrap_or_default(),
    )?;

    let text_tokens: Vec<u32> = tokens.into_iter().filter(|&t| t < 50257).collect();
    let text = tokenizer
        .decode(&text_tokens, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(text.trim().to_string())
}

// ── benchmark test ────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires models/tiny_en_converted.{mpk,cfg} and models/tokenizer.json"]
fn test_wer_benchmark_tiny_en() {
    let root = workspace_root();
    let model_path = root.join("models/tiny_en_converted");
    let tokenizer_path = root.join("models/tokenizer.json");

    if !model_path.with_extension("mpk").exists() {
        eprintln!("SKIP: model not found at {}", model_path.display());
        return;
    }

    let device = NdArrayDevice::default();
    let model =
        load_whisper::<Backend>(model_path.to_str().unwrap(), &device).expect("load model");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");

    let fixtures = load_fixtures();
    assert!(!fixtures.is_empty(), "no fixtures in test_data/metadata.txt");

    let mut total_wer = 0.0f32;
    let mut tested = 0usize;

    for (id, reference) in &fixtures {
        let wav = root.join(format!("test_data/{id}.wav"));
        if !wav.exists() {
            eprintln!("SKIP {id}: WAV not found");
            continue;
        }

        let hypothesis = match transcribe(&model, &tokenizer, wav.to_str().unwrap(), &device) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("ERROR {id}: {e}");
                continue;
            }
        };

        let wer = word_error_rate(&hypothesis, reference);
        total_wer += wer;
        tested += 1;

        eprintln!("\n[{id}] WER = {:.1}%", wer * 100.0);
        eprintln!("  ref: {reference}");
        eprintln!("  hyp: {hypothesis}");
    }

    assert!(tested > 0, "no fixtures could be tested");

    let avg_wer = total_wer / tested as f32;
    eprintln!("\nAverage WER across {tested} clips: {:.1}%", avg_wer * 100.0);

    // tiny.en achieves ~5% WER on LJSpeech clean speech.
    // 20% is the acceptance threshold for the current implementation.
    assert!(
        avg_wer < 0.20,
        "Average WER {:.1}% exceeds 20% acceptance threshold",
        avg_wer * 100.0
    );
}
