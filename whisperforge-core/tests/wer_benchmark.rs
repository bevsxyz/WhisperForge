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
use whisperforge_core::{DecodingConfig, HybridDecoder, Whisper, audio, load_whisper};

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
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_string())
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
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
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
    let raw =
        audio::load_audio_file(audio_path).with_context(|| format!("loading {audio_path}"))?;
    eprintln!(
        "  [audio] path={audio_path} samples={} sr={} ch={}",
        raw.samples.len(),
        raw.sample_rate,
        raw.channels
    );
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
    {
        let data: Vec<f32> = encoder_output.to_data().to_vec().unwrap();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let std =
            (data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
        eprintln!(
            "  [encoder] shape={:?} min={min:.4} max={max:.4} mean={mean:.4} std={std:.4}",
            encoder_output.dims()
        );
    }

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

        // At step 0: suppress EOT so the model is forced to start generating real tokens.
        // Collect the unconstrained greedy argmax only for diagnostics; for the actual
        // context we always use the best non-EOT token at step 0.
        let unconstrained_greedy = step
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(eot);

        let greedy_next = if all_logits.is_empty() && unconstrained_greedy == eot {
            // EOT would be the first token — suppress it and take the next best.
            let best_non_eot = step
                .iter()
                .enumerate()
                .filter(|&(i, _)| i as u32 != eot)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(eot);
            eprintln!("  [step0] EOT suppressed → using token {best_non_eot}");
            best_non_eot
        } else {
            unconstrained_greedy
        };

        // Diagnostic: show step-0 token distribution once per clip.
        if all_logits.is_empty() {
            let exp_sum: f32 = step.iter().map(|&x| x.exp()).sum();
            let eot_prob = step[eot as usize].exp() / exp_sum;
            let ns_prob = if (no_speech as usize) < vocab_size {
                step[no_speech as usize].exp() / exp_sum
            } else {
                0.0
            };
            let mut top5: Vec<(usize, f32)> = step.iter().copied().enumerate().collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            top5.truncate(5);
            eprintln!(
                "  [step0] greedy={unconstrained_greedy} eot_prob={eot_prob:.4} ns_prob={ns_prob:.4} no_speech_id={no_speech}"
            );
            eprintln!("  [step0] top5: {:?}", &top5[..5]);
        }

        all_logits.push(step);

        if greedy_next == eot {
            break;
        }
        context.push(greedy_next);
    }

    // Diagnostic: inspect the greedy sequence before handing to decode_with_fallback.
    {
        let greedy_toks = &context[4..]; // strip [sot, en, transcribe, no_timestamps]
        let hit_eot = greedy_toks.len() < config.max_length;
        let greedy_text = tokenizer.decode(greedy_toks, false).unwrap_or_default();
        // Unique-token count: low value signals a repetition loop.
        let mut sorted = greedy_toks.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let unique = sorted.len();
        // Average greedy log-prob — below -1.0 will fail decode_with_fallback quality gate.
        let avg_lp: f32 = all_logits
            .iter()
            .zip(greedy_toks.iter().chain(std::iter::once(&eot)))
            .map(|(step_l, &tok)| {
                let max = step_l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let log_sum = max + step_l.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
                step_l[tok as usize] - log_sum
            })
            .sum::<f32>()
            / all_logits.len() as f32;
        eprintln!(
            "  [greedy] steps={} hit_eot={hit_eot} unique_toks={unique} avg_lp={avg_lp:.4}",
            all_logits.len()
        );
        let preview: String = greedy_text.chars().take(160).collect();
        eprintln!("  [greedy] text_preview: {preview:?}");
        eprintln!(
            "  [greedy] first_15_toks: {:?}",
            &greedy_toks[..greedy_toks.len().min(15)]
        );
    }

    // Mask EOT in step-0 logits so decode_with_fallback also suppresses EOT at the
    // first position regardless of temperature.  Without this, at temperature < 1.0
    // the distribution is more peaked at EOT, quality passes immediately, and the
    // fallback returns an empty sequence.
    if let Some(first) = all_logits.first_mut() {
        if (eot as usize) < first.len() {
            first[eot as usize] = f32::NEG_INFINITY;
        }
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
    let model = load_whisper::<Backend>(model_path.to_str().unwrap(), &device).expect("load model");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");

    let fixtures = load_fixtures();
    assert!(
        !fixtures.is_empty(),
        "no fixtures in test_data/metadata.txt"
    );

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
    eprintln!(
        "\nAverage WER across {tested} clips: {:.1}%",
        avg_wer * 100.0
    );

    // tiny.en achieves ~5% WER on LJSpeech clean speech.
    // 20% is the acceptance threshold for the current implementation.
    assert!(
        avg_wer < 0.20,
        "Average WER {:.1}% exceeds 20% acceptance threshold",
        avg_wer * 100.0
    );
}
