Run the LJSpeech WER benchmark (requires model files in `models/`):

```bash
cargo test --release -p whisperforge-core wer_benchmark -- --nocapture
```

Report per-utterance WER and average. Baseline is 0.8% on tiny.en — flag any regression above that.