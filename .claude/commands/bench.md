Run the LJSpeech WER benchmark (requires model files in `models/`):

```bash
cargo test --release -p whisperforge-core wer_benchmark -- --nocapture
```

Report per-utterance WER and average. Baseline is 0.8% on tiny.en — flag any regression above that.

For streaming **latency** (not WER), profile per-window p50/p99 for the encoder/decode/total stages:

```bash
cargo run --release -p whisperforge --bin stream_bench -- \
  --audio test_data/LJ001-0001_16k.wav --device cpu
```

Prints one JSON line. Note it runs the heavy config (28 s window, 128 tokens), so `total_window_ms` is well above the live 5 s/32-token defaults. For live real-time keep-up, use `scripts/uat-stream.ps1` (counts dropped samples) instead.