Run the stream subcommand against the LJSpeech test clip in offline mode and print NDJSON:

```bash
cargo run --release -p whisperforge -- stream \
  --model tiny.en \
  --from-file test_data/LJ001-0001_16k.wav \
  --no-realtime --json
```

Every output line must be valid NDJSON with `type` ∈ {partial, commit, endpoint, decode_metrics, shutdown}.
The last line should be `{"type":"shutdown",...}`.
