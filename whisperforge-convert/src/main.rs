use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn_ndarray::NdArrayDevice;
use clap::{Parser, ValueEnum};
use hf_hub::{Repo, RepoType, api::tokio::Api};
use whisperforge_convert::{Precision, convert_openai_to_burn};

#[derive(ValueEnum, Clone, Copy, Default, Debug)]
enum Quantize {
    #[default]
    None,
    Int8,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Convert OpenAI Whisper safetensors to Burn .mpk format"
)]
struct Args {
    /// HuggingFace model ID (e.g., openai/whisper-tiny.en)
    #[arg(long, default_value = "openai/whisper-tiny.en")]
    model_id: String,

    /// Output path without extension (e.g., models/tiny_en_converted)
    #[arg(long)]
    output: String,

    /// Use a local safetensors file instead of downloading
    #[arg(long)]
    local_safetensors: Option<String>,

    /// Quantization mode
    #[arg(long, value_enum, default_value_t = Quantize::None)]
    quantize: Quantize,
}

type B = NdArray<f32>;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = NdArrayDevice::default();

    let output_path = std::path::Path::new(&args.output);

    // Both the CLI and benchmark test expect {output_dir}/tokenizer.json.
    let tokenizer_dest = output_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("tokenizer.json");

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).context("creating output directory")?;
    }

    let safetensors_path = if let Some(local_path) = &args.local_safetensors {
        let local_tokenizer = std::path::Path::new(local_path).with_file_name("tokenizer.json");
        if local_tokenizer.exists() {
            // Canonicalize to avoid copying a file onto itself when source == dest.
            let src = local_tokenizer
                .canonicalize()
                .unwrap_or(local_tokenizer.clone());
            let dst = tokenizer_dest
                .canonicalize()
                .unwrap_or(tokenizer_dest.clone());
            if src != dst {
                std::fs::copy(&local_tokenizer, &tokenizer_dest)
                    .context("copying local tokenizer.json")?;
                println!("Copied tokenizer to: {}", tokenizer_dest.display());
            } else {
                println!(
                    "Tokenizer already at destination: {}",
                    tokenizer_dest.display()
                );
            }
        } else {
            println!("Warning: no tokenizer.json found next to local safetensors file");
        }
        std::path::PathBuf::from(local_path)
    } else {
        println!("Downloading from HuggingFace: {}", args.model_id);
        let api = Api::new().context("initialising HuggingFace API")?;
        let repo = api.repo(Repo::new(args.model_id.clone(), RepoType::Model));

        let tok_cache = repo
            .get("tokenizer.json")
            .await
            .context("downloading tokenizer.json")?;
        std::fs::copy(&tok_cache, &tokenizer_dest).context("saving tokenizer.json")?;
        println!("Saved tokenizer to: {}", tokenizer_dest.display());

        repo.get("model.safetensors")
            .await
            .context("downloading model.safetensors")?
    };

    println!(
        "Converting {} → {}.mpk …",
        safetensors_path.display(),
        output_path.display()
    );

    let precision = match args.quantize {
        Quantize::None => Precision::Fp32,
        Quantize::Int8 => Precision::Int8,
    };

    convert_openai_to_burn::<B>(&safetensors_path, output_path, &device, precision)?;

    println!("Done!");
    Ok(())
}
