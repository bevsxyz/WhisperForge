use anyhow::{Context, Result};
use burn::{
    backend::NdArray,
    module::Param,
    tensor::{Tensor, TensorData},
};
use burn_ndarray::NdArrayDevice;
use clap::Parser;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model ID (e.g., openai/whisper-tiny)
    #[arg(long, default_value = "openai/whisper-tiny")]
    model_id: String,

    /// Output file path (e.g., models/whisper-tiny.mpk)
    #[arg(long)]
    output: String,

    /// Local safetensors file path (optional, if not provided will download)
    #[arg(long)]
    local_safetensors: Option<String>,
}

type B = NdArray<f32>;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = NdArrayDevice::default();

    let output_path = std::path::Path::new(&args.output);
    let tokenizer_output = output_path.with_file_name(format!(
        "{}-tokenizer.json",
        output_path.file_stem().unwrap().to_string_lossy()
    ));

    if let Some(local_path) = &args.local_safetensors {
        // Using local file - check if tokenizer exists locally
        let local_tokenizer = std::path::Path::new(local_path).with_file_name("tokenizer.json");
        if local_tokenizer.exists() {
            std::fs::copy(&local_tokenizer, &tokenizer_output)?;
            println!("Copied local tokenizer to: {:?}", tokenizer_output);
        } else {
            println!("Warning: No local tokenizer found, skipping tokenizer copy");
        }
    } else {
        // Download from HF
        println!("Downloading model: {}", args.model_id);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(args.model_id.clone(), RepoType::Model));

        // Download tokenizer first to ensure it exists
        println!("Downloading tokenizer...");
        let tokenizer_path = repo.get("tokenizer.json").await?;
        std::fs::copy(&tokenizer_path, &tokenizer_output)?;
        println!("Saved tokenizer to: {:?}", tokenizer_output);
    }

    let (weights_path, tensor_data) = if let Some(local_path) = &args.local_safetensors {
        println!("Using local safetensors file: {}", local_path);
        let data = std::fs::read(local_path)?;
        (std::path::PathBuf::from(local_path), data)
    } else {
        println!("Downloading model: {}", args.model_id);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(args.model_id.clone(), RepoType::Model));

        let path = repo.get("model.safetensors").await?;
        println!("Loading downloaded weights from: {:?}", path);
        let data = std::fs::read(&path)?;
        (path, data)
    };

    let tensors = SafeTensors::deserialize(&tensor_data)?;

    // List available tensors for debugging
    println!("Available tensors:");
    for name in tensors.names() {
        if let Ok(view) = tensors.tensor(name) {
            println!(
                "  {} - shape: {:?}, dtype: {:?}",
                name,
                view.shape(),
                view.dtype()
            );
        }
    }

    println!("\nConverting weights...");

    // For now, we'll create a simple demonstration of tensor loading
    // Full model record construction requires matching the WhisperModelRecord structure
    // which depends on the specific Burn 0.20 API for transformer records

    demonstrate_tensor_loading(&tensors, &device)?;

    println!("\nNote: Full model conversion requires implementing WhisperModelRecord");
    println!("      construction compatible with Burn 0.20 transformer API.");
    println!("      The current implementation demonstrates tensor loading capabilities.");

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("Done!");
    Ok(())
}

/// Demonstrate loading tensors from safetensors format
fn demonstrate_tensor_loading(tensors: &SafeTensors, device: &NdArrayDevice) -> Result<()> {
    // Try to load a few key tensors to verify the conversion pipeline works
    let test_tensors = [
        "encoder.conv1.weight",
        "encoder.conv1.bias",
        "decoder.embed_tokens.weight",
    ];

    for name in test_tensors {
        match load_tensor_dynamic(tensors, name, device) {
            Ok((shape, dtype)) => {
                println!("  ✓ Loaded {}: shape={:?}, dtype={}", name, shape, dtype);
            }
            Err(e) => {
                println!("  ✗ Failed to load {}: {}", name, e);
            }
        }
    }

    Ok(())
}

/// Load a tensor dynamically and return its shape and dtype info
fn load_tensor_dynamic(
    tensors: &SafeTensors,
    name: &str,
    _device: &NdArrayDevice,
) -> Result<(Vec<usize>, String)> {
    let view = tensors
        .tensor(name)
        .context(format!("Tensor not found: {}", name))?;

    let shape = view.shape().to_vec();
    let dtype = format!("{:?}", view.dtype());

    // Actually load the tensor data to verify it works
    let _data = tensor_data_from_view(&view);

    Ok((shape, dtype))
}

/// Load a 1D tensor parameter
#[allow(dead_code)]
fn load_param_1d(
    tensors: &SafeTensors,
    name: &str,
    device: &NdArrayDevice,
) -> Result<Param<Tensor<B, 1>>> {
    let tensor = load_tensor_1d(tensors, name, device)?;
    Ok(Param::from_tensor(tensor))
}

/// Load a 2D tensor parameter
#[allow(dead_code)]
fn load_param_2d(
    tensors: &SafeTensors,
    name: &str,
    device: &NdArrayDevice,
) -> Result<Param<Tensor<B, 2>>> {
    let tensor = load_tensor_2d(tensors, name, device)?;
    Ok(Param::from_tensor(tensor))
}

/// Load a 3D tensor parameter
#[allow(dead_code)]
fn load_param_3d(
    tensors: &SafeTensors,
    name: &str,
    device: &NdArrayDevice,
) -> Result<Param<Tensor<B, 3>>> {
    let tensor = load_tensor_3d(tensors, name, device)?;
    Ok(Param::from_tensor(tensor))
}

/// Load a 2D tensor parameter with transposition
#[allow(dead_code)]
fn load_param_2d_transposed(
    tensors: &SafeTensors,
    name: &str,
    device: &NdArrayDevice,
) -> Result<Param<Tensor<B, 2>>> {
    let tensor = load_tensor_2d(tensors, name, device)?;
    Ok(Param::from_tensor(tensor.transpose()))
}

fn load_tensor_1d(
    tensors: &SafeTensors,
    name: &str,
    device: &NdArrayDevice,
) -> Result<Tensor<B, 1>> {
    let view = tensors
        .tensor(name)
        .context(format!("Tensor not found: {}", name))?;
    let data = tensor_data_from_view(&view);
    Ok(Tensor::from_data(data, device))
}

fn load_tensor_2d(
    tensors: &SafeTensors,
    name: &str,
    device: &NdArrayDevice,
) -> Result<Tensor<B, 2>> {
    let view = tensors
        .tensor(name)
        .context(format!("Tensor not found: {}", name))?;
    let data = tensor_data_from_view(&view);
    Ok(Tensor::from_data(data, device))
}

fn load_tensor_3d(
    tensors: &SafeTensors,
    name: &str,
    device: &NdArrayDevice,
) -> Result<Tensor<B, 3>> {
    let view = tensors
        .tensor(name)
        .context(format!("Tensor not found: {}", name))?;
    let data = tensor_data_from_view(&view);
    Ok(Tensor::from_data(data, device))
}

fn tensor_data_from_view(view: &TensorView) -> TensorData {
    let shape = view.shape().to_vec();
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|chunk: &[u8]| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            TensorData::new(data, shape)
        }
        safetensors::Dtype::F16 => {
            let data: Vec<f32> = view
                .data()
                .chunks_exact(2)
                .map(|chunk: &[u8]| {
                    let val = half::f16::from_le_bytes(chunk.try_into().unwrap());
                    val.to_f32()
                })
                .collect();
            TensorData::new(data, shape)
        }
        _ => panic!("Unsupported dtype: {:?}", view.dtype()),
    }
}
