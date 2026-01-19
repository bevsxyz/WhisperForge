// Quick utility to inspect safetensors files and list tensor names

use anyhow::Result;
use safetensors::SafeTensors;
use std::path::Path;

pub fn inspect_safetensors(path: &Path) -> Result<Vec<(String, Vec<usize>, String)>> {
    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let mut result = Vec::new();
    for (name, tensor) in tensors.tensors() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = format!("{:?}", tensor.dtype());
        result.push((name.to_string(), shape, dtype));
    }

    // Sort by name for easier reading
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_inspect_openai_model() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/tiny_en_openai.safetensors");

        if !model_path.exists() {
            eprintln!("Skipping: model not found at {:?}", model_path);
            return;
        }

        let tensors = inspect_safetensors(&model_path).unwrap();

        println!("\n=== OpenAI Whisper tiny.en Tensor Names ===\n");
        for (name, shape, dtype) in &tensors {
            println!("{:60} {:20?} {}", name, shape, dtype);
        }
        println!("\nTotal tensors: {}", tensors.len());
    }
}
