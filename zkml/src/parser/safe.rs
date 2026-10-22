//! Safetensors parser for LLM models (focus: Gemma3).
//!
//! This module provides a lightweight loader over `.safetensors` files similar in spirit
//! to the GGUF loader in `parser::gguf`. It offers:
//! - prefix-based subscoping via `pp()`
//! - on-demand tensor loading as `crate::Tensor<f32>` with dtype conversion (F32/F16/BF16)
//! - optional metadata access via the `__metadata__` string map
//!
//! Notes:
//! - The SafeTensors header only supports `__metadata__` as a string-to-string map, so any
//!   numeric metadata will be parsed from strings on-demand when using `metadata::<T>()`.
//! - This loader does not introduce any model-specific assumptions; Gemma3-specific
//!   conventions are handled by higher-level code.

use std::{
    collections::BTreeMap,
    fmt::Display,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, bail};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json as json;

use crate::{
    Shape, Tensor,
    layers::transformer::normalisation::rmsnorm::RMSNorm,
    parser::{ModelNameProvider, llm::LLMConfig},
    tensor::KeyedTensor,
};

/// Contains the path to the SafeTensors files. This is usually loaded from a hugging face model repository.
#[derive(Clone, Debug)]
pub struct RawSafeTensors {
    model: PathBuf,
    tokenizer: PathBuf,
    config: PathBuf,
}

impl Display for RawSafeTensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SafeTensors: {}", self.model.display())
    }
}

impl RawSafeTensors {
    pub fn new<I: AsRef<Path>>(model: I, tokenizer: I, config: I) -> Self {
        Self {
            model: model.as_ref().to_path_buf(),
            tokenizer: tokenizer.as_ref().to_path_buf(),
            config: config.as_ref().to_path_buf(),
        }
    }

    /// Resolve a local Hugging Face checkpoint, including indexed shards.
    pub fn from_directory(directory: impl AsRef<Path>) -> anyhow::Result<Self> {
        let directory = directory.as_ref();
        let model = if directory.join("model.safetensors.index.json").is_file() {
            directory.join("model.safetensors.index.json")
        } else {
            directory.join("model.safetensors")
        };
        let tokenizer = directory.join("tokenizer.json");
        let config = directory.join("config.json");
        for path in [&model, &tokenizer, &config] {
            anyhow::ensure!(
                path.is_file(),
                "Missing local model file: {}",
                path.display()
            );
        }
        Ok(Self::new(model, tokenizer, config))
    }
    /// Download required files from a Hugging Face repo id (e.g., "google/gemma-3-270m-it")
    /// into `destination_folder` and return a `RawSafeTensors` pointing at them.
    /// This fetches:
    /// - model.safetensors
    /// - tokenizer.json
    /// - config.json
    pub fn from_hugging_face(
        repo: &str,
        destination_folder: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        use std::{fs, io};
        if std::env::var("IN_CI").is_ok() {
            bail!("not downloading model in CI");
        }

        let dest = destination_folder.as_ref();
        if !dest.exists() {
            fs::create_dir_all(dest).with_context(|| format!("create dir {}", dest.display()))?;
        }

        // helper to download a single file
        let download = |filename: &str| -> anyhow::Result<PathBuf> {
            let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
            let path = dest.join(filename);
            let resp = ureq::get(&url)
                .call()
                .map_err(|e| anyhow::anyhow!("GET {} failed: {}", url, e))?;
            let mut body = resp.into_body();
            let mut reader = body.as_reader();
            let mut writer =
                fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
            io::copy(&mut reader, &mut writer)
                .with_context(|| format!("writing {}", path.display()))?;
            Ok(path)
        };

        let model = download("model.safetensors")?;
        let tokenizer = download("tokenizer.json")?;
        let config = download("config.json")?;

        Ok(Self::new(model, tokenizer, config))
    }

    /// Cached variant that stores files under the test cache directory (model_cache/<repo>/)
    /// and only downloads when missing. Paths are resolved via `parser::file_cache::from_cache`.
    pub fn from_hugging_face_cached(repo: &str) -> anyhow::Result<Self> {
        use crate::parser::file_cache;
        use std::fs;

        let rel_model = format!("{repo}/model.safetensors");
        let rel_tokenizer = format!("{repo}/tokenizer.json");
        let rel_config = format!("{repo}/config.json");

        let model_path = file_cache::cache_path(&rel_model);
        let tokenizer_path = file_cache::cache_path(&rel_tokenizer);
        let config_path = file_cache::cache_path(&rel_config);

        if !(model_path.exists() && tokenizer_path.exists() && config_path.exists()) {
            let dest = PathBuf::from("model_cache").join(repo);
            if !dest.exists() {
                fs::create_dir_all(&dest)
                    .with_context(|| format!("create dir {}", dest.display()))?;
            }
            let _ = Self::from_hugging_face(repo, &dest)?;
        }

        let model = file_cache::from_cache(&rel_model)?;
        let tokenizer = file_cache::from_cache(&rel_tokenizer)?;
        let config = file_cache::from_cache(&rel_config)?;

        Ok(Self::new(model, tokenizer, config))
    }

    /// Path to the `model.safetensors` file
    pub fn model_path(&self) -> &Path {
        &self.model
    }

    /// Path to the `tokenizer.json` file
    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer
    }

    /// Path to the `config.json` file
    pub fn config_path(&self) -> &Path {
        &self.config
    }

    /// Create a `FileTensorLoader` over the model file
    pub fn loader(&self) -> anyhow::Result<FileTensorLoader> {
        FileTensorLoader::from_path(&self.model)
    }

    /// Read and parse the HuggingFace `config.json` file as raw JSON
    pub fn read_config_json(&self) -> anyhow::Result<ConfigJSON> {
        let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(&self.config)?)
            .with_context(|| "parsing config.json".to_string())?;
        Ok(ConfigJSON(v))
    }
}

#[derive(Clone, Debug, derive_more::From, derive_more::Into)]
pub struct ConfigJSON(serde_json::Value);

impl ConfigJSON {
    pub fn get<T, I: serde_json::value::Index>(&self, key: I) -> Option<T>
    where
        T: FromValue,
    {
        self.0.get(key).and_then(|v| T::from_value(v))
    }
}

pub trait FromValue {
    fn from_value(v: &serde_json::Value) -> Option<Self>
    where
        Self: Sized;
    fn is_correct_type(v: &serde_json::Value) -> bool;
}

impl FromValue for String {
    fn from_value(v: &serde_json::Value) -> Option<Self> {
        v.as_str().map(|s| s.to_string())
    }
    fn is_correct_type(v: &serde_json::Value) -> bool {
        v.is_string()
    }
}

impl FromValue for usize {
    fn from_value(v: &serde_json::Value) -> Option<Self> {
        v.as_u64().map(|v| v as usize)
    }
    fn is_correct_type(v: &serde_json::Value) -> bool {
        v.is_u64()
    }
}

impl FromValue for f32 {
    fn from_value(v: &serde_json::Value) -> Option<Self> {
        v.as_f64().map(|v| v as f32)
    }
    fn is_correct_type(v: &serde_json::Value) -> bool {
        v.is_f64()
    }
}

impl FromValue for f64 {
    fn from_value(v: &serde_json::Value) -> Option<Self> {
        v.as_f64()
    }
    fn is_correct_type(v: &serde_json::Value) -> bool {
        v.is_f64()
    }
}

impl FromValue for bool {
    fn from_value(v: &serde_json::Value) -> Option<Self> {
        v.as_bool()
    }
    fn is_correct_type(v: &serde_json::Value) -> bool {
        v.is_boolean()
    }
}

impl FromValue for u64 {
    fn from_value(v: &serde_json::Value) -> Option<Self> {
        v.as_u64()
    }
    fn is_correct_type(v: &serde_json::Value) -> bool {
        v.is_u64()
    }
}

impl<T> FromValue for Vec<T>
where
    T: FromValue,
{
    fn from_value(v: &serde_json::Value) -> Option<Self> {
        v.as_array().map(|arr| {
            arr.iter()
                .map(|v| T::from_value(v).unwrap())
                .collect::<Vec<T>>()
        })
    }
    fn is_correct_type(v: &serde_json::Value) -> bool {
        v.is_array() && v.as_array().unwrap().iter().all(|v| T::is_correct_type(v))
    }
}

impl ModelNameProvider for RawSafeTensors {
    fn model_metadata(&self) -> anyhow::Result<Vec<String>> {
        let v = self
            .read_config_json()
            .with_context(|| "reading model metadata from config.json")?;
        let mut names = Vec::new();
        if let Some(s) = v.get::<String, _>("model_type") {
            names.push(s.to_string());
        }
        if let Some(arr) = v.get::<Vec<String>, _>("architectures") {
            names.extend(arr);
        }
        if names.is_empty() {
            bail!("model name not found in config.json");
        }
        Ok(names)
    }
}

/// Loader for tensors stored in a SafeTensors file.
///
/// The loader maintains an internal prefix to support hierarchical naming
/// (e.g., `blk.0.attn_q.weight`), mimicking the ergonomics of the GGUF loader.
#[derive(Clone)]
pub struct FileTensorLoader {
    /// Entire file contents held in memory to satisfy the borrowing
    /// requirements of `safetensors::SafeTensors` views.
    bytes: Arc<Vec<u8>>,
    indexed_tensors: Option<Arc<BTreeMap<String, FileTensorLoader>>>,
    /// Current prefix for tensor names. Appended in front of requested keys.
    current_prefix: String,
}

impl FileTensorLoader {
    /// Build a loader from a file path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return RawSafeTensors::from_directory(path)?.loader();
        }
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            return Self::from_index(path);
        }
        let mut f = File::open(path)
            .with_context(|| format!("Failed to open SafeTensors file: {}", path.display()))?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)
            .with_context(|| "Failed to read SafeTensors file")?;
        Ok(Self {
            bytes: Arc::new(bytes),
            indexed_tensors: None,
            current_prefix: String::new(),
        })
    }

    fn from_index(path: &Path) -> anyhow::Result<Self> {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let index: Index =
            serde_json::from_reader(File::open(path)?).context("Invalid SafeTensors index")?;
        anyhow::ensure!(
            !index.weight_map.is_empty(),
            "SafeTensors index has no tensors"
        );
        let directory = path
            .parent()
            .context("SafeTensors index has no parent directory")?;
        let mut shards = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for (name, shard) in index.weight_map {
            let relative = Path::new(&shard);
            anyhow::ensure!(
                relative
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_))),
                "SafeTensors shard must be a relative file path: {shard}"
            );
            anyhow::ensure!(
                relative
                    .extension()
                    .is_some_and(|extension| extension == "safetensors"),
                "Indexed shard must be a SafeTensors file: {shard}"
            );
            if !shards.contains_key(&shard) {
                shards.insert(shard.clone(), Self::from_path(directory.join(&shard))?);
            }
            let loader = shards.get(&shard).expect("loaded shard").clone();
            loader
                .get_tensor_shape(&name)
                .with_context(|| format!("Index tensor {name} missing from {shard}"))?;
            tensors.insert(name, loader);
        }
        Ok(Self {
            bytes: Arc::new(Vec::new()),
            indexed_tensors: Some(Arc::new(tensors)),
            current_prefix: String::new(),
        })
    }

    #[cfg(test)]
    pub fn print_keys(&self) {
        let (meta, tensors) = self.meta_and_tensor_keys();

        let header = self.header_json().ok().unwrap();
        println!("raw header: {:?}", header);
        println!("metadata:");
        for k in meta {
            println!("\t- {}", k);
        }
        println!("tensor_infos:");
        for k in tensors {
            println!("\t- {}", k);
        }
    }

    /// Create a subscope by appending a `prefix_extension` to the current prefix.
    pub fn pp(&self, prefix_extension: &str) -> Self {
        Self {
            bytes: Arc::clone(&self.bytes),
            indexed_tensors: self.indexed_tensors.clone(),
            current_prefix: format!("{}{}", self.current_prefix, prefix_extension),
        }
    }

    /// Return the shape of a tensor without materializing it.
    pub fn get_tensor_shape(&self, name: &str) -> anyhow::Result<Shape> {
        let full = self.full_name(name);
        if let Some(tensors) = &self.indexed_tensors {
            return tensors
                .get(&full)
                .with_context(|| format!("tensor not indexed: {full}"))?
                .get_tensor_shape(&full);
        }
        let st = self.deserialize()?;
        let view = st
            .tensor(&full)
            .with_context(|| format!("tensor not found: {full}"))?;
        Ok(Shape::new(view.shape().to_vec()))
    }

    /// Load a tensor and convert to `Tensor<f32>` (supports F32/F16/BF16).
    pub fn get_tensor(&self, name: &str) -> anyhow::Result<KeyedTensor<f32>> {
        let full = self.full_name(name);
        if let Some(tensors) = &self.indexed_tensors {
            return tensors
                .get(&full)
                .with_context(|| format!("tensor not indexed: {full}"))?
                .get_tensor(&full);
        }
        let st = self.deserialize()?;
        let view = st
            .tensor(&full)
            .with_context(|| format!("tensor not found: {full}"))?;

        let shape = Shape::new(view.shape().to_vec());
        let data_f32 = view_to_f32(view)?;
        Ok(KeyedTensor::new(full, Tensor::new(shape, data_f32)?))
    }

    /// Access metadata value converted to type `T` if possible. Values are read from
    /// the `__metadata__` string map and parsed as needed.
    pub fn metadata<T>(&self, key: &str) -> Option<T>
    where
        T: FromMeta,
    {
        let header = self.header_json().ok()?;
        let meta = header.get("__metadata__")?.as_object()?;
        let v = meta.get(key)?.as_str()?;
        T::from_meta(v)
    }

    /// Return the raw metadata string for a given key, if any.
    pub fn raw_metadata(&self, key: &str) -> Option<String> {
        let header = self.header_json().ok()?;
        let meta = header.get("__metadata__")?.as_object()?;
        meta.get(key)?.as_str().map(|s| s.to_string())
    }

    /// List sorted metadata keys and tensor keys for inspection/debugging.
    pub fn meta_and_tensor_keys(&self) -> (Vec<String>, Vec<String>) {
        if let Some(tensors) = &self.indexed_tensors {
            return (Vec::new(), tensors.keys().cloned().collect());
        }
        let header = match self.header_json() {
            Ok(h) => h,
            Err(_) => return (vec![], vec![]),
        };
        let mut meta_keys = header
            .get("__metadata__")
            .and_then(|v| v.as_object())
            .map(|m| m.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut tensor_keys = header
            .as_object()
            .map(|o| {
                o.keys()
                    .filter(|k| k.as_str() != "__metadata__")
                    .map(|k| k.to_string())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        meta_keys.sort();
        tensor_keys.sort();
        (meta_keys, tensor_keys)
    }

    pub fn current_prefix(&self) -> &str {
        &self.current_prefix
    }

    fn full_name(&self, name: &str) -> String {
        format!("{}{}", self.current_prefix, name)
    }

    fn deserialize(&self) -> anyhow::Result<SafeTensors<'_>> {
        // Parsing the header is fast relative to IO; we keep the code simple and re-create the view.
        SafeTensors::deserialize(&self.bytes).map_err(anyhow::Error::from)
    }

    fn header_json(&self) -> anyhow::Result<json::Value> {
        anyhow::ensure!(self.bytes.len() >= 8, "safetensors file too small");
        let header_len = u64::from_le_bytes(self.bytes[0..8].try_into()?) as usize;
        anyhow::ensure!(
            self.bytes.len() >= 8 + header_len,
            "safetensors header length exceeds file length"
        );
        let header_slice = &self.bytes[8..8 + header_len];
        let v: json::Value = json::from_slice(header_slice)?;
        Ok(v)
    }
}

/// Helper trait for parsing metadata values from strings.
pub trait FromMeta: Sized {
    /// Attempt to parse a value of `Self` from a string metadata entry.
    fn from_meta(s: &str) -> Option<Self>;
}

impl FromMeta for String {
    fn from_meta(s: &str) -> Option<Self> {
        Some(s.to_string())
    }
}

impl FromMeta for usize {
    fn from_meta(s: &str) -> Option<Self> {
        s.parse::<u64>().ok().map(|v| v as usize)
    }
}

impl FromMeta for u32 {
    fn from_meta(s: &str) -> Option<Self> {
        s.parse::<u64>().ok().map(|v| v as u32)
    }
}

impl FromMeta for f32 {
    fn from_meta(s: &str) -> Option<Self> {
        s.parse::<f32>().ok()
    }
}

impl FromMeta for f64 {
    fn from_meta(s: &str) -> Option<Self> {
        s.parse::<f64>().ok()
    }
}

impl FromMeta for bool {
    fn from_meta(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }
}

fn view_to_f32(view: TensorView<'_>) -> anyhow::Result<Vec<f32>> {
    let bytes = view.data();
    let dt = view.dtype();
    let numel: usize = view.shape().iter().product();
    match dt {
        Dtype::F32 => {
            let needed = numel
                .checked_mul(4)
                .context("overflow computing f32 bytes")?;
            anyhow::ensure!(
                bytes.len() == needed,
                "Invalid f32 tensor byte length: {} vs expected {}",
                bytes.len(),
                needed
            );
            let mut out = Vec::with_capacity(numel);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let needed = numel
                .checked_mul(2)
                .context("overflow computing f16 bytes")?;
            anyhow::ensure!(
                bytes.len() == needed,
                "Invalid f16 tensor byte length: {} vs expected {}",
                bytes.len(),
                needed
            );
            let mut out = Vec::with_capacity(numel);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let needed = numel
                .checked_mul(2)
                .context("overflow computing bf16 bytes")?;
            anyhow::ensure!(
                bytes.len() == needed,
                "Invalid bf16 tensor byte length: {} vs expected {}",
                bytes.len(),
                needed
            );
            let mut out = Vec::with_capacity(numel);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        other => bail!("Unsupported dtype for conversion to f32: {other:?}"),
    }
}

impl RMSNorm<f32> {
    /// Build an RMSNorm layer from a SafeTensors loader scoped to a `..._` prefix
    /// similar to the GGUF loader. If `stack` is true, the alpha vector is stacked
    /// across heads (temporary hack used by Gemma3 GQA emulation).
    pub fn from_safe(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        // Try common HF naming first ("weight"); fall back to our GGUF-style ("norm.weight")

        let alpha = loader
            .get_tensor("weight")
            .or_else(|_| loader.get_tensor("norm.weight"))?;
        let eps = c.norm_epsilon;
        // If alpha is all ones or zeroes we can just set it to None
        let trivial_alpha = alpha.data().iter().all(|&x| x == 1.0 || x == 0.0f32);

        if trivial_alpha {
            RMSNorm::new(None, eps, Some(alpha.shape().dim(-1)))
        } else {
            RMSNorm::new(Some(alpha.into()), eps, None)
        }
    }
}

#[cfg(test)]
#[path = "qwen35_loader_tests.rs"]
mod qwen35_loader_tests;

#[cfg(test)]
mod tests {
    use crate::{init_test_logging, parser::llm::models::gemma3::safe_tests::GEMMA3_SAFE_MODEL};

    use super::*;

    fn write_tiny_shard(path: &Path, name: &str, values: &[f32]) -> anyhow::Result<()> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![values.len()], &bytes)?;
        safetensors::tensor::serialize_to_file([(name, view)], None, path)?;
        Ok(())
    }

    #[test]
    fn indexed_safe_loader_resolves_multiple_shards_and_prefixes() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        write_tiny_shard(
            &directory.path().join("first.safetensors"),
            "model.first",
            &[1.0, 2.0],
        )?;
        write_tiny_shard(
            &directory.path().join("second.safetensors"),
            "model.second",
            &[3.0, 4.0],
        )?;
        let index = directory.path().join("model.safetensors.index.json");
        std::fs::write(
            &index,
            serde_json::to_vec(&serde_json::json!({"weight_map": {
            "model.first":"first.safetensors", "model.second":"second.safetensors"}}))?,
        )?;
        let loader = FileTensorLoader::from_path(&index)?.pp("model.");
        assert_eq!(loader.get_tensor("first")?.data(), &[1.0, 2.0]);
        assert_eq!(loader.get_tensor("second")?.data(), &[3.0, 4.0]);
        assert_eq!(loader.get_tensor_shape("second")?, Shape::from(vec![2]));
        assert!(loader.get_tensor("missing").is_err());
        assert_eq!(loader.meta_and_tensor_keys().1.len(), 2);
        Ok(())
    }

    #[test]
    fn indexed_safe_loader_rejects_missing_and_external_shards() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let index = directory.path().join("model.safetensors.index.json");
        for name in [
            "missing.safetensors",
            "../external.safetensors",
            "/tmp/external.safetensors",
        ] {
            std::fs::write(
                &index,
                serde_json::to_vec(&serde_json::json!({"weight_map":{"x":name}}))?,
            )?;
            assert!(FileTensorLoader::from_path(&index).is_err());
        }
        Ok(())
    }

    #[test]
    fn test_safe_file_tensor_loader() -> anyhow::Result<()> {
        init_test_logging("debug");
        let raw = RawSafeTensors::from_hugging_face_cached(GEMMA3_SAFE_MODEL)?;
        let loader = FileTensorLoader::from_path(raw.model_path())?;
        loader.print_keys();
        println!("config: {:#}", raw.read_config_json()?.0);
        Ok(())
    }
}
