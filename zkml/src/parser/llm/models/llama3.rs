//! Llama 3/3.1/3.2 causal language models in Hugging Face SafeTensors format.
//!
//! The decoder graph is shared with Llama 2. The shared loader handles Llama 3's
//! frequency-dependent RoPE scaling and both tied and independent LM heads.

use crate::{
    model::Model,
    parser::{
        ModelLoader,
        llm::{HFTokenizer, LLMMetadata, models::LLMModelLoader, tokenizer::TokenizerLoader},
        safe::RawSafeTensors,
    },
};

pub const LLAMA3_NAME: &str = "llama3";

#[derive(Clone, Copy, Debug, Default)]
pub struct Llama3 {
    max_ctx_length: Option<usize>,
}

impl Llama3 {
    pub fn new() -> Self {
        Self::default()
    }

    /// Limit RoPE precomputation to the intended inference/proving context.
    pub fn with_max_ctx_length(mut self, length: usize) -> Self {
        self.max_ctx_length = Some(length);
        self
    }
}

impl TokenizerLoader<RawSafeTensors> for Llama3 {
    fn load_tokenizer(&self, raw: &RawSafeTensors) -> anyhow::Result<HFTokenizer> {
        // tokenizer.json includes the Llama 3 BPE pipeline and BOS postprocessor.
        HFTokenizer::from_tokenizer_json_path(raw.tokenizer_path())
    }
}

impl<DataFormat> LLMModelLoader<DataFormat> for Llama3
where
    Llama3: ModelLoader<DataFormat, Metadata = LLMMetadata>,
{
    fn with_max_context_length(self, length: usize) -> Self {
        self.with_max_ctx_length(length)
    }
}

impl ModelLoader<RawSafeTensors> for Llama3 {
    type Metadata = LLMMetadata;

    fn model_name(&self) -> String {
        LLAMA3_NAME.to_owned()
    }

    fn parse(&self, raw: &RawSafeTensors) -> anyhow::Result<(Model<f32>, Self::Metadata)> {
        super::llama::parse(raw, self.max_ctx_length, LLAMA3_NAME)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::{Context, Result};
    use tenstore::GenStore;

    use crate::{
        Tensor,
        model::llm::Driver,
        parser::llm::{LLMTokenizer, Token},
    };

    use super::*;

    /// Run explicitly with LLAMA3_MODEL_DIR pointing at the local 1B checkpoint.
    #[test]
    #[ignore = "requires local Llama-3.2-1B weights and several GB of memory"]
    fn llama3_local_1b_float_inference() -> Result<()> {
        let directory = PathBuf::from(
            std::env::var("LLAMA3_MODEL_DIR")
                .context("Set LLAMA3_MODEL_DIR to a local Llama-3.2-1B directory")?,
        );
        let raw = RawSafeTensors::new(
            directory.join("model.safetensors"),
            directory.join("tokenizer.json"),
            directory.join("config.json"),
        );
        let tokenizer = Llama3::new().load_tokenizer(&raw)?;
        let tokens = tokenizer.tokenize("The capital of France is");
        assert_eq!(tokens.first().copied(), Some(Token::from(128000usize)));
        let mut driver = Driver::load_from_model(Llama3::new(), &raw, Some(64))?;
        assert_eq!(driver.md.config.model_name, "llama3");
        assert_eq!(driver.md.config.num_block, 16);
        assert_eq!(driver.md.config.num_heads, 32);
        assert_eq!(driver.md.config.head_size, 64);
        assert_eq!(driver.md.config.vocab_size, 128256);
        driver.with_max_context(tokens.len());
        let input = Tensor::new(
            vec![tokens.len()].into(),
            tokens
                .iter()
                .map(|token| token.as_tensor_type_param::<f32>())
                .collect(),
        )?;
        let trace = driver.run(vec![input], &mut GenStore::default())?;
        let logits = trace
            .get_step(&driver.md.final_proj)
            .context("Missing final projection")?
            .output_tensors()?[0]
            .clone();
        assert_eq!(
            logits.shape(),
            &crate::Shape::from(vec![tokens.len(), 128256])
        );
        assert!(logits.data().iter().all(|value| value.is_finite()));
        assert!(logits.data().iter().any(|value| value.abs() > 1.0));
        Ok(())
    }
}
