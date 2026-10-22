use crate::parser::{ModelLoader, llm::LLMMetadata};

pub mod gemma3;
pub mod gpt2;
mod llama;
pub mod llama2;
pub mod llama3;
pub mod qwen35;

pub trait LLMModelLoader<DataFormat>: ModelLoader<DataFormat, Metadata = LLMMetadata> {
    fn with_max_context_length(self, _max_ctx_length: usize) -> Self;
}
