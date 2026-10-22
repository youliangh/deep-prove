pub mod config;
use crate::parser::llm::metadata::{LLMMetadata, TransformerMetadata};

pub mod metadata;
pub mod models;
pub mod tokenizer;
pub mod transformer;
pub use crate::parser::{
    gguf::{self, FileTensorLoader as GGUFLoader},
    json::{self, FileTensorLoader as JSONLoader},
    safe::{ConfigJSON, FileTensorLoader as SafeLoader},
};
use anyhow::Context;
pub use config::LLMConfig;
use serde::{Deserialize, Serialize};
use tenstore::StorageKey;
pub use tokenizer::{HFTokenizer, LLMTokenizer};

use crate::{
    Shape,
    layers::{
        Layer,
        einsum::EinSum,
        transformer::{
            embeddings::Embeddings, logits::Logits, normalisation::layernorm::LayerNorm,
            positional::Positional,
        },
    },
    model::Model,
    number::Number,
    parser::{
        LayerInsertion, Load,
        llm::{config::LLMStructure, transformer::Norm},
    },
    tensor::{KeyedTensor, TensorTypeParam},
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    derive_more::From,
    derive_more::Into,
    Serialize,
    Deserialize,
)]
pub struct Token(pub(crate) usize);

// i64 is the type used by token_to_i
impl From<i64> for Token {
    fn from(t: i64) -> Self {
        Self(t as usize)
    }
}

impl From<Token> for i64 {
    fn from(t: Token) -> Self {
        t.0 as i64
    }
}

impl From<&Token> for i64 {
    fn from(t: &Token) -> Self {
        t.0 as i64
    }
}

impl From<u32> for Token {
    fn from(t: u32) -> Self {
        Self(t as usize)
    }
}

impl From<&Token> for u32 {
    fn from(t: &Token) -> Self {
        t.0 as u32
    }
}

impl From<u64> for Token {
    fn from(t: u64) -> Self {
        Self(t as usize)
    }
}

impl Token {
    pub fn as_number<N: Number>(&self) -> N {
        N::from_usize(self.0)
    }

    pub fn as_tensor_type_param<T: TensorTypeParam>(&self) -> T {
        T::from_usize(self.0)
    }
}

/// Intermediate representation of a LLM model.
/// This is used to store the model weight in generic way before transforming it into the graph representation.
/// It is a common representation that can be used to build the final model.
#[derive(Debug, Clone)]
pub struct LLMIR<T> {
    pub embeddings: Embeddings<f32>,
    pub global_positional: Option<Positional<f32>>,
    pub blocks: Vec<T>,
    /// Final LayerNorm applied after all transformer blocks (ln_f in GPT-2)
    pub final_norm: Norm<f32>,
    /// final projection on token sizes to before selecting next token
    pub final_proj: EinSum<f32>,
}

impl<T> LLMIR<T> {
    pub fn new(
        embeddings: Embeddings<f32>,
        global_positional: Option<Positional<f32>>,
        blocks: Vec<T>,
        final_norm: Norm<f32>,
        final_proj: EinSum<f32>,
    ) -> Self {
        Self {
            embeddings,
            global_positional,
            blocks,
            final_norm,
            final_proj,
        }
    }
}

/// The generic T holds the logic to create attention layers.
impl<T: LayerInsertion<Metadata = TransformerMetadata>> LLMIR<T> {
    /// Creates a Model<f32> from the [`LLMModel`].
    pub fn into_model(
        self,
        llm_config: LLMConfig,
        user_input_shape: Shape,
    ) -> anyhow::Result<(Model<f32>, LLMMetadata)> {
        let mut model = Model::new_from_input_shapes(vec![user_input_shape]);

        let embeddings_id =
            model.add_consecutive_layer(Layer::Embeddings(self.embeddings), None)?;
        let mut last_node_id = Some(embeddings_id);
        let positional_id = if let Some(positional) = self.global_positional {
            last_node_id =
                Some(model.add_consecutive_layer(Layer::Positional(positional), last_node_id)?);
            last_node_id
        } else {
            None
        };

        let mut transformers = Vec::with_capacity(self.blocks.len());
        for block in self.blocks {
            let (id, metadata) = block.add_to_model(&mut model, last_node_id)?;
            transformers.push(metadata);
            last_node_id = Some(id);
        }
        last_node_id = Some(model.add_consecutive_layer(self.final_norm.to_layer(), last_node_id)?);

        let final_proj_id =
            model.add_consecutive_layer(
                Layer::EinSum(
                    self.final_proj.disable_requantisation() // we can skip requantisation here since we can handle bigger values in Argmax
                ), last_node_id)?;

        let logits_id = model
            .add_consecutive_layer(Layer::Logits(Logits::new_argmax()), Some(final_proj_id))?;
        model.automatic_output_labelling()?;
        let metadata = LLMMetadata {
            config: llm_config,
            additional_eos_tokens: Vec::new(),
            transformers,
            embeddings: embeddings_id,
            positional: positional_id,
            final_proj: final_proj_id,
            logits: final_proj_id,
            argmax: logits_id,
        };
        Ok((model, metadata))
    }
}

impl<T> Load<GGUFLoader> for LLMIR<T>
where
    // Note we don't care about the metadata here as we're only building the IR.
    // The final graph model is when we care about the metadata.
    T: Load<GGUFLoader, Config = LLMStructure> + LayerInsertion,
{
    type Config = LLMStructure;

    fn from_loader(loader: &GGUFLoader, config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embeddings = Embeddings::from_loader(loader)?;
        let global_positional = config
            .global_positional
            .as_ref()
            .map(|p| Positional::from_gguf(loader, config, p))
            .transpose()?;

        let num_layers = config.generic.num_block;
        let blocks = (0..num_layers)
            .map(|i| T::from_loader(&loader.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<T>>>()?;

        let final_norm = config
            .norm_type
            .from_gguf(&loader.pp("output_"), &config.generic)?;

        // Now we work out the final projection
        // there may or may not be a bias
        let input_terms = "X(se)@WE(ve)";
        let proj_bias = loader.get_tensor("output.bias").ok();
        let output_terms = if proj_bias.is_some() {
            "O(sv)+BIAS(v)"
        } else {
            "O(sv)"
        };
        let equation = format!("{input_terms}->{output_terms}");
        let mut proj_weights = KeyedTensor::try_from(&embeddings.mat)?;
        // We need to modify the key to avoid conflicts between the embeddings matrix and the final projection
        // the embeddings matrix is scaled _after_ parsing the model, but the final projection is *not* scaled
        // so in effect, these are two different tensors and thus need to be represented by two different keys
        proj_weights.key = StorageKey::from(format!("{}_final_proj", proj_weights.storage_key()));
        let final_proj = EinSum::<f32>::new(
            equation,
            vec![Some(proj_weights.into())],
            vec![proj_bias.map(|tensor| tensor.into())],
        )?;

        Ok(Self::new(
            embeddings,
            global_positional,
            blocks,
            final_norm,
            final_proj,
        ))
    }
}

pub const FINAL_PROJ_KEYS: &[&str] = &[
    "lm_head.weight",
    "output.weight",
    "output_projection.weight",
    "decoder.lm_head.weight",
    "model.lm_head.weight",
];

impl<T> Load<SafeLoader> for LLMIR<T>
where
    T: Load<SafeLoader, Config = (LLMStructure, ConfigJSON)> + LayerInsertion,
{
    type Config = (LLMStructure, ConfigJSON);

    fn from_loader(loader: &SafeLoader, loader_config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let (structure, config) = loader_config;
        let embeddings = Embeddings::from_safetensors_loader(loader)?;
        let global_positional = structure
            .global_positional
            .as_ref()
            .map(|p| Positional::from_safetensors_loader(loader, structure, p))
            .transpose()?;

        let num_layers = structure.generic.num_block;
        let blocks = (0..num_layers)
            .map(|i| T::from_loader(&loader.pp(&format!("model.layers.{i}.")), loader_config))
            .collect::<anyhow::Result<Vec<T>>>()?;

        let final_norm = structure.norm_type.from_safetensors(
            &loader.pp("model."),
            config,
            &structure.generic,
        )?;
        // Now we work out the final projection
        // Check for separate lm_head (tie_word_embeddings=false) or use embeddings (tied)
        let proj_weights = match config.get::<bool, _>("tie_word_embeddings") {
            Some(true) => embeddings.mat.clone(),
            Some(false) => {
                // lm_head.weight is already [vocab_size, hidden_size], no transpose needed
                let maybe_tensor = FINAL_PROJ_KEYS
                    .iter()
                    .find_map(|k| loader.get_tensor(k).ok());
                maybe_tensor
                    .context("unable to find lm_head weight")?
                    .into()
            }
            None => embeddings.mat.clone(),
        };

        // there may or may not be a bias
        let input_terms = "X(se)@WE(ve)";
        let proj_bias = loader.get_tensor("lm_head.bias").ok();
        let output_terms = if proj_bias.is_some() {
            "O(sv)+BIAS(v)"
        } else {
            "O(sv)"
        };
        let equation = format!("{input_terms}->{output_terms}");
        let mut proj_weights = KeyedTensor::try_from(&proj_weights)?;
        // We need to modify the key to avoid conflicts between the embeddings matrix and the final projection
        // the embeddings matrix is scaled _after_ parsing the model, but the final projection is *not* scaled
        // so in effect, these are two different tensors and thus need to be represented by two different keys
        proj_weights.key = StorageKey::from(format!("{}_final_proj", proj_weights.key));
        let final_proj = EinSum::<f32>::new(
            equation,
            vec![Some(proj_weights.into())],
            vec![proj_bias.map(|tensor| tensor.into())],
        )?;

        Ok(Self::new(
            embeddings,
            global_positional,
            blocks,
            final_norm,
            final_proj,
        ))
    }
}

impl<T> Load<JSONLoader> for LLMIR<T>
where
    T: Load<JSONLoader, Config = LLMConfig> + LayerInsertion,
{
    type Config = LLMConfig;

    fn from_loader(l: &JSONLoader, config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embeddings = Embeddings::from_json(l)?;
        let positional = Some(Positional::from_json(l, config)?);
        let num_layers = config.num_block;
        let blocks = (0..num_layers)
            .map(|i| T::from_loader(&l.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<T>>>()?;
        let final_norm = Norm::LayerNorm(LayerNorm::from_json(&l.pp("output_"), config)?);

        // Now we work out the final projection
        // there may or may not be a bias
        let input_terms = "X(se)@WE(ve)";
        let proj_bias = l.get_tensor("output.bias").ok();
        let output_terms = if proj_bias.is_some() {
            "O(sv)+BIAS(v)"
        } else {
            "O(sv)"
        };
        let equation = format!("{input_terms}->{output_terms}");
        let proj_weights = KeyedTensor::try_from(&embeddings.mat)?;
        let final_proj = EinSum::<f32>::new(
            equation,
            vec![Some(proj_weights.into())],
            vec![proj_bias.map(|tensor| tensor.into())],
        )?;
        Ok(Self::new(
            embeddings, positional, blocks, final_norm, final_proj,
        ))
    }
}
