//! Module for code relating to normalisation layers.

use crate::{
    Claim, Element, Shape, Tensor,
    graph::NodeId,
    iop::{prover::Prover, verifier::Verifier},
    layers::{
        LayerOut, ShapeStep,
        provable::{QuantizeOp, QuantizeOutput},
    },
    lookup::{logup_gkr::structs::LogUpBatchProof, table::Table},
    model::Step,
    parser::{
        gguf, json,
        llm::{LLMConfig, transformer::NormType},
        safe,
    },
    quantization::{Quantize, ScalingFactor, ScalingStrategy},
    tensor::{TensorHandle, WrappedTensor},
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::{Arc, Mutex};
use tenstore::StorageKey;
use tracing::trace;

pub mod layernorm;
pub mod qwen_norm;
pub mod rmsnorm;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NormalisationCache {
    max_shift: Element,
    initialized: bool,
}

impl NormalisationCache {
    fn new() -> Self {
        Self {
            max_shift: 0,
            initialized: false,
        }
    }

    fn update(&mut self, shift: Element) {
        if !self.initialized || shift > self.max_shift {
            self.max_shift = shift;
            self.initialized = true;
        }
    }

    fn get_shift(&self) -> Element {
        self.max_shift
    }

    fn reset(&mut self) {
        self.max_shift = 0;
        self.initialized = false;
    }
}
