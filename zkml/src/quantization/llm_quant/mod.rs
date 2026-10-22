//! Implementation of the DuQuant quantisation method as described in https://arxiv.org/pdf/2406.01721

use crate::model::llm::Driver;

use super::*;

use anyhow::Result;

pub mod gemma3;
pub mod gpt2;
pub mod llama3;
pub mod qwen35;
pub mod rotation;

pub trait FPTransformModel {
    fn adapt_model(driver: Driver<f32>) -> Result<Driver<f32>>;
}

#[cfg(test)]
mod tests {

    use anyhow::ensure;
    use tenstore::GenStore;

    use crate::{
        Shape, init_test_logging,
        iop::chunking::LLMChunkingStrategy,
        model::llm::WithMaxContext,
        parser::{
            llm::{
                LLMTokenizer,
                models::{
                    LLMModelLoader,
                    gemma3::{Gemma3, safe_tests::GEMMA3_SAFE_MODEL},
                    gpt2::{GPT2, GPT2_SAFE_MODEL},
                },
                tokenizer::TokenizerLoader,
            },
            safe::RawSafeTensors,
        },
        testing::Pcs,
    };

    type F = ark_bn254::Fr;

    use super::*;

    /// Represents the KL divergence of a single output tensor of a node.
    /// The KL divergence is computed on a channel basis, over sub tensors of the specified shape.
    pub struct KLDivergence {
        kl_divergences: Vec<f32>,
        shape: Shape,
    }

    impl KLDivergence {
        pub fn from_tensors(
            tensor1: &Tensor<f32>,
            tensor2: &Tensor<f32>,
            dim: usize,
        ) -> anyhow::Result<Self> {
            let (p_slices, p_shapes) = tensor1.slice_on_dim(dim);
            let (q_slices, q_shapes) = tensor2.slice_on_dim(dim);
            ensure!(
                p_shapes == q_shapes,
                "Shapes must match, p: {:?}, q: {:?}",
                p_shapes,
                q_shapes
            );
            let kl_divergences = p_slices
                .zip(q_slices)
                .map(|(p_slice, q_slice)| {
                    p_slice
                        .iter()
                        .zip(q_slice.iter())
                        .map(|(p_i, q_i)| p_i * (p_i.max(1e-10) / q_i.max(1e-10)).ln())
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            Ok(Self {
                kl_divergences,
                shape: p_shapes,
            })
        }
    }

    impl std::fmt::Debug for KLDivergence {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "KLDivergence(shape: {})", self.shape)?;
            for (i, kl_divergence) in self.kl_divergences.iter().enumerate() {
                writeln!(f, "\tChannel {i}: {kl_divergence:?}")?;
            }
            Ok(())
        }
    }

    #[test]
    #[ignore = "Used for comparison of float and quantised model outputs, not a test of the proving system itself"]
    fn compare_gpt2() -> anyhow::Result<()> {
        comparison_test_helper::<GPT2>(GPT2_SAFE_MODEL)
    }

    #[test]
    fn prove_gpt2() -> anyhow::Result<()> {
        prove_transformed_model_helper::<GPT2>(GPT2_SAFE_MODEL)
    }

    #[test]
    #[ignore = "Used for comparison of float and quantised model outputs, not a test of the proving system itself"]
    fn compare_gemma3() -> anyhow::Result<()> {
        comparison_test_helper::<Gemma3>(GEMMA3_SAFE_MODEL)
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn prove_gemma3() -> anyhow::Result<()> {
        prove_transformed_model_helper::<Gemma3>(GEMMA3_SAFE_MODEL)
    }

    #[test]
    fn gemma3_inference() -> anyhow::Result<()> {
        transformed_model_inference_helper::<Gemma3>(GEMMA3_SAFE_MODEL)
    }

    #[test]
    #[ignore = "Used for comparison of float and quantised model outputs, not a test of the proving system itself"]
    fn compare_float_gemma3() -> anyhow::Result<()> {
        compare_float_models_helper::<Gemma3>(GEMMA3_SAFE_MODEL)
    }

    fn compare_float_models_helper<
        M: FPTransformModel
            + Default
            + TokenizerLoader<RawSafeTensors>
            + Clone
            + LLMModelLoader<RawSafeTensors>,
    >(
        model_path: &str,
    ) -> anyhow::Result<()> {
        init_test_logging("info");
        // First we load the model
        let raw = RawSafeTensors::from_hugging_face_cached(model_path)?;
        let model_type = M::default();
        let original_driver = Driver::load_from_model(model_type.clone(), &raw, Some(10))?;
        let driver = M::adapt_model(original_driver.clone())?;
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = model_type.load_tokenizer(&raw)?;
        let user_tokens = tokenizer.tokenize(sentence);

        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let mut store = GenStore::default();
        // Inference always runs on unpadded tensors.
        let transformed_trace = driver.run(vec![input_tensor.clone()], &mut store)?;

        let original_trace = original_driver.run(vec![input_tensor.clone()], &mut store)?;

        let logits_id = driver.md.logits;

        let transformed_logits = transformed_trace
            .get_step(&logits_id)
            .unwrap()
            .input_tensors()?[0]
            .clone();
        let original_logits = original_trace
            .get_step(&logits_id)
            .unwrap()
            .input_tensors()?[0]
            .clone();

        for (transformed, original) in transformed_logits
            .data()
            .iter()
            .zip(original_logits.data().iter())
        {
            let diff = (transformed - original).abs();
            ensure!(
                diff < 1e-4,
                "Logits differ by more than 1e-4, got diff {}, transformed {}, original {}",
                diff,
                transformed,
                original
            );
        }
        Ok(())
    }

    fn comparison_test_helper<
        M: FPTransformModel
            + Default
            + TokenizerLoader<RawSafeTensors>
            + Clone
            + LLMModelLoader<RawSafeTensors>,
    >(
        model_path: &str,
    ) -> anyhow::Result<()> {
        const MAX_CONTEXT: usize = 10;
        init_test_logging("info");
        // First we load the model
        let raw = RawSafeTensors::from_hugging_face_cached(model_path)?;
        let model_type = M::default();
        let original_driver = Driver::load_from_model(model_type.clone(), &raw, Some(MAX_CONTEXT))?;
        let driver = M::adapt_model(original_driver.clone())?;
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = model_type.load_tokenizer(&raw)?;
        let user_tokens = tokenizer.tokenize(sentence);

        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let mut store = GenStore::default();
        // Inference always runs on unpadded tensors.
        let trace = driver.run(vec![input_tensor.clone()], &mut store)?;
        driver.model.reset();

        // Rewrite the model by applying our transformation rule
        let (new_driver, model_metadata) =
            original_driver.into_provable_llm_with_transform::<M>(&tokenizer)?;

        // Now we generate the post-transformation trace and extract the logits step data
        let mut store = GenStore::default();
        let input_tensor = new_driver.tokens_to_tensor(&user_tokens)?;
        // Inference always runs on unpadded tensors.
        let new_trace = new_driver.run(vec![input_tensor], &mut store)?;

        // Compare the pre and post transformation data
        // check embeddings output
        if let Some(positional_id) = new_driver.md.positional {
            let float_pos_out =
                trace.get_step(&positional_id).unwrap().output_tensors()?[0].clone();
            let float_pos_in = trace.get_step(&positional_id).unwrap().input_tensors()?[0].clone();
            let (_, edge) = new_driver
                .model
                .graph()
                .neighbors(positional_id, crate::graph::Direction::Outgoing)
                .next()
                .unwrap();
            let post_pos_requant = new_trace.get_step(&edge.target()).unwrap();
            let dequant_pos_padded = post_pos_requant
                .to_dequantize(&model_metadata, edge.target())?
                .output_tensors()?[0]
                .reduce_to_shape(float_pos_out.shape())?;
            let int_pos_in = new_trace
                .get_step(&positional_id)
                .unwrap()
                .to_dequantize(&model_metadata, positional_id)?
                .input_tensors()?[0]
                .clone();
            let dequant_pos_in = int_pos_in.reduce_to_shape(float_pos_in.shape())?;

            println!(
                "First 10 Positional Input Float: {:?}",
                &float_pos_in.data()[0..10]
            );
            println!(
                "First 10 Positional Input Dequant: {:?}",
                &dequant_pos_in.data()[0..10]
            );
            println!(
                "First 10 Positional Output Float: {:?}",
                &float_pos_out.data()[768..778]
            );
            println!(
                "First 10 Positional Output Dequant: {:?}",
                &dequant_pos_padded.data()[768..778]
            );
        }

        let embedding_size = new_driver.md.config.hidden_size;

        for (block_number, transformer_metadata) in new_driver.md.transformers.iter().enumerate() {
            let softmax_id = transformer_metadata
                .transformer
                .as_ref()
                .expect("softmax attention metadata")
                .softmax_id;
            let int_softmax = new_trace.get_step(&softmax_id).unwrap();
            let dequant_softmax_padded = int_softmax
                .to_dequantize(&model_metadata, softmax_id)?
                .output_tensors()?[0]
                .clone();
            let float_softmax = trace.get_step(&softmax_id).unwrap().output_tensors()?[0].clone();
            let dequant_softmax = dequant_softmax_padded.reduce_to_shape(float_softmax.shape())?;
            let slice_dim = float_softmax.shape().rank() - 1;
            let kl_divergence =
                KLDivergence::from_tensors(&dequant_softmax, &float_softmax, slice_dim)?;
            println!(
                "Transformer Block {}: Softmax KL Divergence: {:?}",
                block_number, kl_divergence
            );
            let float_softmax_step = trace.get_step(&softmax_id).unwrap();
            let float_softmax = float_softmax_step.input_tensors()?[0].clone();
            let float_softmax_output = float_softmax_step.output_tensors()?[0].clone();

            let int_softmax_step = int_softmax.to_dequantize(&model_metadata, softmax_id)?;
            let dequant_softmax_padded = int_softmax_step.input_tensors()?[0].clone();
            let dequant_softmax = dequant_softmax_padded.reduce_to_shape(float_softmax.shape())?;

            let dequant_softmax_output_padded = int_softmax_step.output_tensors()?[0].clone();
            let dequant_softmax_output =
                dequant_softmax_output_padded.reduce_to_shape(float_softmax_output.shape())?;

            println!(
                "First 10 Softmax Input Float: {:?}",
                &float_softmax.data()[0..10]
            );
            println!(
                "First 10 Softmax Input Dequant: {:?}",
                &dequant_softmax.data()[0..10]
            );
            println!("==========================");
            println!(
                "First 10 Softmax Output Float: {:?}",
                &float_softmax_output.data()[0..10]
            );
            println!(
                "First 10 Softmax Output Dequant: {:?}",
                &dequant_softmax_output.data()[0..10]
            );
            let qkv_id = transformer_metadata
                .transformer
                .as_ref()
                .expect("softmax attention metadata")
                .qkv_id;
            let int_qkv = new_trace.get_step(&qkv_id).unwrap();
            let dequant_qkv_padded = int_qkv
                .to_dequantize(&model_metadata, qkv_id)?
                .input_tensors()?[0]
                .clone();
            let float_qkv_step = trace.get_step(&qkv_id).unwrap();
            let float_qkv = float_qkv_step.input_tensors()?[0].clone();
            let dequant_qkv = dequant_qkv_padded.reduce_to_shape(float_qkv.shape())?;
            // let dequant_qkv = apply_hadamard::<true, true>(&dequant_qkv, 8)?;
            for (i, (deq, flo)) in dequant_qkv
                .data()
                .chunks(embedding_size)
                .zip(float_qkv.data().chunks(embedding_size))
                .enumerate()
            {
                let deq_row_normed = deq.iter().map(|x| x * x).sum::<f32>().sqrt();
                let flo_row_normed = flo.iter().map(|x| x * x).sum::<f32>().sqrt();

                let row_l4_norm = deq
                    .iter()
                    .zip(flo.iter())
                    .map(|(a, b)| (a - b).powi(4))
                    .sum::<f32>()
                    .powf(1.0 / 4.0);

                let (row_max_abs_diff, pos) = deq
                    .iter()
                    .zip(flo.iter())
                    .map(|(a, b)| (a - b).abs())
                    .enumerate()
                    .fold((0.0f32, 0usize), |(acc, pos_acc), (p, x)| {
                        if x > acc { (x, p) } else { (acc, pos_acc) }
                    });
                let quant_abs_max = deq.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));
                println!(
                    "Transformer Block {}: QKV Input Row {} L4 Norm Difference: {}, Dequant Norm: {}, Float Norm: {}",
                    block_number, i, row_l4_norm, deq_row_normed, flo_row_normed
                );
                println!(
                    " Transformer Block {}: QKV Input Row {} Max Abs Diff: {}, dequant value at max pos: {}, quant abs max {} float value at max pos: {}",
                    block_number, i, row_max_abs_diff, deq[pos], quant_abs_max, flo[pos]
                );
            }
            println!("QKV input original: {:?}", &float_qkv.data()[0..10]);
            println!("QKV input dequant: {:?}", &dequant_qkv.data()[0..10]);

            let qkt_id = transformer_metadata
                .transformer
                .as_ref()
                .expect("softmax attention metadata")
                .qkt_id;
            let float_qkt_step = trace.get_step(&qkt_id).unwrap();
            let float_q = float_qkt_step.input_tensors()?[0].clone();
            let float_k = float_qkt_step.input_tensors()?[1].clone();
            let int_qkt = new_trace.get_step(&qkt_id).unwrap();
            let dequant_q_padded = int_qkt
                .to_dequantize(&model_metadata, qkt_id)?
                .input_tensors()?[0]
                .clone();
            let dequant_k_padded = int_qkt
                .to_dequantize(&model_metadata, qkt_id)?
                .input_tensors()?[1]
                .clone();
            let dequant_q = dequant_q_padded.reduce_to_shape(float_q.shape())?;
            let dequant_k = dequant_k_padded.reduce_to_shape(float_k.shape())?;

            println!("--- QKT Inputs ---");
            let q_l4_norm = dequant_q
                .data()
                .iter()
                .zip(float_q.data().iter())
                .map(|(a, b)| (a - b).powi(4))
                .sum::<f32>()
                .powf(1.0 / 4.0);
            let q_max_abs_diff = dequant_q
                .data()
                .iter()
                .zip(float_q.data().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |acc, x| acc.max(x));
            println!(
                "Transformer Block {}: Q Input L4 Norm Difference: {}, Max Abs Diff: {}",
                block_number, q_l4_norm, q_max_abs_diff
            );
            let k_l4_norm = dequant_k
                .data()
                .iter()
                .zip(float_k.data().iter())
                .map(|(a, b)| (a - b).powi(4))
                .sum::<f32>()
                .powf(1.0 / 4.0);
            let (k_max_abs_diff, pos) = dequant_k
                .data()
                .iter()
                .zip(float_k.data().iter())
                .map(|(a, b)| (a - b).abs())
                .enumerate()
                .fold((0.0f32, 0usize), |(acc, pos_acc), (p, x)| {
                    if x > acc { (x, p) } else { (acc, pos_acc) }
                });
            println!(
                "Transformer Block {}: K Input L4 Norm Difference: {}, max Abs Diff: {}, dequant value at max pos: {}, float value at max pos: {}",
                block_number,
                k_l4_norm,
                k_max_abs_diff,
                dequant_k.data()[pos],
                float_k.data()[pos]
            );
            println!("QKT input original Q: {:?}", &float_q.data()[0..10]);
            println!("QKT input dequant Q: {:?}", &dequant_q.data()[0..10]);
            println!("QKT input original K: {:?}", &float_k.data()[0..10]);
            println!("QKT input dequant K: {:?}", &dequant_k.data()[0..10]);
            println!("--- End QKT Inputs ---");
            let attention_values = transformer_metadata
                .transformer
                .as_ref()
                .expect("softmax attention metadata")
                .residual_id;
            let float_attn_values_step = trace.get_step(&attention_values).unwrap();
            let float_attn_values = float_attn_values_step.input_tensors()?[1].clone();
            let int_attn_values = new_trace.get_step(&attention_values).unwrap();
            let dequant_attn_values_padded = int_attn_values
                .to_dequantize(&model_metadata, attention_values)?
                .input_tensors()?[1]
                .clone();
            let dequant_attn_values =
                dequant_attn_values_padded.reduce_to_shape(float_attn_values.shape())?;
            let attn_l4_norm = dequant_attn_values
                .data()
                .iter()
                .zip(float_attn_values.data().iter())
                .map(|(a, b)| (a - b).powi(4))
                .sum::<f32>()
                .powf(1.0 / 4.0);
            let (attn_max_abs_diff, pos) = dequant_attn_values
                .data()
                .iter()
                .zip(float_attn_values.data().iter())
                .map(|(a, b)| (a - b).abs())
                .enumerate()
                .fold((0.0f32, 0usize), |(acc, pos_acc), (p, x)| {
                    if x > acc { (x, p) } else { (acc, pos_acc) }
                });
            println!(
                "Transformer Block {}: Attention Values Input L4 Norm Difference: {}, max Abs Diff: {}, dequant value at max pos: {}, float value at max pos: {}",
                block_number,
                attn_l4_norm,
                attn_max_abs_diff,
                dequant_attn_values.data()[pos],
                float_attn_values.data()[pos]
            );
            println!("-----------FFN UP--------");

            let up_id = transformer_metadata.ffn.up_id;
            let int_up = new_trace.get_step(&up_id).unwrap();
            let dequant_up_padded = int_up
                .to_dequantize(&model_metadata, up_id)?
                .input_tensors()?[0]
                .clone();
            let float_up = trace.get_step(&up_id).unwrap().input_tensors()?[0].clone();
            let dequant_up = dequant_up_padded.reduce_to_shape(float_up.shape())?;

            for (i, (deq, flo)) in dequant_up
                .data()
                .chunks(embedding_size)
                .zip(float_up.data().chunks(embedding_size))
                .enumerate()
            {
                let deq_row_normed = deq.iter().map(|x| x * x).sum::<f32>().sqrt();
                let flo_row_normed = flo.iter().map(|x| x * x).sum::<f32>().sqrt();

                let row_l4_norm = deq
                    .iter()
                    .zip(flo.iter())
                    .map(|(a, b)| (a - b).abs().powi(4))
                    .sum::<f32>()
                    .powf(1.0 / 4.0);
                println!(
                    "Transformer Block {}: FFN UP Input Row {} L4 Norm Difference: {}, Dequant Norm: {}, Float Norm: {}",
                    block_number, i, row_l4_norm, deq_row_normed, flo_row_normed
                );
            }
            println!("FFN UP input original: {:?}", &float_up.data()[0..10]);
            println!("FFN UP input dequant: {:?}", &dequant_up.data()[0..10]);
            println!("---------FFN DOWN--------");
            let down_id = transformer_metadata.ffn.down_id;
            let int_down = new_trace.get_step(&down_id).unwrap();
            let dequant_down_padded = int_down
                .to_dequantize(&model_metadata, down_id)?
                .input_tensors()?[0]
                .clone();
            let float_down = trace.get_step(&down_id).unwrap().input_tensors()?[0].clone();
            let dequant_down = dequant_down_padded.reduce_to_shape(float_down.shape())?;
            let float_down_max = float_down.data().iter().fold(f32::MIN, |a, &b| a.max(b));
            let float_down_min = float_down.data().iter().fold(f32::MAX, |a, &b| a.min(b));
            let dequant_down_max = dequant_down.data().iter().fold(f32::MIN, |a, &b| a.max(b));
            let dequant_down_min = dequant_down.data().iter().fold(f32::MAX, |a, &b| a.min(b));
            println!(
                "Transformer Block {}: FFN DOWN Float Max: {}, Float Min: {}, Dequant Max: {}, Dequant Min: {}",
                block_number, float_down_max, float_down_min, dequant_down_max, dequant_down_min
            );
            println!("----------END OF BLOCK-----------");
        }

        let final_proj_id = new_driver.md.final_proj;
        let float_final_proj = trace.get_step(&final_proj_id).unwrap();
        let float_fp_input = float_final_proj.input_tensors()?[0].clone();
        let int_final_proj = new_trace.get_step(&final_proj_id).unwrap();
        let dequant_fp_padded = int_final_proj
            .to_dequantize(&model_metadata, final_proj_id)?
            .input_tensors()?[0]
            .clone();
        let dequant_fp = dequant_fp_padded.reduce_to_shape(float_fp_input.shape())?;
        for (i, (deq, flo)) in dequant_fp
            .data()
            .chunks(768)
            .zip(float_fp_input.data().chunks(768))
            .enumerate()
        {
            let deq_row_normed = deq.iter().map(|x| x * x).sum::<f32>().sqrt();
            let flo_row_normed = flo.iter().map(|x| x * x).sum::<f32>().sqrt();

            let row_l4_norm = deq
                .iter()
                .zip(flo.iter())
                .map(|(a, b)| (a - b).abs().powi(4))
                .sum::<f32>()
                .powf(1.0 / 4.0);
            println!(
                "Final Projection Input Row {} L4 Norm Difference: {}, Dequant Norm: {}, Float Norm: {}",
                i, row_l4_norm, deq_row_normed, flo_row_normed
            );
            println!(
                "Final Projection input original: {:?}",
                &float_fp_input.data()[0..10]
            );
            println!(
                "Final Projection input dequant: {:?}",
                &dequant_fp.data()[0..10]
            );
        }
        Ok(())
    }

    fn transformed_model_inference_helper<
        M: FPTransformModel
            + Default
            + TokenizerLoader<RawSafeTensors>
            + Clone
            + LLMModelLoader<RawSafeTensors>,
    >(
        model_path: &str,
    ) -> anyhow::Result<()> {
        init_test_logging("debug");
        const MAX_CONTEXT: usize = 64;
        // First we load the model
        let raw_safe = RawSafeTensors::from_hugging_face_cached(model_path)?;
        let model_type = M::default();

        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let tokenizer = model_type.load_tokenizer(&raw_safe)?;

        // Generate or load the prover & verifier contexts
        let (driver, _) =
            Driver::load_from_model(model_type.clone(), &raw_safe, Some(MAX_CONTEXT))?
                .into_provable_llm_with_transform::<M>(&tokenizer)?;

        let user_tokens = driver.random_sequence(MAX_CONTEXT - 2);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let mut store = GenStore::default();

        let _ = driver.run_elements(input_tensor, &mut store)?;
        Ok(())
    }

    fn prove_transformed_model_helper<
        M: FPTransformModel
            + Default
            + TokenizerLoader<RawSafeTensors>
            + Clone
            + LLMModelLoader<RawSafeTensors>,
    >(
        model_path: &str,
    ) -> anyhow::Result<()> {
        init_test_logging("debug");
        const MAX_CONTEXT: usize = 64;
        // First we load the model
        let raw = RawSafeTensors::from_hugging_face_cached(model_path)?;
        let model_type = M::default();

        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let tokenizer = model_type.load_tokenizer(&raw)?;

        // Generate or load the prover & verifier contexts
        let (driver, _) = Driver::load_from_model(model_type.clone(), &raw, Some(MAX_CONTEXT))?
            .into_provable_llm_with_transform::<M>(&tokenizer)?;

        driver.model.describe();
        let (prover_ctx, verifier_ctx) = driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

        let user_tokens = driver.random_sequence(MAX_CONTEXT - 2);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let mut store = GenStore::default();

        let trace = driver.run_elements(input_tensor, &mut store)?;
        let chunking_strategy = LLMChunkingStrategy::from((&trace, &prover_ctx.model_ctx));
        let (proof, io) = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            Some(1),
            chunking_strategy,
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        tracing::info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }
}
