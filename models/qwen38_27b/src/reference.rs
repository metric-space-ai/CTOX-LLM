//! Scalar Qwen3.8 operator oracle.
//!
//! Production dispatch must never use this module. Its equations track the
//! pinned Transformers Qwen3.5 implementation used to create BF16 evidence:
//! <https://github.com/huggingface/transformers/blob/a353632607c59463e6ced86a44c2de3c2cd62d5e/src/transformers/models/qwen3_5/modeling_qwen3_5.py>.

use crate::{EngineError, Result};

const L2_EPSILON: f32 = 1e-6;

#[inline]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn matrix_contract(operation: &str, values: &[f32], rows: usize, columns: usize) -> Result<()> {
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| EngineError::Shape(format!("{operation} shape overflows")))?;
    if rows == 0 || columns == 0 || values.len() != expected {
        return Err(EngineError::Shape(format!(
            "{operation} has {} values, expected {rows}x{columns}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::InvalidArtifact(format!(
            "{operation} input is non-finite"
        )));
    }
    Ok(())
}

/// Qwen3.5 uses `(normalized * (1 + weight))`, not Llama's weight convention.
// ref: modeling_qwen3_5.py:721-735
pub fn rms_norm_1p_weight(
    values: &[f32],
    rows: usize,
    columns: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    matrix_contract("rms_norm", values, rows, columns)?;
    if weight.len() != columns || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(EngineError::Shape(
            "rms_norm weight or epsilon differs".into(),
        ));
    }
    let mut output = Vec::with_capacity(values.len());
    for row in values.chunks_exact(columns) {
        let variance = row.iter().map(|value| value * value).sum::<f32>() / columns as f32;
        let inverse = (variance + epsilon).sqrt().recip();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(value, weight)| value * inverse * (1.0 + weight)),
        );
    }
    Ok(output)
}

/// GatedDeltaNet normalizes first, applies its direct weight, then SiLU(gate).
// ref: modeling_qwen3_5.py:168-184
pub fn rms_norm_gated(
    values: &[f32],
    gate: &[f32],
    rows: usize,
    columns: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    matrix_contract("rms_norm_gated", values, rows, columns)?;
    matrix_contract("rms_norm_gated gate", gate, rows, columns)?;
    if weight.len() != columns || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(EngineError::Shape(
            "rms_norm_gated weight or epsilon differs".into(),
        ));
    }
    let mut output = Vec::with_capacity(values.len());
    for (row, gate_row) in values.chunks_exact(columns).zip(gate.chunks_exact(columns)) {
        let variance = row.iter().map(|value| value * value).sum::<f32>() / columns as f32;
        let inverse = (variance + epsilon).sqrt().recip();
        output.extend(
            row.iter()
                .zip(gate_row)
                .zip(weight)
                .map(|((value, gate), weight)| value * inverse * weight * silu(*gate)),
        );
    }
    Ok(output)
}

// ref: modeling_qwen3_5.py:705-718
pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len()
        || gate.is_empty()
        || gate.iter().chain(up).any(|value| !value.is_finite())
    {
        return Err(EngineError::Shape(
            "SwiGLU inputs are empty, non-finite, or differently shaped".into(),
        ));
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect())
}

/// Apply Qwen's non-interleaved partial RoPE to flattened heads in place.
// ref: modeling_qwen3_5.py:547-590
#[allow(clippy::too_many_arguments)]
pub fn apply_partial_rope(
    query: &mut [f32],
    key: &mut [f32],
    query_heads: usize,
    key_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: u64,
    theta: f32,
) -> Result<()> {
    matrix_contract("RoPE query", query, query_heads, head_dim)?;
    matrix_contract("RoPE key", key, key_heads, head_dim)?;
    if rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(EngineError::Shape("invalid partial RoPE contract".into()));
    }
    let half = rotary_dim / 2;
    let mut cosine = Vec::with_capacity(half);
    let mut sine = Vec::with_capacity(half);
    for index in 0..half {
        let inverse_frequency = theta.powf(-((2 * index) as f32) / rotary_dim as f32);
        let angle = position as f32 * inverse_frequency;
        cosine.push(angle.cos());
        sine.push(angle.sin());
    }
    for values in [query, key] {
        for head in values.chunks_exact_mut(head_dim) {
            for index in 0..half {
                let left = head[index];
                let right = head[index + half];
                head[index] = left * cosine[index] - right * sine[index];
                head[index + half] = right * cosine[index] + left * sine[index];
            }
        }
    }
    Ok(())
}

/// Update one causal depthwise-convolution state and return SiLU outputs.
// ref: modeling_qwen3_5.py:199-216
pub fn causal_conv_silu_update(
    input: &[f32],
    state: &mut [f32],
    weight: &[f32],
    channels: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    matrix_contract("causal convolution state", state, channels, kernel)?;
    matrix_contract("causal convolution weight", weight, channels, kernel)?;
    if input.len() != channels || input.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Shape(
            "causal convolution input differs".into(),
        ));
    }
    let mut output = Vec::with_capacity(channels);
    for (channel, input_value) in input.iter().enumerate().take(channels) {
        let start = channel * kernel;
        let local_state = &mut state[start..start + kernel];
        local_state.copy_within(1.., 0);
        local_state[kernel - 1] = *input_value;
        let sum = local_state
            .iter()
            .zip(&weight[start..start + kernel])
            .map(|(value, weight)| value * weight)
            .sum::<f32>();
        output.push(silu(sum));
    }
    Ok(output)
}

/// Exact single-token recurrent GatedDeltaNet rule after Q/K head repetition.
// ref: modeling_qwen3_5.py:330-380
#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_step(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    state: &mut [f32],
    heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<Vec<f32>> {
    matrix_contract("delta query", query, heads, key_dim)?;
    matrix_contract("delta key", key, heads, key_dim)?;
    matrix_contract("delta value", value, heads, value_dim)?;
    let expected_state = heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| EngineError::Shape("delta state shape overflows".into()))?;
    if state.len() != expected_state
        || log_decay.len() != heads
        || beta.len() != heads
        || state
            .iter()
            .chain(log_decay)
            .chain(beta)
            .any(|value| !value.is_finite())
    {
        return Err(EngineError::Shape("delta state contract differs".into()));
    }
    let mut output = vec![0.0; heads * value_dim];
    let query_scale = (key_dim as f32).sqrt().recip();
    for head in 0..heads {
        let q = &query[head * key_dim..(head + 1) * key_dim];
        let k = &key[head * key_dim..(head + 1) * key_dim];
        let v = &value[head * value_dim..(head + 1) * value_dim];
        let q_inverse = (q.iter().map(|x| x * x).sum::<f32>() + L2_EPSILON)
            .sqrt()
            .recip();
        let k_inverse = (k.iter().map(|x| x * x).sum::<f32>() + L2_EPSILON)
            .sqrt()
            .recip();
        let state_start = head * key_dim * value_dim;
        let local_state = &mut state[state_start..state_start + key_dim * value_dim];
        let decay = log_decay[head].exp();
        local_state.iter_mut().for_each(|item| *item *= decay);
        for value_index in 0..value_dim {
            let memory = (0..key_dim)
                .map(|index| local_state[index * value_dim + value_index] * k[index] * k_inverse)
                .sum::<f32>();
            let delta = (v[value_index] - memory) * beta[head];
            for key_index in 0..key_dim {
                local_state[key_index * value_dim + value_index] +=
                    k[key_index] * k_inverse * delta;
            }
        }
        for value_index in 0..value_dim {
            output[head * value_dim + value_index] = (0..key_dim)
                .map(|index| {
                    local_state[index * value_dim + value_index]
                        * q[index]
                        * q_inverse
                        * query_scale
                })
                .sum();
        }
    }
    Ok(output)
}

/// Sequential prefill oracle for the same recurrence used by chunked kernels.
#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_sequence(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    state: &mut [f32],
    tokens: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<Vec<f32>> {
    let qk_per_token = heads
        .checked_mul(key_dim)
        .ok_or_else(|| EngineError::Shape("delta sequence QK shape overflows".into()))?;
    let value_per_token = heads
        .checked_mul(value_dim)
        .ok_or_else(|| EngineError::Shape("delta sequence value shape overflows".into()))?;
    matrix_contract("delta sequence query", query, tokens, qk_per_token)?;
    matrix_contract("delta sequence key", key, tokens, qk_per_token)?;
    matrix_contract("delta sequence value", value, tokens, value_per_token)?;
    matrix_contract("delta sequence decay", log_decay, tokens, heads)?;
    matrix_contract("delta sequence beta", beta, tokens, heads)?;
    let mut output = Vec::with_capacity(tokens * value_per_token);
    for token in 0..tokens {
        output.extend(recurrent_gated_delta_step(
            &query[token * qk_per_token..(token + 1) * qk_per_token],
            &key[token * qk_per_token..(token + 1) * qk_per_token],
            &value[token * value_per_token..(token + 1) * value_per_token],
            &log_decay[token * heads..(token + 1) * heads],
            &beta[token * heads..(token + 1) * heads],
            state,
            heads,
            key_dim,
            value_dim,
        )?);
    }
    Ok(output)
}

/// Causal grouped-query attention, returning token-major `[tokens, heads, dim]`.
// ref: modeling_qwen3_5.py:593-627
#[allow(clippy::too_many_arguments)]
pub fn grouped_query_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_heads: usize,
    key_value_heads: usize,
    query_tokens: usize,
    key_value_tokens: usize,
    head_dim: usize,
    query_start_position: usize,
) -> Result<Vec<f32>> {
    matrix_contract(
        "attention query",
        query,
        query_heads,
        query_tokens
            .checked_mul(head_dim)
            .ok_or_else(|| EngineError::Shape("attention query shape overflows".into()))?,
    )?;
    let kv_columns = key_value_tokens
        .checked_mul(head_dim)
        .ok_or_else(|| EngineError::Shape("attention KV shape overflows".into()))?;
    matrix_contract("attention key", key, key_value_heads, kv_columns)?;
    matrix_contract("attention value", value, key_value_heads, kv_columns)?;
    if key_value_heads == 0
        || !query_heads.is_multiple_of(key_value_heads)
        || query_start_position
            .checked_add(query_tokens)
            .is_none_or(|end| end > key_value_tokens)
    {
        return Err(EngineError::Shape(
            "grouped-query attention topology or causal range differs".into(),
        ));
    }
    let groups = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; query_tokens * query_heads * head_dim];
    let mut scores = Vec::with_capacity(key_value_tokens);
    for token in 0..query_tokens {
        let available = query_start_position + token + 1;
        for query_head in 0..query_heads {
            let kv_head = query_head / groups;
            let query_start = (query_head * query_tokens + token) * head_dim;
            let query_row = &query[query_start..query_start + head_dim];
            scores.clear();
            for kv_token in 0..available {
                let key_start = (kv_head * key_value_tokens + kv_token) * head_dim;
                let score = query_row
                    .iter()
                    .zip(&key[key_start..key_start + head_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(EngineError::InvalidArtifact(
                    "attention softmax normalization is invalid".into(),
                ));
            }
            let output_start = (token * query_heads + query_head) * head_dim;
            for (kv_token, probability) in scores.iter().enumerate() {
                let value_start = (kv_head * key_value_tokens + kv_token) * head_dim;
                for dimension in 0..head_dim {
                    output[output_start + dimension] +=
                        probability / denominator * value[value_start + dimension];
                }
            }
        }
    }
    Ok(output)
}

// ref: modeling_qwen3_5.py:668-701
pub fn sigmoid_gate(values: &mut [f32], gate: &[f32]) -> Result<()> {
    if values.len() != gate.len()
        || values.is_empty()
        || values.iter().chain(gate).any(|value| !value.is_finite())
    {
        return Err(EngineError::Shape(
            "attention output and query gate differ".into(),
        ));
    }
    for (value, gate) in values.iter_mut().zip(gate) {
        *value *= 1.0 / (1.0 + (-gate).exp());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "index {index}: {left} != {right}"
            );
        }
    }

    #[test]
    fn qwen_norm_and_swiglu_match_exact_equations() {
        let values = [1.0, -2.0, 3.0, -4.0];
        let weight = [0.1, -0.2, 0.3, -0.4];
        let norm = rms_norm_1p_weight(&values, 1, 4, &weight, 1e-6).unwrap();
        close(&norm, &[0.4016632, -0.5842374, 1.4240787, -0.8763561], 2e-6);
        let fused = swiglu(&[0.0, 1.0, -1.0], &[2.0, 3.0, 4.0]).unwrap();
        close(&fused, &[0.0, 2.1931758, -1.0757657], 2e-6);
        let gated = rms_norm_gated(
            &values,
            &[0.1, -0.2, 0.3, -0.4],
            1,
            4,
            &[0.5, 1.0, 1.5, 2.0],
            1e-6,
        )
        .unwrap();
        close(
            &gated,
            &[0.009584765, 0.06575096, 0.28317162, 0.4689234],
            2e-6,
        );
    }

    #[test]
    fn partial_rope_leaves_non_rotary_dimensions_unchanged() {
        let mut query = vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0];
        let mut key = vec![-1.0, 0.5, 2.0, -3.0, 7.0, 8.0];
        apply_partial_rope(&mut query, &mut key, 1, 1, 6, 4, 3, 10_000.0).unwrap();
        close(
            &query,
            &[-1.4133525, 1.8791181, -2.8288574, 4.0581913, 9.0, 10.0],
            2e-6,
        );
        close(
            &key,
            &[0.70775247, 0.5897615, -2.121105, -2.9836524, 7.0, 8.0],
            2e-6,
        );
        close(&query[4..], &[9.0, 10.0], 0.0);
        close(&key[4..], &[7.0, 8.0], 0.0);
    }

    #[test]
    fn convolution_update_retains_only_the_latest_kernel_values() {
        let mut state = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let output = causal_conv_silu_update(
            &[5.0, -5.0],
            &mut state,
            &[0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
            2,
            4,
        )
        .unwrap();
        close(&state, &[2.0, 3.0, 4.0, 5.0, -2.0, -3.0, -4.0, -5.0], 0.0);
        close(&output, &[3.928055, -0.14227761], 2e-6);
    }

    #[test]
    fn recurrent_delta_step_updates_state_and_output() {
        let mut state: Vec<f32> = (1..=12).map(|value| value as f32 / 100.0).collect();
        let output = recurrent_gated_delta_step(
            &[1.0, 2.0, -1.0, 0.5, 1.5, -0.5],
            &[0.5, -1.0, 2.0, 1.0, -0.5, 0.25],
            &[1.0, -2.0, 0.5, 1.5],
            &[-0.2, -0.7],
            &[0.6, 0.25],
            &mut state,
            2,
            3,
            2,
        )
        .unwrap();
        close(
            &output,
            &[-0.20637584, 0.44671553, 0.0062854784, -0.0195187],
            2e-6,
        );
        close(
            &state,
            &[
                0.1356092,
                -0.24969745,
                -0.23028184,
                0.56489336,
                0.5506241,
                -1.0151644,
                0.13890404,
                0.3613783,
                -0.00737885,
                -0.11116721,
                0.08066015,
                0.14000311,
            ],
            2e-6,
        );
    }

    #[test]
    fn grouped_query_attention_matches_pinned_python_oracle() {
        let query = [1.0, 0.0, -1.0, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.5, 2.0];
        let key = [0.2, 0.4, -0.5, 1.0, -0.25, 0.75];
        let value = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0];
        let output = grouped_query_attention(&query, &key, &value, 2, 1, 2, 2, 3, 0).unwrap();
        close(
            &output,
            &[
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                0.07204375,
                1.3040327,
                2.5360217,
                -0.37731764,
                0.9670118,
                2.3113413,
            ],
            2e-6,
        );
        let mut gated = output.clone();
        sigmoid_gate(&mut gated, &[0.0; 12]).unwrap();
        for (actual, ungated) in gated.iter().zip(output) {
            assert!((actual - ungated * 0.5).abs() <= 1e-7);
        }
    }

    #[test]
    fn delta_prefill_sequence_matches_recurrent_python_oracle() {
        let mut state: Vec<f32> = (1..=8).map(|value| value as f32 / 100.0).collect();
        let output = recurrent_gated_delta_sequence(
            &[1.0, 2.0, 0.5, -1.0, -1.0, 0.25, 2.0, 1.0],
            &[0.5, -0.25, 1.0, 0.5, 0.75, 1.25, -0.5, 2.0],
            &[1.0, -1.0, 0.5, 2.0, -2.0, 0.25, 1.5, -0.5],
            &[-0.2, -0.4, -0.1, -0.3],
            &[0.6, 0.25, 0.8, 0.4],
            &mut state,
            2,
            2,
            2,
            2,
        )
        .unwrap();
        close(
            &output,
            &[
                0.018123373,
                0.025_890_53,
                -0.019077633,
                -0.021197364,
                -0.0373289,
                0.32707188,
                0.17473069,
                0.24809137,
            ],
            2e-6,
        );
        close(
            &state,
            &[
                -0.35503477,
                -0.36432835,
                -1.6378022,
                0.44982785,
                -0.04187046,
                0.41027403,
                0.636288,
                -0.03601411,
            ],
            2e-6,
        );
    }
}
