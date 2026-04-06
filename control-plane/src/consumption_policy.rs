const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MODEL_SIZE_BUDGET_SCALE_GIB: f64 = 32.0;

#[derive(Debug, Clone, Copy)]
pub struct ConsumptionQuoteInput {
    pub prompt_tokens: u32,
    pub requested_completion_tokens: u32,
    pub total_model_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ConsumptionSettlementInput {
    pub prompt_tokens: u32,
    pub actual_completion_tokens: u32,
    pub total_model_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ConsumptionPolicyOutput {
    pub model_size_factor: f64,
    pub prompt_credits: f64,
    pub completion_credits: f64,
    pub total_credits: f64,
}

pub fn quote_consumption(input: ConsumptionQuoteInput) -> ConsumptionPolicyOutput {
    compute_consumption(input.prompt_tokens, input.requested_completion_tokens, input.total_model_bytes)
}

pub fn settle_consumption(input: ConsumptionSettlementInput) -> ConsumptionPolicyOutput {
    compute_consumption(input.prompt_tokens, input.actual_completion_tokens, input.total_model_bytes)
}

fn compute_consumption(
    prompt_tokens: u32,
    completion_tokens: u32,
    total_model_bytes: u64,
) -> ConsumptionPolicyOutput {
    let model_size_factor = (total_model_bytes as f64 / GIB / MODEL_SIZE_BUDGET_SCALE_GIB).max(1.0);
    let prompt_credits = prompt_tokens.max(1) as f64 * model_size_factor;
    let completion_credits = completion_tokens as f64 * model_size_factor;
    ConsumptionPolicyOutput {
        model_size_factor,
        prompt_credits,
        completion_credits,
        total_credits: prompt_credits + completion_credits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta < 1e-9, "expected {left} ~= {right}, delta={delta}");
    }

    #[test]
    fn quote_scales_with_model_size_and_requested_tokens() {
        let result = quote_consumption(ConsumptionQuoteInput {
            prompt_tokens: 8,
            requested_completion_tokens: 4,
            total_model_bytes: 64 * 1024 * 1024 * 1024,
        });

        approx_eq(result.model_size_factor, 2.0);
        approx_eq(result.prompt_credits, 16.0);
        approx_eq(result.completion_credits, 8.0);
        approx_eq(result.total_credits, 24.0);
    }

    #[test]
    fn settlement_uses_actual_completion_tokens() {
        let result = settle_consumption(ConsumptionSettlementInput {
            prompt_tokens: 10,
            actual_completion_tokens: 3,
            total_model_bytes: 32 * 1024 * 1024 * 1024,
        });

        approx_eq(result.model_size_factor, 1.0);
        approx_eq(result.prompt_credits, 10.0);
        approx_eq(result.completion_credits, 3.0);
        approx_eq(result.total_credits, 13.0);
    }
}
