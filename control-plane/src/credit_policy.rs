use serde::Serialize;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MODEL_SIZE_BUDGET_SCALE_GIB: f64 = 32.0;
const MIN_EXECUTION_TIME_SECS: f64 = 0.001;
const MIN_THROUGHPUT_MULTIPLIER: f64 = 0.5;
const MAX_THROUGHPUT_MULTIPLIER: f64 = 1.5;
const RESOURCE_MULTIPLIER_BASE: f64 = 0.85;
const RESOURCE_MULTIPLIER_SPAN: f64 = 0.30;

#[derive(Debug, Clone)]
pub struct CreditPolicyInput {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_model_bytes: u64,
    pub total_columns: u32,
    pub assignments: Vec<AssignmentCreditInput>,
}

#[derive(Debug, Clone)]
pub struct AssignmentCreditInput {
    pub device_id: String,
    pub execution_time_ms: u64,
    pub assigned_capacity_units: u32,
    pub shard_column_start: u32,
    pub shard_column_end: u32,
    pub available_memory_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreditPolicyOutput {
    pub job_credit_budget: f64,
    pub assignments: Vec<AssignmentCreditOutput>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssignmentCreditOutput {
    pub device_id: String,
    pub credits: f64,
    pub compute_share: f64,
    pub throughput_multiplier: f64,
    pub resource_pressure_multiplier: f64,
    pub normalized_contribution_share: f64,
    pub measured_service_rate: f64,
    pub reference_service_rate: f64,
    pub memory_pressure: f64,
}

pub fn compute_credit_policy(input: CreditPolicyInput) -> CreditPolicyOutput {
    if input.assignments.is_empty() {
        return CreditPolicyOutput {
            job_credit_budget: 0.0,
            assignments: Vec::new(),
        };
    }

    let total_tokens = input
        .prompt_tokens
        .saturating_add(input.completion_tokens)
        .max(1) as f64;
    let model_size_factor =
        (input.total_model_bytes as f64 / GIB / MODEL_SIZE_BUDGET_SCALE_GIB).max(1.0);
    let job_credit_budget = total_tokens * model_size_factor;

    let prepared = input
        .assignments
        .iter()
        .map(|assignment| {
            let shard_columns = assignment
                .shard_column_end
                .saturating_sub(assignment.shard_column_start)
                .max(1);
            let shard_fraction = shard_columns as f64 / input.total_columns.max(1) as f64;
            let compute_units =
                assignment.assigned_capacity_units.max(1) as f64 * shard_fraction.max(f64::EPSILON);
            let execution_time_secs =
                (assignment.execution_time_ms as f64 / 1000.0).max(MIN_EXECUTION_TIME_SECS);
            let service_rate = compute_units / execution_time_secs;
            let estimated_memory_bytes = input.total_model_bytes as f64 * shard_fraction;
            let memory_pressure = if assignment.available_memory_bytes == 0 {
                1.0
            } else {
                (estimated_memory_bytes / assignment.available_memory_bytes as f64).clamp(0.0, 1.0)
            };

            PreparedAssignment {
                device_id: assignment.device_id.clone(),
                compute_units,
                service_rate,
                memory_pressure,
            }
        })
        .collect::<Vec<_>>();

    let total_compute_units = prepared
        .iter()
        .map(|assignment| assignment.compute_units)
        .sum::<f64>()
        .max(f64::EPSILON);
    let reference_service_rate = median(
        prepared
            .iter()
            .map(|assignment| assignment.service_rate)
            .collect(),
    )
    .max(f64::EPSILON);

    let adjusted_weights = prepared
        .iter()
        .map(|assignment| {
            let compute_share = assignment.compute_units / total_compute_units;
            let throughput_multiplier = (assignment.service_rate / reference_service_rate)
                .clamp(MIN_THROUGHPUT_MULTIPLIER, MAX_THROUGHPUT_MULTIPLIER);
            let resource_pressure_multiplier =
                RESOURCE_MULTIPLIER_BASE + (RESOURCE_MULTIPLIER_SPAN * assignment.memory_pressure);
            compute_share * throughput_multiplier * resource_pressure_multiplier
        })
        .collect::<Vec<_>>();

    let total_adjusted_weight = adjusted_weights
        .iter()
        .copied()
        .sum::<f64>()
        .max(f64::EPSILON);

    let assignments = prepared
        .into_iter()
        .zip(adjusted_weights)
        .map(|(assignment, adjusted_weight)| {
            let compute_share = assignment.compute_units / total_compute_units;
            let throughput_multiplier = (assignment.service_rate / reference_service_rate)
                .clamp(MIN_THROUGHPUT_MULTIPLIER, MAX_THROUGHPUT_MULTIPLIER);
            let resource_pressure_multiplier =
                RESOURCE_MULTIPLIER_BASE + (RESOURCE_MULTIPLIER_SPAN * assignment.memory_pressure);
            let normalized_contribution_share = adjusted_weight / total_adjusted_weight;
            AssignmentCreditOutput {
                device_id: assignment.device_id,
                credits: job_credit_budget * normalized_contribution_share,
                compute_share,
                throughput_multiplier,
                resource_pressure_multiplier,
                normalized_contribution_share,
                measured_service_rate: assignment.service_rate,
                reference_service_rate,
                memory_pressure: assignment.memory_pressure,
            }
        })
        .collect();

    CreditPolicyOutput {
        job_credit_budget,
        assignments,
    }
}

#[derive(Debug, Clone)]
struct PreparedAssignment {
    device_id: String,
    compute_units: f64,
    service_rate: f64,
    memory_pressure: f64,
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
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
    fn normalizes_credits_to_job_budget() {
        let result = compute_credit_policy(CreditPolicyInput {
            prompt_tokens: 8,
            completion_tokens: 4,
            total_model_bytes: 64 * 1024 * 1024 * 1024,
            total_columns: 8192,
            assignments: vec![
                AssignmentCreditInput {
                    device_id: "worker-a".into(),
                    execution_time_ms: 500,
                    assigned_capacity_units: 4,
                    shard_column_start: 0,
                    shard_column_end: 4096,
                    available_memory_bytes: 16 * 1024 * 1024 * 1024,
                },
                AssignmentCreditInput {
                    device_id: "worker-b".into(),
                    execution_time_ms: 500,
                    assigned_capacity_units: 4,
                    shard_column_start: 4096,
                    shard_column_end: 8192,
                    available_memory_bytes: 16 * 1024 * 1024 * 1024,
                },
            ],
        });

        let credited = result
            .assignments
            .iter()
            .map(|item| item.credits)
            .sum::<f64>();
        approx_eq(credited, result.job_credit_budget);
        approx_eq(result.assignments[0].credits, result.assignments[1].credits);
    }

    #[test]
    fn rewards_faster_service_for_same_assigned_work() {
        let result = compute_credit_policy(CreditPolicyInput {
            prompt_tokens: 8,
            completion_tokens: 4,
            total_model_bytes: 64 * 1024 * 1024 * 1024,
            total_columns: 8192,
            assignments: vec![
                AssignmentCreditInput {
                    device_id: "fast".into(),
                    execution_time_ms: 400,
                    assigned_capacity_units: 4,
                    shard_column_start: 0,
                    shard_column_end: 4096,
                    available_memory_bytes: 16 * 1024 * 1024 * 1024,
                },
                AssignmentCreditInput {
                    device_id: "slow".into(),
                    execution_time_ms: 1200,
                    assigned_capacity_units: 4,
                    shard_column_start: 4096,
                    shard_column_end: 8192,
                    available_memory_bytes: 16 * 1024 * 1024 * 1024,
                },
            ],
        });

        assert!(result.assignments[0].credits > result.assignments[1].credits);
        assert!(
            result.assignments[0].throughput_multiplier
                > result.assignments[1].throughput_multiplier
        );
    }

    #[test]
    fn rewards_larger_assigned_work_even_when_service_is_equal() {
        let result = compute_credit_policy(CreditPolicyInput {
            prompt_tokens: 8,
            completion_tokens: 4,
            total_model_bytes: 64 * 1024 * 1024 * 1024,
            total_columns: 8192,
            assignments: vec![
                AssignmentCreditInput {
                    device_id: "large".into(),
                    execution_time_ms: 1000,
                    assigned_capacity_units: 8,
                    shard_column_start: 0,
                    shard_column_end: 6144,
                    available_memory_bytes: 24 * 1024 * 1024 * 1024,
                },
                AssignmentCreditInput {
                    device_id: "small".into(),
                    execution_time_ms: 1000,
                    assigned_capacity_units: 4,
                    shard_column_start: 6144,
                    shard_column_end: 8192,
                    available_memory_bytes: 24 * 1024 * 1024 * 1024,
                },
            ],
        });

        assert!(result.assignments[0].compute_share > result.assignments[1].compute_share);
        assert!(result.assignments[0].credits > result.assignments[1].credits);
    }

    #[test]
    fn rewards_higher_memory_pressure_when_other_factors_match() {
        let result = compute_credit_policy(CreditPolicyInput {
            prompt_tokens: 8,
            completion_tokens: 4,
            total_model_bytes: 64 * 1024 * 1024 * 1024,
            total_columns: 8192,
            assignments: vec![
                AssignmentCreditInput {
                    device_id: "tight".into(),
                    execution_time_ms: 800,
                    assigned_capacity_units: 4,
                    shard_column_start: 0,
                    shard_column_end: 4096,
                    available_memory_bytes: 40 * 1024 * 1024 * 1024,
                },
                AssignmentCreditInput {
                    device_id: "roomy".into(),
                    execution_time_ms: 800,
                    assigned_capacity_units: 4,
                    shard_column_start: 4096,
                    shard_column_end: 8192,
                    available_memory_bytes: 80 * 1024 * 1024 * 1024,
                },
            ],
        });

        assert!(result.assignments[0].memory_pressure > result.assignments[1].memory_pressure);
        assert!(
            result.assignments[0].resource_pressure_multiplier
                > result.assignments[1].resource_pressure_multiplier
        );
        assert!(result.assignments[0].credits > result.assignments[1].credits);
    }
}
