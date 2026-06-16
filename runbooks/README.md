# Zip Operations Runbooks

These runbooks cover the current production `zip` scheduler and runtime.
They are written for the pooled decode architecture now shipped in Mesh.

## Start Here

Use these status surfaces first:

- `GET /api/inference/networks/:network_id/scheduler-status`
- `GET /api/inference/jobs/:job_id/scheduler-status`

Use the network view when the problem spans multiple jobs or workers. Use the
job view when one session cohort is stuck, degraded, or failing over.

## Healthy Production Expectations

For a healthy pooled decode system, expect all of the following to be true:

- decode cohorts expose `lease_session_members` rather than only a flat
  `lease_session_ids` list
- decode leases expose `lease_target_session_count` and
  `lease_target_batch_size`
- pooled queue rows expose `pooled_ready_sessions`, `pooled_blocked_sessions`,
  `pooled_leased_sessions`, and `pooled_active_sessions`
- scheduler status exposes `peak_batch_size` and batch telemetry once decode
  traffic is flowing
- worker stats show positive `total_tokens_generated` for successful work
- worker stats show real pooled decode activity through
  `decode_multi_session_microbatches` and `decode_batch_size_peak`

If these signals are missing during real pooled decode traffic, treat that as a
regression.

## Retired Failure Signatures

The following signatures are not expected in the current production design:

- pooled decode lease forms correctly, but runtime only ever executes singleton
  decode batches
- pooled decode lease ends in `decode_lease_released` with
  `segment_execution_failed` because the runtime reported a local resumed
  segment id instead of the canonical control-plane segment id
- jobs complete in the control plane but worker stats still report
  `total_tokens_generated = 0`

If any of those reappear, do not treat them as normal scheduler noise. Escalate
them as engine regressions.

## Which Runbook To Use

- [Scheduler Stalls](/Users/deepsaint/Desktop/meshnet/runbooks/scheduler-stalls.md)
  when queue depth, lease state, or decode progress is stuck
- [KV Pressure](/Users/deepsaint/Desktop/meshnet/runbooks/kv-pressure.md)
  when sessions are blocked on transfer, fallback, or residency pressure
- [Failover And Regroup](/Users/deepsaint/Desktop/meshnet/runbooks/failover-regroup.md)
  when a participant dies, a cohort shrinks, or a replacement worker is needed

## Escalation Rule

If the scheduler surface says the system is healthy but worker-level decode
stats disagree, trust the mismatch and escalate. In this architecture, control
plane state and runtime batch telemetry are both first-class production signals.
