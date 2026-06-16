# Scheduler Stalls

Use this runbook when decode work is queued but end-to-end inference throughput
has stopped improving.

## What Healthy Looks Like

For a healthy pooled decode cohort, expect this progression:

1. `blocked_on_prefill` or `blocked_on_transfer`
2. `ready`
3. `decode_claimed`
4. `leased`
5. `active`
6. batch telemetry appears with nonzero `peak_batch_size` or
   `peak_active_decode_sessions`

When a cohort is meant to pool, the lease should expose:

- `lease_session_members`
- `lease_target_session_count`
- `lease_target_batch_size`
- nonzero `pooled_*` counters

If a lease claims to be pooled but the cohort only exposes singleton execution
signals, treat that as a regression.

## What To Check First

1. Open the network scheduler status for the affected network.
2. Compare `decode_queue_depth`, `runnable_sessions`, `blocked_sessions`,
   `leased_sessions`, and `active_sessions`.
3. Inspect `peak_batch_size`, `peak_active_decode_sessions`,
   `deferred_decode_sessions`, and `sessions_with_batch_telemetry`.
4. Inspect `scheduler_events` for the most recent `decode_claimed`,
   `decode_queue_unblocked`, `decode_lease_released`, and `decode_regroup_*`
   events.
5. Inspect the relevant decode queue rows for:
   - `lease_target_session_count`
   - `lease_target_batch_size`
   - `pooled_ready_sessions`
   - `pooled_blocked_sessions`
   - `pooled_leased_sessions`
   - `pooled_active_sessions`
6. Inspect the serving group lease and ownership summary for the same cohort.

## Likely Patterns

### Queue depth rises, runnable stays low

This is usually prefill or transfer pressure, not a decode execution bug.

Look at:

- `blocked_on_prefill_sessions`
- `blocked_on_transfer_sessions`
- `recent_regroup_event_count`
- `recent_recovery_sample_count`

Interpretation:

- high `blocked_on_prefill_sessions` means the scheduler is correctly waiting
  for the prefill cohort to form
- high `blocked_on_transfer_sessions` means the scheduler is correctly waiting
  for KV movement, not losing the cohort

If transfer blockers dominate, switch to the KV pressure runbook.

### Runnable stays high, leased and active stay low

This usually means workers are not claiming, the owning worker is stale, or
the scheduler is repeatedly reconciling a cohort that never starts.

Look at:

- recent `decode_claimed` events
- recent `decode_lease_released` events
- serving-group `lease.owner_device_id`
- decode-queue `lease_owner_device_id`
- whether `readiness.blockers` is already flagging a failover or batch issue

Interpretation:

- no new `decode_claimed` events points to claim starvation or worker liveness
- repeated `decode_claimed` followed by `decode_lease_released` points to an
  execution failure after claim

### Leased stays high, active stays low

This means work was claimed but runtime execution did not advance into a real
decode step.

Look at:

- `decode_claimed` without a follow-on active queue state
- recent agent logs for decode admission and session recovery
- whether the lease exposes `lease_session_members` with valid
  `local_replica` and `checkpoint` data
- whether the queue row still shows the expected pooled target through
  `lease_target_session_count`

Interpretation:

- if `lease_session_members` is incomplete or missing resume-ready replica
  state, the runtime is blocked before decode starts
- if `segment_execution_failed` appears after a valid pooled claim, treat it as
  a hard regression, not expected scheduler behavior

### Active stays high, throughput stays flat

This means the worker entered decode but the batch is ineffective or unhealthy.

Look at:

- `peak_batch_size`
- `peak_active_decode_sessions`
- session `recent_decode_batches`
- worker stats:
  - `decode_microbatches_executed`
  - `decode_multi_session_microbatches`
  - `decode_batch_size_peak`
  - `total_tokens_generated`

Interpretation:

- active decode with `decode_batch_size_peak = 1` across a pooled cohort is a
  runtime regression
- successful control-plane completion with `total_tokens_generated = 0` is an
  accounting regression

## Recovery Steps

1. Decide whether the stall is prefill-bound, transfer-bound, lease-bound, or
   runtime-bound.
2. If prefill-bound, verify the remaining siblings are still progressing toward
   `ready` instead of forcing decode open early.
3. If transfer-bound, move to the KV pressure runbook.
4. If lease-bound, verify the owning device is still online and heartbeating.
5. If runtime-bound, compare control-plane batch telemetry with worker decode
   stats before restarting anything.
6. If the owner is dead, confirm a failover/regroup event was recorded.
7. If the owner is healthy but not making progress, allow normal lease expiry,
   reconciliation, or worker recovery to clean up the cohort rather than
   editing rows manually.
8. Re-check that `runnable_sessions`, `leased_sessions`, `active_sessions`, and
   batch telemetry move within the next scheduler cycle.

## Immediate Escalation Conditions

Escalate directly to an engine regression if any of the following appear:

- pooled cohort lease is present but decode execution never exceeds batch size 1
- `decode_lease_released` with `segment_execution_failed` after a valid pooled
  claim
- control-plane job completion is positive but worker
  `total_tokens_generated` stays zero
