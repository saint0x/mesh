# Scheduler Stalls

## What To Check First

1. Open the network scheduler status for the affected network.
2. Compare `decode_queue_depth`, `runnable_sessions`, `blocked_sessions`,
   `leased_sessions`, and `active_sessions`.
3. Inspect `scheduler_events` for the most recent `decode_claimed`,
   `decode_lease_released`, `decode_queue_unblocked`, and `decode_regroup_*`
   events.
4. Inspect each serving group's `lease` and `workload` summary.

## Likely Patterns

### Queue depth rises, runnable stays low

This usually means sessions are blocked behind prefill or KV transfer.

Look at:

- `blocked_on_prefill_sessions`
- `blocked_on_transfer_sessions`
- `recent_regroup_event_count`
- `recent_recovery_sample_count`

If blocked-on-transfer is dominant, move to the KV pressure runbook.

### Runnable stays high, leased and active stay low

This usually means workers are not claiming, leases are stale, or a serving
group lease is pinned to the wrong worker.

Look at:

- recent `decode_claimed` and `decode_lease_released` events
- serving-group `lease.owner_device_id`
- decode-queue `lease_owner_device_id`

If a worker lease is stale, the scheduler reconciliation loop should recover
it. If it does not, verify the worker is still heartbeating and confirm the
session is not stuck in `active` without progress.

### Leased stays high, active stays low

This usually means workers claimed decode work but did not acknowledge or did
not start stepping the batch.

Look at:

- `decode_claimed` without matching `active` queue progression
- recent agent logs for decode-batch admission
- per-group workload telemetry and target-versus-actual batch history

## Recovery Steps

1. Confirm whether the pressure is prefill-bound, transfer-bound, or lease-bound.
2. If lease-bound, verify the owning device is still online.
3. If the owner is dead, confirm a failover/regroup event was recorded.
4. If the owner is healthy but not making progress, release or expire the
   stuck lease through normal worker/control-plane recovery flows rather than
   mutating rows manually.
5. Re-check that `runnable_sessions` or `active_sessions` move within the next
   scheduler cycle.
