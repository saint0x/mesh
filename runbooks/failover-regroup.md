# Failover And Regroup

## What To Check First

1. Open the job scheduler status for the affected session.
2. Inspect:
   - `regroup_events`
   - `scheduler_events`
   - `recent_regroup_event_count`
   - `recent_regroup_failure_count`
   - `recent_avg_recovery_latency_ms`
   - `recent_peak_recovery_latency_ms`
3. Inspect serving-group membership and decode-queue state for the session.

## Expected Event Shapes

### Healthy shrink

You should see:

- regroup/member status changes on the lost participant
- `decode_regroup_shrink`
- queue status return to `ready`, then `leased` or `active`

### Replacement with checkpoint transfer

You should see:

- `decode_regroup_transfer`
- decode queue move to `blocked_on_transfer`
- transfer metadata appear in the session KV transfer list
- queue status return to `ready` after transfer completion

### Terminal failure

You should see:

- `decode_regroup_failed`
- session and job state settle to `failed`
- serving-group and lease cleanup follow

## Recovery Steps

1. Verify whether the failover was supposed to shrink, replace, or fail.
2. If replacement was expected, confirm a real persisted checkpoint exists.
3. If a checkpoint exists but recovery latency is rising:
   - inspect transfer progress,
   - inspect whether the replacement worker is online,
   - inspect whether the decode queue ever returned to `ready`.
4. If repeated regroup failures occur for the same cohort, stop treating it as
   a transient scheduler issue and inspect planner legality or worker capacity.
5. After recovery, confirm that:
   - stale leases are gone,
   - the serving group reflects the new ownership,
   - the next decode step is claimable or active.
