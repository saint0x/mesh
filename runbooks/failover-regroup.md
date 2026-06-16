# Failover And Regroup

Use this runbook when a serving participant dies, a cohort must shrink or
replace a worker, or decode recovery is slower than production policy allows.

## What Healthy Looks Like

Healthy failover is visible and bounded. After a participant loss, expect:

- explicit `decode_regroup_*` events
- readiness surfaces that explain any remaining blockers
- queue state that either returns to `ready`, moves through
  `blocked_on_transfer`, or settles to a deliberate terminal failure
- leases and serving groups that converge on the new membership without manual
  row editing

## What To Check First

1. Open the job scheduler status for the affected job.
2. Inspect:
   - `regroup_events`
   - `scheduler_events`
   - `readiness.ready`
   - `readiness.blockers`
   - `recent_regroup_event_count`
   - `recent_regroup_failure_count`
   - `recent_regroup_transfer_count`
   - `recent_regroup_shrink_count`
   - `recent_regroup_replace_count`
   - `checkpoint_handoff_transfer_count`
   - `live_kv_handoff_transfer_count`
   - `recent_avg_recovery_latency_ms`
   - `recent_peak_recovery_latency_ms`
   - `recent_avg_degraded_duration_ms`
   - `recent_peak_degraded_duration_ms`
   - `recent_avg_post_failover_throughput_loss_pct`
   - `recent_peak_post_failover_throughput_loss_pct`
3. Inspect the affected decode queue row and its `lease_session_members`.
4. Inspect serving-group membership and transfer state for the same cohort.

## Readiness Gates

The readiness surface should stay green only when all of the following hold:

- `checkpoint_fallback_rate` is at or below the configured threshold
- recent regroup failures stay within policy, ideally zero
- peak recovery latency stays within the allowed window
- average post-failover throughput loss stays within the allowed bound
- peak decode batch size stays at or above the minimum threshold

If `readiness.ready` is `false`, use `readiness.blockers` as the primary
operator triage queue before debugging internals.

## Expected Event Shapes

### Healthy shrink

You should see:

- regroup or member status changes on the lost participant
- `decode_regroup_shrink`
- lease and cohort membership settle to the smaller serving group
- queue status return to `ready`, then `leased` or `active`

### Replacement with live KV handoff

You should see:

- `decode_regroup_live_kv` or `decode_regroup_replace`
- transfer metadata reflect live handoff behavior
- queue returns to `ready` without extended checkpoint fallback pressure

### Replacement with checkpoint transfer

You should see:

- `decode_regroup_transfer`
- decode queue move to `blocked_on_transfer`
- transfer metadata appear in the session transfer list
- queue status return to `ready` after transfer completion

### Terminal failure

You should see:

- `decode_regroup_failed`
- session and job state settle to `failed`
- serving-group and lease cleanup follow

## Recovery Steps

1. Verify whether the system chose to shrink, replace, or fail.
2. If replacement was expected, confirm a real persisted checkpoint exists or
   a live KV handoff path is available.
3. If recovery latency is rising:
   - inspect transfer progress,
   - inspect replacement worker liveness,
   - inspect whether the queue ever returned to `ready`,
   - inspect whether the cohort retained its intended pooled target.
4. If repeated regroup failures occur for the same cohort, stop treating it as
   transient scheduler noise and inspect planner legality, worker capacity, or
   artifact compatibility.
5. After recovery, confirm that:
   - stale leases are gone,
   - serving-group membership reflects the new ownership,
   - decode queue state is claimable or active,
   - pooled decode telemetry resumes if the cohort is supposed to continue as a
     pooled group.

## Immediate Escalation Conditions

Escalate directly if any of the following appear:

- failover completes, but the recovered cohort never resumes pooled decode
- regroup finishes, but the next decode lease exits through
  `segment_execution_failed`
- readiness stays green while worker decode stats show no batch progress
- successful post-failover job completion reports positive control-plane output
  but zero worker `total_tokens_generated`
