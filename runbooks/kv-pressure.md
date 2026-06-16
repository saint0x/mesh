# KV Pressure

Use this runbook when decode is blocked behind KV residency pressure, checkpoint
handoff, or transfer backlog.

## What Healthy Looks Like

A healthy KV system may show transient transfer pressure, but it should still
converge toward:

- falling `active_kv_transfer_count`
- advancing `kv_transfer_bytes_transferred`
- bounded `checkpoint_fallback_rate`
- decode queue rows leaving `blocked_on_transfer` and returning to `ready`
- pooled cohorts preserving their target size instead of collapsing into
  singleton execution

## What To Check First

1. Open the scheduler status for the affected job or network.
2. Inspect the KV metrics:
   - `kv_residency_slice_count`
   - `remote_kv_residency_slice_count`
   - `pinned_kv_residency_slice_count`
   - `evictable_kv_residency_slice_count`
   - `kv_cached_tokens`
   - `kv_payload_bytes`
   - `kv_transfer_count`
   - `active_kv_transfer_count`
   - `kv_transfer_bytes_total`
   - `kv_transfer_bytes_transferred`
   - `checkpoint_fallback_transfer_count`
   - `checkpoint_fallback_rate`
   - `checkpoint_handoff_transfer_count`
   - `live_kv_handoff_transfer_count`
3. Inspect the affected session lease or queue row for:
   - `kv_owner_device_id`
   - `kv_sequence_position`
   - `checkpoint`
   - `local_replica`
   - `lease_session_members`
4. Inspect `kv_residency` entries and the `transfers` list for the same cohort.

## Likely Patterns

### Pinned slices stay high and evictable stays low

The serving group is carrying too much active decode state to reclaim memory
aggressively.

Expect:

- higher `deferred_decode_sessions`
- lower admission for new decode work
- growing batch KV footprint

### Active transfers stay high and bytes transferred barely move

Checkpoint-mediated handoff is the bottleneck.

Expect:

- more `blocked_on_transfer_sessions`
- higher `checkpoint_fallback_rate`
- `decode_regroup_transfer` or `decode_pending_transfer` states

If the bytes do not move and the target worker is healthy, inspect whether the
transfer path regressed from live KV access to full checkpoint shipping.

### Remote slices rise while payload bytes stay flat

The system is leaning on remote residency rather than local materialization.

This is fine if:

- latency remains acceptable
- `checkpoint_fallback_rate` stays bounded
- decode cohorts still return to `ready`

It is not fine if the system repeatedly falls back to full checkpoint transfer.

### Cohort target stays high but only one member becomes resume-ready

This means the scheduler preserved the pooled target, but KV materialization did
not catch up across the whole cohort.

Look at:

- per-member `checkpoint`
- per-member `local_replica`
- per-member `kv_sequence_position`
- transfer list progress for the blocked members

Do not paper over this by manually shrinking the cohort unless failover policy
explicitly chose a shrink path.

## Recovery Steps

1. Decide whether the pressure is local memory saturation or transfer backlog.
2. For transfer backlog:
   - confirm the latest checkpoint exists,
   - confirm the transfer target is still online,
   - confirm bytes transferred is still advancing,
   - confirm blocked members in `lease_session_members` are the same members
     reflected in `transfers`.
3. For local KV saturation:
   - inspect which sessions are pinned for active decode,
   - verify evictable slices are actually being reclaimed,
   - verify deferred decode pressure falls after active work completes.
4. If checkpoint fallback rate spikes unexpectedly, inspect whether a live KV
   path regressed into full checkpoint handoff.
5. After recovery, confirm the decode queue leaves `blocked_on_transfer`,
   returns to `ready`, and preserves its intended pooled target.

## Immediate Escalation Conditions

Escalate directly if any of the following appear:

- pooled decode cohorts repeatedly degrade into singleton execution after KV
  recovery
- per-member checkpoint metadata is present but local replica state never
  becomes resume-ready
- transfer pressure clears, but the queue still exits through
  `segment_execution_failed` instead of progressing into active decode

Do not delete residency, transfer, or checkpoint rows manually unless you are
retiring a dead environment and no active jobs remain.
