# KV Pressure

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
3. Inspect the `kv_residency` entries for the affected sessions and the
   `transfers` list for stuck or repeated handoffs.

## Likely Patterns

### Pinned slices stay high and evictable stays low

The serving group is carrying too much active decode state to reclaim memory
aggressively. Expect decode admission to defer more sessions and batch KV
footprint to rise.

### Active transfers stay high and bytes transferred barely move

Checkpoint-mediated handoff is the bottleneck. Expect:

- more `blocked_on_transfer_sessions`
- higher `checkpoint_fallback_rate`
- more `decode_regroup_transfer` or `decode_pending_transfer` states

### Remote slices rise while payload bytes stay flat

The system is leaning on remote references rather than local checkpoint
materialization. This is usually fine unless decode latency is regressing or
remote access is repeatedly falling back to checkpoint transfer.

## Recovery Steps

1. Identify whether pressure is local memory saturation or transfer backlog.
2. For transfer backlog:
   - confirm the latest checkpoint exists,
   - confirm the transfer target is still online,
   - confirm bytes transferred is still advancing.
3. For local KV saturation:
   - inspect which sessions are pinned for decode,
   - verify evictable slices are actually being reclaimed,
   - confirm deferred decode sessions drop after pressure subsides.
4. If checkpoint fallback rate spikes unexpectedly, inspect whether a remote
   access path regressed into full checkpoint handoff.

Do not delete residency or checkpoint rows manually unless you are deliberately
retiring a dead test environment and no active jobs remain.
