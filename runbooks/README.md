# Zip Operations Runbooks

These runbooks cover the main operator-facing failure modes exposed by the
current production `zip` control plane and agent runtime.

- [Scheduler Stalls](/Users/deepsaint/Desktop/meshnet/runbooks/scheduler-stalls.md)
- [KV Pressure](/Users/deepsaint/Desktop/meshnet/runbooks/kv-pressure.md)
- [Failover And Regroup](/Users/deepsaint/Desktop/meshnet/runbooks/failover-regroup.md)

Use them together with the scheduler status surfaces:

- `GET /api/inference/networks/:network_id/scheduler-status`
- `GET /api/inference/jobs/:job_id/scheduler-status`
