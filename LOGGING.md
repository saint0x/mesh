# Mesh Agent Logging Guide

## Overview

The Mesh agent uses production-grade structured logging with:
- **Daily log rotation** to `~/.meshnet/logs/agent.log`
- **Dual output** (file + stdout)
- **Environment-based configuration** via `RUST_LOG`

## Quick Start

### Default Logging (Info level)

```bash
# Start agent with default info-level logging
cargo run --bin agent -- start --relay /ip4/127.0.0.1/tcp/4001
```

Logs will be written to:
- **Console**: Pretty-formatted output
- **File**: `~/.meshnet/logs/agent.log` (rotated daily)

### Custom Log Levels

Use the `--log-level` flag or `RUST_LOG` environment variable:

```bash
# Debug level logging
cargo run --bin agent -- start --relay /ip4/127.0.0.1/tcp/4001 --log-level debug

# Trace level logging
RUST_LOG=trace cargo run --bin agent -- start --relay /ip4/127.0.0.1/tcp/4001
```

## Log Levels

From most to least verbose:

| Level | Use Case | Example |
|-------|----------|---------|
| `trace` | Deep debugging (very verbose) | Protocol-level details, every event |
| `debug` | Development debugging | Job processing steps, network events |
| `info` | **Production default** | Startup, job completions, connections |
| `warn` | Warnings and recoverable errors | Failed jobs, retries, deprecations |
| `error` | Critical errors | Unrecoverable failures |

## Advanced Configuration

### Module-Level Filtering

Control verbosity per module using `RUST_LOG`:

```bash
# Debug agent code, info for dependencies
RUST_LOG=agent=debug,info cargo run --bin agent -- start

# Trace network module only
RUST_LOG=agent::network=trace,info cargo run --bin agent -- start

# Debug libp2p, trace job executor
RUST_LOG=agent::executor=trace,libp2p=debug,info cargo run --bin agent -- start
```

### Common Patterns

```bash
# Quiet mode (errors only)
cargo run --bin agent -- start --log-level error

# Network debugging
RUST_LOG=agent::network=trace,libp2p=debug cargo run --bin agent -- start

# Job execution debugging
RUST_LOG=agent::executor=trace cargo run --bin agent -- start

# Full system trace (extremely verbose!)
RUST_LOG=trace cargo run --bin agent -- start
```

## Log File Rotation

Logs are automatically rotated **daily**:

```
~/.meshnet/logs/
├── agent.log                 # Current log file
├── agent.log.2026-01-03     # Yesterday's logs
└── agent.log.2026-01-02     # Day before
```

- **Retention**: Managed manually (delete old files as needed)
- **Format**: Plain text (no ANSI colors in log files)
- **Size**: Unbounded (rotation is time-based, not size-based)

## Viewing Logs

### Tail live logs

```bash
tail -f ~/.meshnet/logs/agent.log
```

### Search logs

```bash
# Find all job completions
grep "Job completed" ~/.meshnet/logs/agent.log

# Find errors
grep "ERROR" ~/.meshnet/logs/agent.log

# Follow specific job
grep "job_id=123e4567" ~/.meshnet/logs/agent.log
```

### Parse structured logs

Logs include structured fields for parsing:

```
2026-01-04T15:30:45.123Z INFO agent::executor: Job completed job_id="abc123" duration_ms=250 success=true
```

Extract with tools like `jq`, `awk`, or log analysis platforms.

## Metrics vs Logs

- **Logs** (this guide): Detailed event trail, debugging, auditing
- **Metrics** (`mesh metrics`): Aggregated statistics (job count, success rate, uptime)

Use `mesh metrics` for quick stats, logs for troubleshooting.

## Troubleshooting

### Logs not appearing?

1. Check log directory exists: `ls ~/.meshnet/logs/`
2. Verify permissions: `ls -la ~/.meshnet/`
3. Try creating directory manually: `mkdir -p ~/.meshnet/logs`

### Too verbose?

Reduce log level:
```bash
cargo run --bin agent -- start --log-level warn
```

### Missing important events?

Increase log level:
```bash
RUST_LOG=debug cargo run --bin agent -- start
```

## Production Recommendations

For production deployments:

1. **Use `info` level** (default) - Good balance of detail and noise
2. **Monitor log file size** - Set up logrotate or similar
3. **Centralize logs** - Ship to Elasticsearch, Splunk, etc.
4. **Alert on errors** - Grep for `ERROR` and trigger alerts
5. **Sample trace logs** - Only enable `trace` for specific debugging sessions

## Examples

### Normal Operation

```bash
$ cargo run --bin agent -- start
2026-01-04T15:30:00.000Z  INFO agent: Production logging initialized log_dir="~/.meshnet/logs" level="info"
2026-01-04T15:30:00.100Z  INFO agent::network: Connecting to relay relay="/ip4/127.0.0.1/tcp/4001"
2026-01-04T15:30:01.200Z  INFO agent::network: Relay connected
2026-01-04T15:30:05.500Z  INFO agent::executor: Job received job_id="abc123" workload="embeddings"
2026-01-04T15:30:05.750Z  INFO agent::executor: Job completed job_id="abc123" duration_ms=250 success=true
```

### Debug Mode

```bash
$ RUST_LOG=debug cargo run --bin agent -- start
2026-01-04T15:30:00.000Z DEBUG agent::network: Building swarm local_peer="12D3KooW..."
2026-01-04T15:30:00.050Z DEBUG agent::network: Adding TCP transport
2026-01-04T15:30:00.100Z DEBUG agent::network: Adding relay client
2026-01-04T15:30:00.150Z DEBUG agent::network: Swarm built successfully
2026-01-04T15:30:05.500Z DEBUG agent::executor: Deserializing job payload size=128
2026-01-04T15:30:05.550Z DEBUG agent::executor: Executing embeddings workload input_len=50
2026-01-04T15:30:05.750Z DEBUG agent::executor: Serializing job result vector_dim=384
```

## See Also

- [Agent Configuration](README.md) - General agent setup
- [Metrics Guide](METRICS.md) - Understanding agent metrics
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
