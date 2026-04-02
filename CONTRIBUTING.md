# Contributing to Mesh

This document reflects the repo as it exists today.

## Prerequisites

- Rust stable
- SQLite tooling only if you are inspecting the control-plane database directly
- Docker only if you are using repo-specific helper workflows that require it
- Fozzy CLI for system validation

## Repository Layout

```text
meshnet/
├── agent/            agent library, daemon, CLI, inference coordinator, tensor plane
├── control-plane/    Axum + SQLite control plane
├── relay-server/     libp2p relay service
├── tests/            Fozzy scenarios
├── scripts/          helper scripts
└── tl-client.sh      repo version-control wrapper
```

## Version Control

Use `./tl-client.sh` for repository version-control operations instead of raw git workflows.

Typical flow:

```bash
./tl-client.sh setup
./tl-client.sh save
./tl-client.sh publish HEAD
./tl-client.sh push
```

## Build

```bash
cargo build --workspace
```

## Test

Fozzy comes first for system and production-readiness work.

Recommended baseline:

```bash
fozzy map suites --root . --scenario-root tests --profile pedantic --json
fozzy doctor --deep --scenario tests/production_dispatch.fozzy.json --runs 5 --seed 424242 --json
fozzy test --det --strict tests/production_dispatch.fozzy.json tests/live_relay_runtime.fozzy.json --json
fozzy run tests/production_dispatch.fozzy.json --det --record /tmp/production_dispatch_trace.fozzy --json
fozzy trace verify /tmp/production_dispatch_trace.fozzy --strict --json
fozzy replay /tmp/production_dispatch_trace.fozzy --json
fozzy ci /tmp/production_dispatch_trace.fozzy --json
```

Then run Rust tests:

```bash
cargo test --workspace
```

Current note:

- deterministic Fozzy checks for the main production-dispatch scenario are passing
- the full Rust workspace suite still has at least one intermittent connectivity failure that should be treated as active hardening work

## Formatting And Linting

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

## Coding Guidelines

- use typed errors and add context on failure paths
- prefer `Result`-based APIs over panicking code in production paths
- keep control-plane and hot-path tensor concerns separate
- add tests when changing dispatch, connectivity, recovery, or tensor-plane behavior

## Documentation Guidelines

When you update docs:

- do not use stale “phase complete” language unless it is explicitly historical
- do not claim all tests pass unless you verified that exact statement now
- do not describe removed or unsupported runtime modes as active options
- prefer current repo paths and commands over aspirational workflows

## Pull Requests

A good change should include:

- code updates
- doc updates when behavior changed
- Fozzy validation for system-level changes
- Rust test results
