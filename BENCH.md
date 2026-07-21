# Mesh Bench

## Mac + GMK Heterogeneous TinyLlama

Date: 2026-07-21

Commit: `11405cb`

Model: `tinyllama-1.1b-chat-v1.0`

Topology:

- Mac Metal worker: shard `0..512`, memory budget `8GB`
- GMK ROCm worker: shard `512..2048`, memory budget `64GB`
- Weighted artifact split: KV groups `1,3`, MLP weights `1,5`
- Transport: LAN tensor plane through the GMK WSL bridge

Runtime notes:

- Decode lease failures now report terminal assignment failure instead of recycling unrecoverable work.
- Serving tensor transports are reused across decode steps instead of rebinding streams per token.
- The cached-transport fix removed the prior position-81 timeout on the 96-token run.
- Short-run throughput improved from `0.5701 tok/s` to `0.8756 tok/s`, a `53.6%` increase, after transport reuse.

| Label | Status | Tokens | TTFT ms | Wall ms | Tok/s |
|---|---:|---:|---:|---:|---:|
| latency-short | completed | 32 | 1596 | 36546 | 0.8756 |
| reasoning-medium | completed | 96 | 1301 | 160640 | 0.5976 |
| code-systems | completed | 128 | 5205 | 215043 | 0.5952 |
| creative-recall | completed | 96 | 4254 | 170129 | 0.5643 |
| long-context-style | completed | 160 | 2205 | 221968 | 0.7208 |

Qualitative read:

- Runtime stability is materially better after transport reuse: the full 5-prompt suite completed without retry storms or WSL socket exhaustion.
- TTFT is usable for this prototype path, ranging from `1301ms` to `5205ms`.
- Decode throughput remains low for production claims, roughly `0.56-0.88 tok/s`, so the heterogeneous path is correct but not yet performance-positive.
- TinyLlama is acceptable as a runtime smoke model but not a quality benchmark model. It answered the Big Bang prompt coherently but became repetitive or nonsensical on systems-code, creative, and long-context prompts.

Raw qualitative outputs are preserved in `/tmp/mesh-mac-gmk/bench/BENCH.md` for this run.
