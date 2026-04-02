# Ready To Use

This file is a short operational note for local bring-up.

## Bring-Up

Start the relay:

```bash
mesh-relay
```

Start the control plane:

```bash
mesh-control-plane
```

Start a worker:

```bash
mesh init --network-id demo --name "Worker 1"
mesh join-ring --model-id llama-70b
mesh start
```

Start another worker:

```bash
export MESHNET_HOME=~/.meshnet-worker2
mesh init --network-id demo --name "Worker 2"
mesh join-ring --model-id llama-70b
mesh start
```

Submit a test inference request:

```bash
mesh inference --prompt "Hello world" --max-tokens 10 --model-id llama-70b
```

## Reality Check

- the dispatch and execution path is implemented
- the dedicated tensor data plane is active
- current validation still leans heavily on deterministic scenarios and mock-weight paths
- full-suite Rust test stability is still an active hardening area

For more detail, use:

- [QUICKSTART.md](/Users/deepsaint/Desktop/meshnet/QUICKSTART.md)
- [FINAL_STATUS.md](/Users/deepsaint/Desktop/meshnet/FINAL_STATUS.md)
