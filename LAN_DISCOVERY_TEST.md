# LAN Discovery Test Guide

This guide focuses on the current LAN beacon and pool membership flow.

## What This Covers

- pool creation
- pool join via pool credentials
- LAN beacon discovery
- peer visibility inside a pool

## Prerequisites

- 2 or more devices on the same WiFi or wired LAN
- this repo installed on each device with `./install.sh`

## 1. Initialize Each Device

On each machine:

```bash
mesh init --network-id test-network --name "Device 1"
```

## 2. Create A Pool On One Device

On the first device:

```bash
mesh pool-create --name "LAN Test Pool"
```

Save the output values:

- pool ID
- pool root public key

## 3. Join The Pool On Other Devices

On each additional device:

```bash
mesh pool-join \
  --pool-id <POOL_ID> \
  --pool-root-pubkey <POOL_ROOT_PUBKEY> \
  --name "LAN Test Pool"
```

## 4. Start The Agent

On every device:

```bash
mesh start
```

Expected behavior:

- LAN beacon broadcaster starts
- LAN beacon listener starts
- peers in the same pool begin appearing within a few seconds

## 5. Verify Discovery

List configured pools:

```bash
mesh pool-list
```

Inspect discovered peers:

```bash
mesh pool-peers --pool-id <POOL_ID>
```

You should see:

- peer node IDs
- LAN addresses
- discovery source
- online/stale status
- last-seen timing

## Success Criteria

- multiple devices can join the same pool
- beacon discovery works on a shared LAN
- peers appear in `pool-peers`
- discovery works without requiring public internet services

## Current Scope

The current repo already uses the pool and beacon flow as part of the active connectivity model. This is not a placeholder path, but it is also not the entire production networking story. The full runtime also includes:

- direct and relayed connectivity selection
- dedicated tensor-plane transport
- durable control-plane dispatch for inference assignments

## Security Notes

- pool membership is anchored by pool root key material
- LAN discovery should still be treated as a trusted-network workflow
- relay and direct connectivity hardening continue outside the beacon layer

## Troubleshooting

No peers discovered:

- verify same subnet or multicast-friendly LAN
- check firewall settings for UDP `42424`
- verify all devices joined the same pool

Peers become stale:

- check whether the agent is still running
- inspect agent logs under `~/.meshnet/logs/`

Useful packet check:

```bash
sudo tcpdump -i any udp port 42424
```
