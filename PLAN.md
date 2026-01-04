{
  "project": {
    "name": "MeshAI Net",
    "tagline": "Tailscale-style private networks that pool local + trusted compute to run AI workloads.",
    "one_liner": "Create a network, add devices and people, and route AI jobs across pooled compute with simple credits.",
    "goals": [
      "Tailscale-like UX for creating private networks and adding devices/users",
      "Mobile-friendly, opportunistic compute pooling (burst workloads first)",
      "Production-grade security + accounting without distributed-inference overengineering",
      "Works behind NAT with minimal user setup; P2P optional later"
    ],
    "non_goals_v1": [
      "Public anonymous marketplace compute",
      "Precise FLOPs metering / cryptographic verifiable compute",
      "Model-parallel distributed inference across many unreliable mobile nodes",
      "On-chain tokenomics"
    ],
    "target_users": [
      "Builders running local AI across their personal devices",
      "Small teams that want private pooled compute (friends/team tailnet)",
      "Communities who opt-in to share compute within a private network"
    ],
    "success_metrics": {
      "activation": [
        "Network created",
        "First device joined",
        "First workload completed"
      ],
      "retention": [
        "Weekly active networks",
        "Median devices per network",
        "Workloads per network per week"
      ],
      "quality": [
        "Median job latency by workload class",
        "Job success rate",
        "Device churn tolerance (retry success)"
      ],
      "economics": [
        "Relay bandwidth per workload",
        "Credits minted vs burned balance per network"
      ]
    }
  },

  "ux": {
    "design_principles": [
      "Feels like adding devices to a private LAN",
      "Default-safe (charging + Wi-Fi for mobile contributions)",
      "Clear visibility: who’s online, what they can do, what it costs",
      "Trust scoped to a network; no surprise compute sharing"
    ],
    "platforms": {
      "mobile": ["iOS", "Android"],
      "desktop": ["macOS", "Windows", "Linux"],
      "optional": ["Web dashboard"]
    },
    "core_user_flows": {
      "signup_and_bootstrap": {
        "steps": [
          "Create account (passkey / OAuth / email magic link)",
          "Create first Network",
          "Name network + choose defaults",
          "Add first Device via QR / deep-link"
        ],
        "defaults": {
          "mobile_contribution_policy": "charging_and_wifi_only",
          "privacy": "private_network_only",
          "workload_visibility": "members_only",
          "device_discovery": "via_control_plane"
        }
      },
      "create_network": {
        "fields": [
          { "key": "name", "required": true },
          { "key": "visibility", "required": true, "enum": ["private"] },
          { "key": "billing_mode", "required": true, "enum": ["credits_internal"] }
        ],
        "result": "Network created with admin = creator, default policies applied."
      },
      "join_device": {
        "mechanism": ["QR_code", "invite_link", "device_code_pairing"],
        "steps": [
          "Install app/agent",
          "Authenticate user",
          "Select network to join (if multiple)",
          "Device registers keys + capabilities",
          "Device appears online with status"
        ],
        "device_identity": {
          "approach": "device_keypair + installation_id",
          "notes": "Avoid hard-binding to immutable hardware IDs in v1; use platform attestation signals where available, but treat as best-effort."
        }
      },
      "invite_user": {
        "invite_types": ["email", "username", "share_link_with_approval"],
        "roles": ["admin", "member", "guest"],
        "approval": {
          "default": "admin_approves_new_members",
          "optional": "auto_approve_domain_or_allowlist"
        }
      },
      "run_ai_workload": {
        "entry_points": ["mobile_app", "desktop_app", "cli", "sdk"],
        "steps": [
          "Choose workload (e.g., embeddings, OCR, small chat)",
          "Choose routing mode (auto / pinned device / pinned tier)",
          "See estimated credit cost",
          "Run; stream progress; view result",
          "Ledger updates (burn credits)"
        ]
      },
      "contribute_compute": {
        "modes": ["manual_toggle", "scheduled", "policy_driven"],
        "mobile_defaults": [
          "Charging + Wi-Fi only",
          "Stop on thermal threshold",
          "Stop below battery threshold"
        ],
        "desktop_defaults": [
          "Contribute when idle (CPU/GPU usage below threshold)",
          "Allow user to set max utilization"
        ]
      }
    },
    "screens": {
      "network_list": {
        "shows": ["networks", "online_devices_count", "credit_balance_summary"]
      },
      "network_detail": {
        "tabs": ["Devices", "Members", "Workloads", "Ledger", "Settings"]
      },
      "devices_tab": {
        "device_card_fields": [
          "name",
          "platform",
          "online_status",
          "capability_tier",
          "battery_state_if_mobile",
          "thermal_state_if_available",
          "latency_estimate",
          "contribution_mode"
        ],
        "actions": ["pin_workloads_here", "disable_contribution", "remove_device"]
      },
      "workloads_tab": {
        "shows": ["recent_runs", "status", "latency", "cost", "node_used", "logs"]
      },
      "ledger_tab": {
        "shows": ["credits_minted", "credits_burned", "per_member_breakdown", "per_device_breakdown"]
      },
      "settings_tab": {
        "sections": [
          "Policies (who can run what)",
          "Mobile safety defaults",
          "Credit rates",
          "Invites / approvals",
          "Key rotation / security"
        ]
      }
    }
  },

  "network_model": {
    "concepts": {
      "global_platform": "MeshAI Platform",
      "network": "Private trust scope. Users form many independent networks under the platform.",
      "membership": "Users + devices belong to networks with roles and policies."
    },
    "roles": {
      "admin": {
        "can": ["invite/remove members", "set policies", "set credit rates", "rotate keys", "view all logs"]
      },
      "member": {
        "can": ["add own devices", "run allowed workloads", "view own usage", "contribute compute"]
      },
      "guest": {
        "can": ["run restricted workloads", "no ledger admin", "limited visibility"]
      }
    },
    "policy_engine_v1": {
      "dimensions": [
        "workload_class",
        "data_sensitivity_tag",
        "device_tier_min",
        "member_role",
        "time_of_day",
        "mobile_state_constraints"
      ],
      "defaults": {
        "sensitive_data": "admin_only_or_pinned_devices",
        "mobile": "no_sensitive_by_default",
        "guest": "allow_public_models_only"
      }
    }
  },

  "workloads": {
    "v1_supported": [
      {
        "id": "embeddings",
        "description": "Text embeddings endpoint for RAG / search",
        "preferred_tiers": ["tier0", "tier1", "tier2"],
        "latency_sensitivity": "medium",
        "streaming": false
      },
      {
        "id": "ocr_preprocess",
        "description": "Image resize/denoise + OCR extract",
        "preferred_tiers": ["tier0", "tier1", "tier2"],
        "latency_sensitivity": "low",
        "streaming": false
      },
      {
        "id": "small_chat",
        "description": "Small LLM chat for quick tasks (device-local weights or fetched per policy)",
        "preferred_tiers": ["tier2", "tier3"],
        "latency_sensitivity": "high",
        "streaming": true
      },
      {
        "id": "audio_transcribe_chunked",
        "description": "Chunk-based transcription; retries safe",
        "preferred_tiers": ["tier1", "tier2", "tier3"],
        "latency_sensitivity": "low",
        "streaming": "optional"
      }
    ],
    "v2_candidates": [
      "rerank",
      "vision_caption",
      "speculative_decode_draft",
      "batch_jobs_eval_or_synth"
    ],
    "explicitly_deferred": [
      "model_parallel_multi_node_inference_across_unreliable_mobile_nodes",
      "public untrusted job execution without attestation"
    ],
    "data_model": {
      "workload_envelope": {
        "fields": [
          "job_id",
          "network_id",
          "submitted_by_user_id",
          "workload_id",
          "input_manifest",
          "data_sensitivity_tag",
          "resource_request",
          "routing_mode",
          "max_cost_credits",
          "timeout_ms",
          "auth_signature",
          "payload_encryption"
        ]
      },
      "resource_request": {
        "fields": [
          "min_tier",
          "needs_gpu",
          "min_ram_mb",
          "max_latency_ms",
          "requires_pinned_device_ids_optional"
        ]
      }
    }
  },

  "capabilities_and_tiers": {
    "tier_definitions": [
      {
        "tier": "tier0",
        "label": "Phone CPU",
        "typical_devices": ["iPhone/Android CPU"],
        "default_credit_rate": { "earn_per_min": 1, "burn_per_min": 2 }
      },
      {
        "tier": "tier1",
        "label": "Phone/Tablet NPU",
        "typical_devices": ["Android NNAPI", "Apple Neural Engine (best-effort access)"],
        "default_credit_rate": { "earn_per_min": 2, "burn_per_min": 4 }
      },
      {
        "tier": "tier2",
        "label": "Laptop CPU",
        "typical_devices": ["Mac/PC CPU"],
        "default_credit_rate": { "earn_per_min": 4, "burn_per_min": 6 }
      },
      {
        "tier": "tier3",
        "label": "Laptop/Small GPU",
        "typical_devices": ["Integrated GPU / eGPU / modest dGPU"],
        "default_credit_rate": { "earn_per_min": 10, "burn_per_min": 14 }
      },
      {
        "tier": "tier4",
        "label": "Server GPU",
        "typical_devices": ["Desktop/server GPU box"],
        "default_credit_rate": { "earn_per_min": 20, "burn_per_min": 28 }
      }
    ],
    "capability_report": {
      "fields": [
        "cpu_cores",
        "cpu_arch",
        "ram_mb",
        "gpu_present",
        "gpu_vram_mb",
        "npu_present",
        "os",
        "battery_state_optional",
        "thermal_state_optional",
        "network_rtt_ms",
        "uplink_mbps",
        "downlink_mbps"
      ],
      "heartbeat": {
        "interval_seconds": 5,
        "grace_seconds": 20,
        "presence_model": "soft_state"
      }
    }
  },

  "routing_and_scheduling": {
    "routing_modes": [
      { "id": "auto", "description": "Scheduler picks best eligible device" },
      { "id": "pinned_device", "description": "Run only on selected device(s)" },
      { "id": "pinned_tier", "description": "Run on any device within tier constraint" },
      { "id": "local_prefer", "description": "Prefer LAN peers, fallback to relay" }
    ],
    "eligibility_filters": [
      "network_membership",
      "policy_allows(workload, user_role, sensitivity)",
      "device_meets(min_tier, ram, gpu)",
      "device_state_ok(battery/thermal constraints)",
      "device_trust_level_ok(if configured)"
    ],
    "scoring": {
      "features": [
        { "name": "rtt_ms", "weight": -0.30 },
        { "name": "is_plugged_in", "weight": 0.20 },
        { "name": "thermal_headroom", "weight": 0.20 },
        { "name": "idle_score", "weight": 0.15 },
        { "name": "credit_rate_efficiency", "weight": 0.15 }
      ],
      "notes": "Weights are tuneable per network; keep simple linear scoring v1."
    },
    "retries_and_chunking": {
      "default_retry_policy": {
        "max_attempts": 3,
        "backoff_ms": [250, 750, 2000],
        "retry_on": ["disconnect", "timeout", "node_decline"]
      },
      "chunkable_workloads": ["audio_transcribe_chunked", "batch_jobs_eval_or_synth"],
      "idempotency": "Required for chunkable workloads; job_id + chunk_id keys"
    }
  },

  "credits_and_accounting": {
    "ledger_scope": "per_network",
    "philosophy": "Simple time-by-tier metering v1; avoid FLOPs.",
    "minting": {
      "unit": "device_minute",
      "rules": [
        "Mint only when device executes validated work",
        "Mint rate depends on tier and network-configured multiplier",
        "Cap minting on battery or high thermal (mobile)"
      ]
    },
    "burning": {
      "unit": "job_minute",
      "rules": [
        "Burn based on actual runtime rounded up to billing granularity",
        "Additional surcharge for relay bandwidth (optional v1 toggle)"
      ]
    },
    "fairness": {
      "per_member_balances": true,
      "optional_shared_pool": true,
      "defaults": "shared_pool_enabled_with_per_member_visibility"
    },
    "anti_abuse_v1": [
      "Validation checks (spot-check outputs)",
      "Rate limits per device",
      "Require device heartbeat during execution",
      "Refuse minting when device is rooted/jailbroken signal present (best-effort)"
    ],
    "export_and_audit": {
      "events": [
        "credits_minted",
        "credits_burned",
        "job_started",
        "job_completed",
        "job_failed"
      ],
      "formats": ["jsonl", "csv"]
    }
  },

  "security": {
    "threat_model_v1": {
      "assumptions": [
        "Network members are trusted or semi-trusted (friends/team/community)",
        "Control plane is honest and secured",
        "Devices may be intermittently compromised; reduce blast radius by scoping to network"
      ],
      "out_of_scope": [
        "Fully untrusted public execution without attestation",
        "Adversarial model extraction resistance"
      ]
    },
    "identity_and_keys": {
      "user_auth": ["passkeys", "oauth", "magic_link"],
      "device_identity": {
        "device_keypair": "ed25519",
        "device_certificate": "signed by network CA or control plane",
        "rotation": {
          "device_keys": "on reinstall or compromise",
          "network_keys": "admin-triggered or scheduled"
        }
      },
      "network_keying": {
        "model": "per-network trust domain",
        "notes": "Each network has independent membership and cryptographic scope."
      }
    },
    "transport_security": {
      "baseline": "mTLS over QUIC or TLS websockets",
      "encryption": "end-to-end job payload encryption to target device(s)",
      "integrity": "job envelopes signed by scheduler + optionally by submitter"
    },
    "authorization": {
      "checks": [
        "user_role in network",
        "policy allows workload + sensitivity tag",
        "device is member of network and not revoked"
      ]
    },
    "revocation": {
      "member_revocation": "immediate deny on control-plane auth + key rotation recommended",
      "device_revocation": "denylist device cert; require re-approval to rejoin"
    },
    "privacy_controls": {
      "data_sensitivity_tags": ["public", "private", "sensitive"],
      "defaults": {
        "mobile": "private_only; sensitive requires pinned trusted device",
        "guests": "public only"
      }
    }
  },

  "networking": {
    "v1_data_plane": {
      "mode": "relay_first",
      "description": "All nodes maintain outbound connection to relay; relay forwards job streams.",
      "why": "Mobile-friendly, simplest, reliable through NAT.",
      "components": ["relay_gateway", "presence_service", "job_stream_multiplexer"]
    },
    "v2_data_plane": {
      "mode": "p2p_fast_path_with_relay_fallback",
      "techniques": ["ICE-like negotiation", "STUN/TURN equivalent", "local LAN discovery"],
      "notes": "Only after UX and scheduler are proven."
    },
    "protocols": {
      "control_plane_api": "HTTPS JSON (REST) + WebSocket for realtime",
      "job_plane": "QUIC streams preferred; WebSocket fallback",
      "serialization": ["JSON for control", "CBOR or Protobuf for job plane (optional)"]
    }
  },

  "execution_runtime": {
    "node_agent": {
      "responsibilities": [
        "Register device + report capabilities",
        "Maintain presence heartbeat",
        "Accept job envelopes and enforce policy",
        "Run workloads in sandbox",
        "Stream logs/progress/results",
        "Report metering events"
      ],
      "packaging": {
        "desktop": ["system_service", "tray_app_optional"],
        "mobile": ["app_with_background_limited_worker"]
      }
    },
    "sandboxing": {
      "v1": {
        "desktop": ["process isolation", "restricted filesystem", "model cache sandbox dir"],
        "mobile": ["platform sandbox (default)"],
        "notes": "Containers/WASM can be added later; keep v1 practical."
      },
      "v2": {
        "desktop": ["WASM runtime for untrusted plugins", "container option"],
        "notes": "Useful if networks begin to include semi-trusted members."
      }
    },
    "model_management": {
      "modes": [
        "bring_your_own_model",
        "network_managed_models"
      ],
      "distribution": {
        "v1": "device-local model cache; fetch from trusted origin configured per network",
        "integrity": "hash pinning + signed manifests"
      }
    }
  },

  "apis_and_interfaces": {
    "cli": {
      "commands": [
        "meshnet auth login",
        "meshnet net create",
        "meshnet device join",
        "meshnet members invite",
        "meshnet run embeddings --text ...",
        "meshnet run ocr --image ...",
        "meshnet ledger show"
      ],
      "output": "human + json"
    },
    "sdk": {
      "languages_v1": ["TypeScript"],
      "core_calls": [
        "createNetwork()",
        "inviteMember()",
        "registerDevice()",
        "submitJob(workloadId, inputs, routing)",
        "streamJob(jobId)",
        "getLedger(networkId)"
      ]
    },
    "workload_contract": {
      "handler_shape": {
        "inputs": "manifest + blobs/refs",
        "outputs": "json + optional blobs",
        "logging": "structured events",
        "streaming": "token/progress frames for eligible workloads"
      }
    }
  },

  "observability": {
    "principles": [
      "Every job has an end-to-end trace",
      "Users can see why a job routed where it did",
      "Ledger events reconcile from execution events"
    ],
    "entities": ["job", "chunk", "device_session"],
    "telemetry": {
      "logs": "structured json",
      "metrics": [
        "job_latency_ms",
        "job_success_rate",
        "relay_bandwidth_bytes",
        "device_online_rate",
        "credits_minted_per_day",
        "credits_burned_per_day"
      ],
      "tracing": "trace_id propagated through control + job plane"
    },
    "user_facing_debug": {
      "job_debug_view": [
        "eligibility filters result",
        "top candidate scores",
        "selected node + reason",
        "retries + causes"
      ]
    }
  },

  "productization": {
    "pricing_options": {
      "v1": {
        "default": "free_with_reasonable_limits",
        "limits": ["networks_per_user", "relay_bandwidth_per_month", "max_devices_per_network"],
        "upgrade": "paid for higher relay bandwidth + more devices"
      },
      "enterprise_future": [
        "self-hosted coordinator",
        "SSO",
        "audit logs retention",
        "custom policies"
      ]
    },
    "distribution": {
      "desktop": ["signed installers", "auto-update"],
      "mobile": ["app_store", "play_store"]
    }
  },

  "roadmap": {
    "phase_0_prototype": {
      "timebox": "2-4 weeks",
      "deliverables": [
        "Account + create network",
        "Join device (desktop) + presence list",
        "Relay-based job dispatch (embeddings only)",
        "Basic ledger events"
      ]
    },
    "phase_1_mvp": {
      "timebox": "4-8 weeks",
      "deliverables": [
        "Mobile join + contribute mode (charging+wifi)",
        "Workloads: embeddings + OCR + chunked transcription",
        "Job streaming for small_chat (desktop tiers)",
        "Invite members + roles",
        "Ledger UI + export",
        "Revocation + key rotation hooks"
      ]
    },
    "phase_2_polish_and_scale": {
      "timebox": "8-12 weeks",
      "deliverables": [
        "Better scheduling/scoring",
        "LAN/P2P fast-path for same-network peers",
        "Model manifest signing + hash pinning",
        "Bandwidth-aware costing (optional)"
      ]
    },
    "phase_3_optional_exo_like": {
      "timebox": "later",
      "deliverables": [
        "Stable cohort distributed inference experiments",
        "KV-cache streaming between stable nodes",
        "Operator tools for GPU boxes"
      ]
    }
  },

  "implementation_plan_minimal": {
    "stack_choices": {
      "control_plane": {
        "api": "TypeScript (Node) or Rust (Axum) — choose one",
        "db": "Postgres",
        "cache": "Redis",
        "auth": "passkeys/OAuth provider"
      },
      "relay": {
        "language": "Rust or Go",
        "protocol": "QUIC preferred, WebSocket fallback"
      },
      "desktop_agent": {
        "language": "Rust",
        "runtime": "native processes; optional WASM later"
      },
      "mobile_agent": {
        "language": "Swift/Kotlin or RN/Flutter shell with native worker modules",
        "constraints": "iOS background limits; plan for foreground + charging mode"
      }
    },
    "avoid_overengineering": [
      "No blockchain",
      "No perfect attestation",
      "No global marketplace",
      "No FLOPs accounting",
      "No multi-node transformer partitioning v1"
    ]
  },

  "open_questions_but_not_blockers": [
    "Exact iOS background compute strategy (foreground/charging-only UI expectations)",
    "Which workloads must be truly end-to-end encrypted vs acceptable within trusted network scope",
    "Do you want network-level shared credit pool enabled by default or per-member balances default",
    "Do you want a self-hosted coordinator option in the first year or keep cloud-only initially"
  ]
}