# MeshNet Project Setup

## Project Structure

```
meshnet/
├── Cargo.toml              # Workspace configuration
├── plan.md                 # Product specification (original JSON)
├── IMPLEMENTATION.md       # Detailed implementation roadmap
├── SETUP.md               # This file
├── libs/                  # Utility libraries from Carcass
│   ├── README.md          # Library documentation and attribution
│   ├── cyn/               # Error chaining framework
│   ├── ust/               # Terminal styling utilities
│   ├── dup/               # Cheap clone trait
│   ├── ranged/            # Span and size types
│   └── cab-util/          # Expression language utilities
└── (future directories)
    ├── agent/             # Desktop and mobile agent
    ├── relay/             # Relay server
    └── control-plane/     # Control plane (might be separate repo)
```

## What Was Copied from Carcass

We've copied 5 utility libraries from the Carcass project (https://github.com/cull-os/carcass):

### 1. cyn - Error Chaining Framework ⭐ **CRITICAL**
- **Files:** `libs/cyn/` (2 files)
- **Purpose:** Production-grade error handling with context chains
- **License:** MPL-2.0
- **Why:** Best-in-class Rust error handling. Shows full error path instead of just final error.
- **Usage in MeshNet:** All agent, relay, and control plane code will use cyn for error handling

### 2. ust - Universal Styling
- **Files:** `libs/ust/` (8 files)
- **Purpose:** Terminal color codes and formatting
- **License:** MPL-2.0
- **Why:** Used by cyn for pretty error output. Also useful for CLI tools.
- **Usage in MeshNet:** CLI applications, error rendering

### 3. dup - Cheap Clone Trait
- **Files:** `libs/dup/` (5 files, including macros/)
- **Purpose:** Distinguishes expensive clones from cheap clones (Arc/Rc)
- **License:** MPL-2.0
- **Why:** Used by cyn for efficient error chains
- **Usage in MeshNet:** Error handling infrastructure

### 4. ranged - Span and Size Types
- **Files:** `libs/ranged/` (4 files)
- **Purpose:** Text position tracking
- **License:** MPL-2.0
- **Why:** Dependency of ust
- **Usage in MeshNet:** Indirect dependency

### 5. cab-util - Expression Language Utilities
- **Files:** `libs/cab-util/` (4 files)
- **Purpose:** Helper utilities for string manipulation
- **License:** MPL-2.0
- **Why:** Dependency of ust
- **Usage in MeshNet:** Indirect dependency

## License Compliance

All libraries in `libs/` are licensed under **MPL-2.0** (Mozilla Public License 2.0).

### Key Points:
- ✅ We can use these libraries in MeshNet (even in proprietary software)
- ✅ We can modify the libraries if needed
- ⚠️ Any modifications to MPL-licensed files must remain MPL-licensed
- ✅ MeshNet code that uses these libraries can be under any license we choose

See `libs/README.md` for full license compliance details.

## Next Steps

### 1. Verify Workspace Builds
```bash
cd meshnet
cargo check
```

This will verify that all workspace members compile and dependencies are correct.

### 2. Create Agent Skeleton
```bash
cargo new agent --lib
```

Then add to `Cargo.toml`:
```toml
[workspace]
members = [
    # ... existing libs ...
    "agent",
]
```

### 3. Create Relay Server Skeleton
```bash
cargo new relay --bin
```

Then add to `Cargo.toml`:
```toml
[workspace]
members = [
    # ... existing libs ...
    "relay",
]
```

### 4. Start Phase 0 Implementation
Follow the checklist in `IMPLEMENTATION.md`:
- Week 1-2: Critical validation prototypes (iOS background test, relay benchmark, mTLS test)
- Week 3-4: Core networking layer (mesh swarm setup, job protocol handler)
- Week 5-6: Device identity and job execution
- Week 7-8: Integration and testing

## Development Environment

### Required Tools
- **Rust:** 1.75+ (install via https://rustup.rs)
- **Cargo:** Comes with Rust
- **Git:** For version control

### Optional Tools (for mobile development)
- **Xcode:** For iOS development (macOS only)
- **Android Studio:** For Android development
- **Cargo-bundle:** For cross-platform builds (`cargo install cargo-bundle`)

### IDE Recommendations
- **VS Code** with rust-analyzer extension
- **RustRover** (JetBrains)
- **Vim/Neovim** with rust.vim and coc-rust-analyzer

## Running Tests
```bash
# Test all workspace members
cargo test --workspace

# Test specific library
cargo test -p cyn

# Run with verbose output
cargo test --workspace -- --nocapture
```

## Building
```bash
# Build all workspace members (debug)
cargo build --workspace

# Build release
cargo build --workspace --release

# Build specific member
cargo build -p relay --release
```

## Checking Code Quality
```bash
# Format code
cargo fmt --all

# Lint code
cargo clippy --workspace -- -D warnings

# Check without building
cargo check --workspace
```

## References

- **MeshNet Documentation:** `IMPLEMENTATION.md` (phase-by-phase checklist)
- **Product Spec:** `plan.md` (original JSON specification)
- **Utility Libraries:** `libs/README.md` (attribution and usage)
- **Carcass Project:** https://github.com/cull-os/carcass

## Troubleshooting

### Workspace Dependency Errors
If you get errors about missing workspace dependencies:
```bash
# Clean and rebuild
cargo clean
cargo build --workspace
```

### Missing Dependencies
If you get errors about missing crates:
```bash
# Update Cargo.lock
cargo update
```

### Path Dependencies Not Found
Ensure all `path = "../libs/xxx"` references in `Cargo.toml` files are correct relative to the workspace root.

---

**Status:** Initial setup complete. Ready to begin Phase 0 implementation (Week 1-2: Critical Validation Prototypes).
