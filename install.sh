#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Print header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  Meshnet Installation Script          ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if running from meshnet directory
if [ ! -f "Cargo.toml" ] || [ ! -d "agent" ]; then
    error "Please run this script from the meshnet repository root directory"
    exit 1
fi

# Check for Rust
info "Checking for Rust installation..."
if ! command -v cargo &> /dev/null; then
    error "Rust is not installed. Please install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
success "Rust found: $(rustc --version)"

# Build release binaries
info "Building meshnet binaries (this may take a few minutes)..."
cargo build --release

# Detect target directory
TARGET_DIR="target/release"
if [ -d "target/aarch64-apple-darwin/release" ]; then
    TARGET_DIR="target/aarch64-apple-darwin/release"
elif [ -d "target/x86_64-apple-darwin/release" ]; then
    TARGET_DIR="target/x86_64-apple-darwin/release"
elif [ -d "target/x86_64-unknown-linux-gnu/release" ]; then
    TARGET_DIR="target/x86_64-unknown-linux-gnu/release"
elif [ -d "target/aarch64-unknown-linux-gnu/release" ]; then
    TARGET_DIR="target/aarch64-unknown-linux-gnu/release"
fi

# Check for binaries
if [ ! -f "$TARGET_DIR/agent" ]; then
    error "Build failed - agent binary not found in $TARGET_DIR"
    exit 1
fi
if [ ! -f "$TARGET_DIR/relay-server" ]; then
    error "Build failed - relay-server binary not found in $TARGET_DIR"
    exit 1
fi
if [ ! -f "$TARGET_DIR/control-plane" ]; then
    error "Build failed - control-plane binary not found in $TARGET_DIR"
    exit 1
fi
success "Build complete: $TARGET_DIR"

# Determine installation directory
INSTALL_DIR="$HOME/.local/bin"
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'

# Create installation directory if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    info "Creating installation directory: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
fi

# Install binaries
info "Installing binaries to $INSTALL_DIR..."
cp "$TARGET_DIR/agent" "$INSTALL_DIR/mesh"
cp "$TARGET_DIR/relay-server" "$INSTALL_DIR/mesh-relay"
cp "$TARGET_DIR/control-plane" "$INSTALL_DIR/mesh-control-plane"
chmod +x "$INSTALL_DIR/mesh"
chmod +x "$INSTALL_DIR/mesh-relay"
chmod +x "$INSTALL_DIR/mesh-control-plane"
success "Binaries installed:"
success "  - mesh (agent CLI)"
success "  - mesh-relay (relay server)"
success "  - mesh-control-plane (control plane)"

# Add to PATH if needed
add_to_path() {
    local shell_rc="$1"

    if [ ! -f "$shell_rc" ]; then
        info "Creating shell config: $shell_rc"
        touch "$shell_rc"
    fi

    if ! grep -q '\.local/bin' "$shell_rc"; then
        info "Adding $INSTALL_DIR to PATH in $shell_rc"
        echo "" >> "$shell_rc"
        echo "# Meshnet installation" >> "$shell_rc"
        echo "$PATH_LINE" >> "$shell_rc"
        success "Updated $shell_rc"
        return 0
    else
        info "$shell_rc already contains .local/bin in PATH"
        return 1
    fi
}

# Check current shell and update appropriate RC files
UPDATED=0
CURRENT_SHELL=$(basename "$SHELL")

case "$CURRENT_SHELL" in
    zsh)
        add_to_path "$HOME/.zshrc" && UPDATED=1 || true
        add_to_path "$HOME/.zprofile" && UPDATED=1 || true
        add_to_path "$HOME/.profile" && UPDATED=1 || true
        ;;
    bash)
        add_to_path "$HOME/.bash_profile" && UPDATED=1 || true
        add_to_path "$HOME/.bashrc" && UPDATED=1 || true
        add_to_path "$HOME/.profile" && UPDATED=1 || true
        ;;
    *)
        warn "Unknown shell: $CURRENT_SHELL"
        add_to_path "$HOME/.profile" && UPDATED=1 || true
        ;;
esac

case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
        export PATH="$INSTALL_DIR:$PATH"
        info "Added $INSTALL_DIR to PATH for this installer session"
        ;;
esac

# Verify installation
echo ""
info "Verifying installation..."

if [ -x "$INSTALL_DIR/mesh" ] && [ -x "$INSTALL_DIR/mesh-control-plane" ] && [ -x "$INSTALL_DIR/mesh-relay" ]; then
    success "Installed binaries are executable"
else
    error "One or more installed binaries are not executable"
    exit 1
fi

# Print success message
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Installation Complete! 🎉             ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "The 'mesh' command has been installed to: $INSTALL_DIR/mesh"
echo "The 'mesh-control-plane' command has been installed to: $INSTALL_DIR/mesh-control-plane"
echo "The 'mesh-relay' command has been installed to: $INSTALL_DIR/mesh-relay"
echo ""

if [ $UPDATED -eq 1 ]; then
    warn "PATH was updated. Please restart your shell or run:"
    echo "  source ~/.$CURRENT_SHELL"rc
    echo ""
fi

echo "Quick start:"
echo "  1. Initialize device:   mesh device init --network-id test --name \"My Device\""
echo "  2. Create a pool:       mesh pool create --name \"My Pool\""
echo "  3. Join a ring:         mesh ring join --model-id tinyllama-1.1b"
echo "  4. Start agent:         mesh device start"
echo ""
echo "Device automation scripts:"
echo "  ./device1.sh  - Start Device 1 (admin) - runs relay, control plane, and agent"
echo "  ./device2.sh  - Start Device 2 (member) - joins pool and runs agent"
echo ""

# Test if commands are immediately available
if command -v mesh &> /dev/null && command -v mesh-control-plane &> /dev/null && command -v mesh-relay &> /dev/null; then
    success "Mesh commands are ready to use!"
    echo ""
    mesh --version 2>/dev/null || true
    mesh-control-plane --version 2>/dev/null || true
    mesh-relay --version 2>/dev/null || true
else
    warn "Mesh commands are not yet all visible in your PATH"
    echo "Please restart your shell or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
