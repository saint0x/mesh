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

# Build release binary
info "Building meshnet agent (this may take a few minutes)..."
cargo build --release --bin agent

if [ ! -f "target/release/agent" ]; then
    error "Build failed - binary not found at target/release/agent"
    exit 1
fi
success "Build complete"

# Determine installation directory
INSTALL_DIR="$HOME/.local/bin"

# Create installation directory if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    info "Creating installation directory: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
fi

# Install binary
info "Installing binary to $INSTALL_DIR/mesh..."
cp target/release/agent "$INSTALL_DIR/mesh"
chmod +x "$INSTALL_DIR/mesh"
success "Binary installed"

# Add to PATH if needed
add_to_path() {
    local shell_rc="$1"
    local path_line="export PATH=\"\$HOME/.local/bin:\$PATH\""

    if [ -f "$shell_rc" ]; then
        if ! grep -q ".local/bin" "$shell_rc"; then
            info "Adding $INSTALL_DIR to PATH in $shell_rc"
            echo "" >> "$shell_rc"
            echo "# Meshnet installation" >> "$shell_rc"
            echo "$path_line" >> "$shell_rc"
            success "Updated $shell_rc"
            return 0
        else
            info "$shell_rc already contains .local/bin in PATH"
            return 1
        fi
    fi
    return 1
}

# Check current shell and update appropriate RC file
UPDATED=0
CURRENT_SHELL=$(basename "$SHELL")

case "$CURRENT_SHELL" in
    zsh)
        if add_to_path "$HOME/.zshrc"; then
            UPDATED=1
        fi
        ;;
    bash)
        if add_to_path "$HOME/.bash_profile"; then
            UPDATED=1
        elif add_to_path "$HOME/.bashrc"; then
            UPDATED=1
        fi
        ;;
    *)
        warn "Unknown shell: $CURRENT_SHELL"
        ;;
esac

# Verify installation
echo ""
info "Verifying installation..."

if [ -x "$INSTALL_DIR/mesh" ]; then
    success "Binary is executable"
else
    error "Binary is not executable"
    exit 1
fi

# Print success message
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Installation Complete! 🎉             ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "The 'mesh' command has been installed to: $INSTALL_DIR/mesh"
echo ""

if [ $UPDATED -eq 1 ]; then
    warn "PATH was updated. Please restart your shell or run:"
    echo "  source ~/.$CURRENT_SHELL"rc
    echo ""
fi

echo "Quick start:"
echo "  1. Initialize device:   mesh init --network-id test --name \"My Device\""
echo "  2. Create a pool:       mesh pool-create --name \"My Pool\""
echo "  3. Start agent:         mesh start"
echo ""
echo "For LAN discovery testing, see: LAN_DISCOVERY_TEST.md"
echo ""

# Test if mesh is immediately available
if command -v mesh &> /dev/null; then
    success "The 'mesh' command is ready to use!"
    echo ""
    mesh --version 2>/dev/null || true
else
    warn "The 'mesh' command is not yet in your PATH"
    echo "Please restart your shell or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
