#!/bin/bash
# Build WASM package from Rust source
#
# Usage: ./scripts/build-wasm.sh [--release|--debug]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CORE_DIR="$ROOT_DIR/packages/core"
SDK_DIR="$ROOT_DIR/packages/sdk"

BUILD_MODE="${1:-release}"

echo "Building Delta LTSC WASM core..."
echo "Mode: $BUILD_MODE"
echo "Core dir: $CORE_DIR"
echo "SDK dir: $SDK_DIR"

# Check for wasm-pack (try common locations)
WASM_PACK=""
if command -v wasm-pack &> /dev/null; then
    WASM_PACK="wasm-pack"
elif [ -f "$HOME/.cargo/bin/wasm-pack" ]; then
    WASM_PACK="$HOME/.cargo/bin/wasm-pack"
elif [ -d "$HOME/Library/Caches/.wasm-pack" ]; then
    # Find the most recent wasm-pack installation
    WASM_PACK=$(find "$HOME/Library/Caches/.wasm-pack" -name "wasm-pack" -type f 2>/dev/null | head -1)
fi

if [ -z "$WASM_PACK" ]; then
    echo "Error: wasm-pack is not installed."
    echo "Install with: cargo install wasm-pack"
    exit 1
fi

echo "Using wasm-pack: $WASM_PACK"

# Add cargo bin to PATH if needed
if [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check for Rust wasm target
if command -v rustup &> /dev/null; then
    if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
        echo "Installing wasm32-unknown-unknown target..."
        rustup target add wasm32-unknown-unknown
    fi
else
    echo "Warning: rustup not found - assuming wasm32-unknown-unknown target is installed"
fi

cd "$CORE_DIR"

# Strip emojis from output
strip_emoji() {
    LC_ALL=C sed 's/[^[:print:][:space:]]//g'
}

# Build for bundler target (used by most bundlers)
echo ""
echo "Building for bundler target..."
if [ "$BUILD_MODE" = "release" ]; then
    "$WASM_PACK" build --target bundler --release --out-dir "$SDK_DIR/src/wasm/pkg" 2>&1 | strip_emoji
else
    "$WASM_PACK" build --target bundler --dev --out-dir "$SDK_DIR/src/wasm/pkg" 2>&1 | strip_emoji
fi

# Build for nodejs target (for server-side use)
echo ""
echo "Building for Node.js target..."
if [ "$BUILD_MODE" = "release" ]; then
    "$WASM_PACK" build --target nodejs --release --out-dir "$SDK_DIR/src/wasm/pkg-node" 2>&1 | strip_emoji
else
    "$WASM_PACK" build --target nodejs --dev --out-dir "$SDK_DIR/src/wasm/pkg-node" 2>&1 | strip_emoji
fi

# Build for web target (for direct browser use)
echo ""
echo "Building for web target..."
if [ "$BUILD_MODE" = "release" ]; then
    "$WASM_PACK" build --target web --release --out-dir "$SDK_DIR/src/wasm/pkg-web" 2>&1 | strip_emoji
else
    "$WASM_PACK" build --target web --dev --out-dir "$SDK_DIR/src/wasm/pkg-web" 2>&1 | strip_emoji
fi

# Copy main WASM file to SDK src
echo ""
echo "Copying WASM files..."
cp "$SDK_DIR/src/wasm/pkg/delta_ltsc_core_bg.wasm" "$SDK_DIR/src/wasm/" 2>/dev/null || true

echo ""
echo "WASM build complete!"
echo ""
echo "Output directories:"
echo "  - Bundler: $SDK_DIR/src/wasm/pkg/"
echo "  - Node.js: $SDK_DIR/src/wasm/pkg-node/"
echo "  - Web:     $SDK_DIR/src/wasm/pkg-web/"
