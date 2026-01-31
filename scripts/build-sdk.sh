#!/bin/bash
# Build TypeScript SDK package
#
# Usage: ./scripts/build-sdk.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SDK_DIR="$ROOT_DIR/packages/sdk"

echo "Building Delta LTSC SDK..."
echo "SDK dir: $SDK_DIR"

cd "$SDK_DIR"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build ESM
echo ""
echo "Building ESM..."
npx tsc -p tsconfig.esm.json

# Build type declarations
echo ""
echo "Building type declarations..."
npx tsc -p tsconfig.types.json

# Copy WASM files to dist
echo ""
echo "Copying WASM files to dist..."
mkdir -p dist/esm/wasm
cp -r src/wasm/*.wasm dist/esm/wasm/ 2>/dev/null || echo "No WASM files found (run build-wasm.sh first)"
cp -r src/wasm/pkg/*.wasm dist/esm/wasm/ 2>/dev/null || true

# Copy JSON dictionaries to dist
echo "Copying dictionaries..."
mkdir -p dist/esm/dictionaries
cp -r src/dictionaries/*.json dist/esm/dictionaries/ 2>/dev/null || true

echo ""
echo "SDK build complete!"
echo "Output: $SDK_DIR/dist/"
