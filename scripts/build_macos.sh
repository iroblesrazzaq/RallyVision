#!/bin/bash
# Build script for RallyClip macOS App Bundle
# 
# Prerequisites:
#   - Python 3.10+ with uv or pip
#   - All RallyClip dependencies installed
#
# Usage:
#   ./scripts/build_macos.sh
#
# Output:
#   dist/RallyClip.app - The standalone macOS application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== RallyClip macOS Build ==="
echo "Project root: $PROJECT_ROOT"

# Check for required models
echo ""
echo "Checking for required models..."
MISSING_MODELS=0

if [ ! -f "models/lstm_300_v0.1.pth" ]; then
    echo "  WARNING: models/lstm_300_v0.1.pth not found"
    MISSING_MODELS=1
fi

if [ ! -f "models/scaler_300_v0.1.joblib" ]; then
    echo "  WARNING: models/scaler_300_v0.1.joblib not found"
    MISSING_MODELS=1
fi

if [ $MISSING_MODELS -eq 1 ]; then
    echo ""
    echo "Some models are missing. The app will fail at runtime without them."
    echo "See models/README.md for download instructions."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if pyinstaller is installed
echo ""
echo "Checking PyInstaller installation..."
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    if command -v uv &> /dev/null; then
        uv pip install pyinstaller
    else
        pip install pyinstaller
    fi
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/RallyClip
rm -rf dist/RallyClip
rm -rf dist/RallyClip.app

# Run PyInstaller
echo ""
echo "Building RallyClip.app..."
echo "This may take several minutes due to PyTorch and other large dependencies."
echo ""

# Use uv run if available, otherwise fall back to direct pyinstaller
if command -v uv &> /dev/null; then
    uv run pyinstaller apps/gui/RallyClip.spec --noconfirm
else
    python -m PyInstaller apps/gui/RallyClip.spec --noconfirm
fi

# Check if build succeeded
if [ -d "dist/RallyClip.app" ]; then
    echo ""
    echo "=== Build Successful ==="
    echo ""
    echo "Output: dist/RallyClip.app"
    echo ""
    echo "To test: open dist/RallyClip.app"
    echo ""
    
    # Show approximate size
    SIZE=$(du -sh "dist/RallyClip.app" | cut -f1)
    echo "App bundle size: $SIZE"
    echo ""
    echo "All outputs will be saved to: ~/RallyClip/"
    echo ""
    echo "To distribute:"
    echo "  1. Drag RallyClip.app to Applications folder"
    echo "  2. Or create a DMG: hdiutil create -volname RallyClip -srcfolder dist/RallyClip.app -ov dist/RallyClip.dmg"
else
    echo ""
    echo "=== Build Failed ==="
    echo "Check the output above for errors."
    exit 1
fi

