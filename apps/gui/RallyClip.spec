# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for RallyClip macOS App Bundle
# Build with: pyinstaller apps/gui/RallyClip.spec

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

block_cipher = None

# Resolve paths relative to this spec file
SPEC_DIR = Path(SPECPATH).resolve()
PROJECT_ROOT = SPEC_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
FRONTEND_DIR = SPEC_DIR / "frontend"
MODELS_DIR = PROJECT_ROOT / "models"

# Entry point
ENTRY_SCRIPT = SRC_DIR / "gui" / "app.py"

# Collect hidden imports for ML libraries
hiddenimports = []

# PyTorch and related
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')

# Ultralytics (YOLO)
hiddenimports += collect_submodules('ultralytics')

# OpenCV
hiddenimports += ['cv2']

# scikit-learn
hiddenimports += collect_submodules('sklearn')

# Other dependencies
hiddenimports += [
    'joblib',
    'numpy',
    'scipy',
    'scipy.ndimage',
    'flask',
    'flask_cors',
    'werkzeug',
    'av',
    'tqdm',
    'PIL',
    'PIL.Image',
]

# pywebview for native window
hiddenimports += collect_submodules('webview')

# Collect data files
datas = []

# Frontend web UI
datas.append((str(FRONTEND_DIR), 'apps/gui/frontend'))

# ML models (LSTM, scaler, YOLO weights)
if MODELS_DIR.exists():
    datas.append((str(MODELS_DIR), 'models'))

# Ultralytics data files (YOLO configs, etc.)
try:
    ultralytics_datas = collect_data_files('ultralytics')
    datas += ultralytics_datas
except Exception:
    pass

# torch data files
try:
    torch_datas = collect_data_files('torch')
    datas += torch_datas
except Exception:
    pass

# Add src directory to path for imports
pathex = [str(SRC_DIR), str(PROJECT_ROOT)]

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RallyClip',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window - GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RallyClip',
)

# macOS App Bundle
app = BUNDLE(
    coll,
    name='RallyClip.app',
    icon=None,  # TODO: Add app icon (RallyClip.icns)
    bundle_identifier='com.rallyclip.app',
    info_plist={
        'CFBundleName': 'RallyClip',
        'CFBundleDisplayName': 'RallyClip',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
    },
)

