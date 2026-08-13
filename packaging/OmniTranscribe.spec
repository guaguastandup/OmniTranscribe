# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os
import sys

from PyInstaller.utils.hooks import collect_all

ROOT = Path(os.getcwd()).resolve()
SRC = ROOT / "src"

datas = []
binaries = []
hiddenimports = []

for package in ("gradio", "whisper", "webview"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden
    except Exception:
        pass

for source, destination in (
    (ROOT / "assets", "assets"),
    (ROOT / "prompt.md", "."),
    (ROOT / "ChillDuanSansVF.ttf", "."),
    (ROOT / "ChillHuoFangSong_Regular.otf", "."),
):
    if source.exists():
        datas.append((str(source), destination))

bin_dir = ROOT / "bin"
if bin_dir.exists():
    for binary in bin_dir.iterdir():
        if binary.is_file():
            binaries.append((str(binary), "bin"))

a = Analysis(
    [str(ROOT / "desktop_entry.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="OmniTranscribe", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False, disable_windowed_traceback=False,
    argv_emulation=False, target_arch=None, codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name="OmniTranscribe")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="OmniTranscribe.app",
        icon=None,
        bundle_identifier="io.github.guaguastandup.omnitranscribe",
        info_plist={
            "CFBundleName": "OmniTranscribe",
            "CFBundleDisplayName": "OmniTranscribe",
            "NSHighResolutionCapable": True,
        },
    )
