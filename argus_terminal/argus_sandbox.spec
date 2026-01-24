# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\ingester_ops\\argus\\argus_terminal\\argus_terminal\\__main__.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\ingester_ops\\argus\\argus_terminal\\argus_terminal\\themes', 'argus_terminal/themes')],
    hiddenimports=['textual', 'rich', 'argus_terminal', 'argus_terminal.app', 'argus_terminal.screens', 'argus_terminal.widgets', 'argus_terminal.utils'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='argus_sandbox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
