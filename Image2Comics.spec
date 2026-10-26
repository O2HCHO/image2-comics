# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
from PyInstaller.utils.hooks.tcl_tk import tcltk_info

tcl_tk_data = [(source, os.path.dirname(destination)) for destination, source, _ in tcltk_info.data_files]
analysis = Analysis(['app.py'], pathex=[], binaries=collect_dynamic_libs('onnxruntime'),
                    datas=collect_data_files('customtkinter') + collect_data_files('certifi') + tcl_tk_data + [('assets/models', 'assets/models'), ('assets/app.ico', 'assets'), ('THIRD_PARTY_NOTICES.txt', '.')],
                    hiddenimports=[], hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=['pytest', 'torch', 'tensorflow'], noarchive=False)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name='Image2Comics', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          icon='assets/app.ico')
collection = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name='Image2Comics')
