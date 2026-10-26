"""Reproducible folder-mode packaging. Run with the project virtualenv."""
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile
import certifi
from comic_color import __version__
from comic_color.comic_models import model_manifest

root = Path(__file__).resolve().parent
model_manifest()
subprocess.run([sys.executable, 'tools/make_icon.py'], cwd=root, check=True)
test_temp = root / 'build' / ('tests-' + uuid.uuid4().hex[:8])
subprocess.run([sys.executable, '-m', 'pytest', '-q', '--basetemp', str(test_temp)], cwd=root, check=True)
build_output = root / 'build' / f'pyinstaller-{__version__}'
dist_output = root / 'build' / f'dist-{__version__}'
subprocess.run([
    sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
    '--distpath', str(dist_output), '--workpath', str(build_output),
    'Image2Comics.spec'
], cwd=root, check=True)
folder = dist_output / 'Image2Comics'
shutil.copy2(root / 'QUICKSTART.txt', folder / '使用说明.txt')
shutil.copy2(root / 'THIRD_PARTY_NOTICES.txt', folder / 'THIRD_PARTY_NOTICES.txt')
shutil.copytree(root / 'licenses', folder / 'licenses', dirs_exist_ok=True)
shutil.copy2(root / 'LICENSE', folder / 'LICENSE')
certifi_data = folder / '_internal' / 'certifi'
certifi_data.mkdir(parents=True, exist_ok=True)
shutil.copy2(certifi.where(), certifi_data / 'cacert.pem')
# Allowlist excludes user configuration, credentials, images and build environments.
(root / 'dist').mkdir(parents=True, exist_ok=True)
source_archive = root / f'dist/Image2Comics-{__version__}-source.zip'
with zipfile.ZipFile(source_archive, 'w', zipfile.ZIP_DEFLATED) as source:
    for name in ('app.py','build.py','Image2Comics.spec','README.md','QUICKSTART.txt','THIRD_PARTY_NOTICES.txt','LICENSE','requirements.txt','requirements-dev.txt','requirements-models.txt','environment.yml'):
        source.write(root / name, name)
    for directory in ('comic_color','tests','licenses','assets'):
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                source.write(path,path.relative_to(root))
    for name in ('prepare_comic_models.py','download_comic_assets.py','check_comic_models.py','make_icon.py'):
        source.write(root / 'tools' / name,'tools/'+name)
shutil.copy2(source_archive,folder / source_archive.name)
archive = root / f'dist/Image2Comics-{__version__}-Windows-x64.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            package.write(path, path.relative_to(folder.parent))
print(archive)
