"""Content-addressed, checked, bounded on-disk intermediate results."""
import hashlib
import json
import logging
from pathlib import Path
import time

from .config import atomic_bytes, atomic_json


class StageCache:
    def __init__(self, root):
        self.root = Path(root)/'.comic-color/fusion-cache' if root else None

    def key(self, stage, inputs):
        return hashlib.sha256(json.dumps([stage,inputs],sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def load(self, key):
        if self.root is None:
            return None
        try:
            folder = self.root/key
            manifest = json.loads((folder/'index.json').read_text(encoding='utf-8'))
            blobs = {}
            for name,item in manifest['files'].items():
                if Path(name).name != name:
                    return None
                content = (folder/name).read_bytes()
                if hashlib.sha256(content).hexdigest() != item:
                    return None
                blobs[name] = content
            logging.getLogger(__name__).info('cache hit %s',key[:12])
            return blobs,manifest['metadata']
        except (OSError,ValueError,KeyError,TypeError):
            return None

    def save(self, key, blobs, metadata):
        if self.root is None:
            return
        folder = self.root/key
        try:
            for name,content in blobs.items():
                atomic_bytes(folder/name,content)
            atomic_json(folder/'index.json', {'metadata':metadata,'files':{name:hashlib.sha256(content).hexdigest() for name,content in blobs.items()}})
            # Intermediates are disposable. Keep a bounded 2 GiB / 120 entries.
            entries = sorted((p for p in self.root.iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime,reverse=True)
            total = 0
            for i,p in enumerate(entries):
                total += sum(f.stat().st_size for f in p.iterdir() if f.is_file())
                if p != folder and (i >= 120 or total > 2*1024**3):
                    import shutil
                    shutil.rmtree(p)
        except OSError:
            logging.getLogger(__name__).exception('Could not persist optional fusion cache')


class Timing:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self,*args):
        logging.getLogger(__name__).info('%s %.3fs',self.name,time.perf_counter()-self.started)
