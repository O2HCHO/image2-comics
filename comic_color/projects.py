"""Portable project metadata, directory ownership and isolated state."""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .config import atomic_json, data_dir
from .imaging import fingerprint, scan_images

EDIT_SETTINGS = {'prompt', 'references', 'recursive', 'preserve_text', 'denoising_strength', 'output_ratio'}


def read_document(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('无效的项目数据')
        return value
    except (OSError, ValueError) as error:
        raise ValueError(f'无法读取项目数据：{path}') from error


class Project:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.folder = self.path.parent
        self.data = read_document(self.path)
        if self.data.get('version') != 1 or not self.data.get('id'):
            raise ValueError('不支持的项目格式')
        self.id = self.data['id']

    def resolve(self, value):
        return (self.folder / value).resolve()

    def encode(self, path):
        try:
            return Path(os.path.relpath(Path(path).resolve(), self.folder)).as_posix()
        except ValueError:  # A different Windows drive.
            return str(Path(path).resolve())

    @property
    def input(self):
        return self.resolve(self.data['input'])

    @property
    def output(self):
        return self.resolve(self.data['output'])

    @property
    def state(self):
        return self.folder / '.comic-color'

    def save(self):
        atomic_json(self.path, self.data)

    def claim(self):
        if not self.input.is_dir() or not self.output.is_dir():
            raise ValueError('项目目录不可用，需要重新定位；不会自动创建替代目录')
        if self.output == self.input or self.input.is_relative_to(self.output):
            raise ValueError('输出目录必须独立于输入目录')
        from .jobs import output_lock
        with output_lock(self.output):
            marker = self.output / '.comic-color' / 'project-owner.json'
            if marker.exists() and read_document(marker).get('id') != self.id:
                raise ValueError('输出目录已属于另一个项目，请选择独立输出目录')
            atomic_json(marker, {'id': self.id, 'project': str(self.path)})

    @classmethod
    def create(cls, folder, source, output, settings=None):
        folder, source, output = map(lambda p: Path(p).resolve(), (folder, source, output))
        if (folder/'project.json').exists():
            raise ValueError('此目录已有项目，请直接打开')
        if not source.is_dir() or output == source or source.is_relative_to(output):
            raise ValueError('请选择有效且独立的输入、输出目录')
        output.mkdir(parents=True, exist_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
        # Claim before writing metadata so an ownership conflict leaves no half-project.
        marker = output/'.comic-color/project-owner.json'
        if marker.exists():
            raise ValueError('输出目录已由项目接管')
        atomic_json(folder/'project.json', {'version': 1, 'id': uuid.uuid4().hex,
                    'name': folder.name, 'input': str(source), 'output': str(output),
                    'settings': {k:v for k,v in (settings or {}).items() if k in EDIT_SETTINGS}, 'workspace': {}})
        project = cls(folder/'project.json')
        project.data.update(input=project.encode(source), output=project.encode(output))
        project.claim()
        project.import_legacy()
        project.save()
        return project

    def import_legacy(self):
        destination = self.state/'tasks.json'
        if destination.exists():
            return
        from .jobs import TaskStore
        old = TaskStore(self.output)
        root = str(self.input)
        data = {'version': 1, 'entries': {k:copy.deepcopy(v) for k,v in old.entries.items() if v.get('root') == root}}
        for field in ('active_outputs', 'original_pages'):
            data[field] = {}
            for key,value in old.data.get(field, {}).items():
                try:
                    if json.loads(key)[0] == root:
                        data[field][key] = copy.deepcopy(value)
                except (ValueError, TypeError, IndexError):
                    continue
        atomic_json(destination,data)
        # Preserve manual corrections; stale automatic masks rebuild lazily.
        from .text_protection import cache_path
        for record in data['entries'].values():
            source = self.input / record['source']
            old_cache = cache_path(self.output,source)
            new_cache = cache_path(self.folder,source)
            if old_cache.is_file() and not new_cache.exists():
                value = read_document(old_cache)
                value['algorithm'] = 'imported-manual-edits'
                atomic_json(new_cache,value)

    def index(self):
        path = self.state/'pages.json'
        return read_document(path).get('pages', []) if path.exists() else []

    def scan(self, recursive=True):
        started = time.perf_counter()
        files = scan_images(self.input,self.output,recursive)
        # Project state must never be picked up as input images.
        files = [p for p in files if not p.is_relative_to(self.state)]
        old = {p['path']:p for p in self.index()}
        pages = []
        for path in files:
            rel = path.relative_to(self.input).as_posix()
            stat = path.stat()
            item = {'path':rel,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            if all(old.get(rel,{}).get(k) == item[k] for k in ('size','mtime_ns')):
                item.update({k:v for k,v in old[rel].items() if k == 'sha256'})
            pages.append(item)
        atomic_json(self.state/'pages.json',{'version':1,'pages':pages})
        return files, time.perf_counter()-started

    def relocate(self, source, output):
        source,output = Path(source).resolve(),Path(output).resolve()
        if not source.is_dir() or not output.is_dir():
            raise ValueError('重新定位必须选择已存在的目录')
        from .jobs import TaskStore
        store = TaskStore(self.output,self)
        for record in store.entries.values():
            # Validate known content before changing any references.
            for root,relative,expected in ((source,record.get('source'),record.get('source_hash')),
                                           (output,record.get('output'),record.get('output_hash')),
                                           (output,record.get('raw_output'),record.get('raw_hash'))):
                if expected and relative and not relative.startswith('project://'):
                    path = (root/relative).resolve()
                    if not path.is_relative_to(root) or not path.is_file() or fingerprint(path) != expected:
                        raise ValueError('重新定位的图片与项目记录不一致：'+relative)
        old_roots = {r.get('root',str(self.input)) for r in store.entries.values()} or {str(self.input)}
        previous = dict(self.data)
        self.data.update(input=self.encode(source),output=self.encode(output))
        try:
            self.claim()
        except Exception:
            self.data = previous
            raise
        from .text_protection import cache_path
        for record in store.entries.values():
            old_cache = cache_path(self.folder,Path(record.get('root',str(self.input)))/record['source'])
            new_cache = cache_path(self.folder,source/record['source'])
            if old_cache != new_cache and old_cache.is_file():
                value = read_document(old_cache)
                value['algorithm'] = 'relocated-manual-edits'
                atomic_json(new_cache,value)
            record['root'] = str(source)
        for field in ('active_outputs','original_pages'):
            updated = {}
            for key,value in store.data.get(field,{}).items():
                root,rel = json.loads(key)
                updated[store.page_key(source,rel) if root in old_roots else key] = value
            store.data[field] = updated
        store.save()
        self.save()


def recent_projects(project=None):
    path = data_dir()/'projects.json'
    items = read_document(path).get('recent',[]) if path.exists() else []
    if project:
        items = [p for p in items if p.get('path') != str(project.path)]
        items.insert(0,{'name':project.data['name'],'path':str(project.path),
                        'opened':datetime.now(timezone.utc).isoformat()})
        items = items[:20]
        atomic_json(path,{'recent':items})
    return items


class ScopedEvents:
    """Capture the workspace generation when dispatching background work."""
    def __init__(self, queue, scope):
        self.queue, self.scope = queue, scope

    def put(self, event):
        kind,payload = event
        self.queue.put((kind,{**payload,'_scope':self.scope}))

    def get_nowait(self):
        return self.queue.get_nowait()
