from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import uuid

from .api import ApiConfig, ImageApi, safe_error
from .config import atomic_bytes, atomic_json
from .imaging import fingerprint, valid_image

LABELS = {"pending": "待处理", "running": "处理中", "success": "成功", "failed": "失败",
          "uncertain": "待确认", "skipped": "已跳过", "needs_review": "文字需核对"}


@contextmanager
def output_lock(output: Path):
    import msvcrt
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".comic-color.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise ValueError("另一个上色任务正在使用此输出目录，请稍后重试") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class TaskStore:
    def __init__(self, output: Path, project=None):
        self.output = output.resolve()
        self.project = project
        self.state = project.state if project else self.output / '.comic-color'
        self.cache_root = project.folder if project else self.output
        self.path = self.state / 'tasks.json'
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if self.data.get("version") != 1 or not isinstance(self.data.get("entries"), dict):
                    raise ValueError("任务清单版本不支持")
                if not all(isinstance(value, dict) for value in self.data["entries"].values()):
                    raise ValueError("任务清单条目无效")
            except (ValueError, AttributeError) as error:
                raise ValueError("任务清单损坏，已停止操作以保护已有结果；请更换输出目录或备份后移走 tasks.json") from error
        else:
            self.data = {"version": 1, "entries": {}}

    @property
    def entries(self) -> dict:
        return self.data["entries"]

    def save(self) -> None:
        atomic_json(self.path, self.data)

    def recover(self) -> None:
        changed = False
        for record in self.entries.values():
            if record.get("status") == "running":
                record["status"] = "uncertain"
                record["error"] = "上次请求被中断；服务可能已计费，请确认后手动重试"
                if record.get("raw_output"):
                    record["status"] = "needs_review"
                    record["error"] = "原始生成图已保存，可本地修复文字，无需再次生成"
                changed = True
        if changed:
            self.save()

    def output_path(self, record: dict) -> Path:
        if record['output'].startswith('project://'):
            if self.project is None:
                raise ValueError('项目资源缺少项目上下文')
            path = (self.state / record['output'][len('project://'):]).resolve()
            if not path.is_relative_to(self.state.resolve()):
                raise ValueError('项目资源路径超出状态目录')
            return path
        path = (self.output / record["output"]).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError("任务清单的输出路径超出输出目录")
        return path

    def asset(self, relative):
        """Return a persistent reference and safe location for private assets."""
        ref = 'project://' + str(relative).replace('\\','/') if self.project else '.comic-color/' + str(relative).replace('\\','/')
        return ref, self.output_path({'output':ref})

    def reusable(self, record: dict) -> bool:
        path = self.output_path(record)
        return (record.get("status") == "success" and path.is_file()
                and fingerprint(path) == record.get("output_hash") and valid_image(path))

    def latest(self, root: Path) -> dict[str, dict]:
        result = {}
        for record in self.entries.values():
            if record.get("root") == str(root.resolve()):
                rel = record["source"]
                if rel not in result or record.get("updated", "") >= result[rel].get("updated", ""):
                    result[rel] = dict(record)
        return result

    @staticmethod
    def page_key(root: Path, source: str) -> str:
        return json.dumps([str(root.resolve()), source], ensure_ascii=False)

    def activate(self, root: Path, record: dict) -> None:
        self.data.setdefault("active_outputs", {})[self.page_key(root, record["source"])] = {
            key: record[key] for key in ("output", "output_hash", "source_hash", "parameters")
        }

    def display_record(self, root: Path, record: dict) -> dict:
        result = dict(record)
        active = self.data.get("active_outputs", {}).get(self.page_key(root, record["source"]))
        if active is None:
            # Upgrade existing version-1 manifests without changing their entries.
            candidates = getattr(self, '_success_by_page', {}).get(record['source'])
            if candidates is None:
                candidates = [r for r in self.entries.values() if r.get("root") == str(root.resolve())
                              and r.get("source") == record["source"] and r.get("status") == "success"]
            active = max(candidates, key=lambda r: r.get("updated", ""), default=None)
        if active and active.get("output"):
            parameters = active.get("parameters")
            if parameters is None:
                parameters = next((r.get("parameters", {}) for r in self.entries.values()
                                   if r.get("root") == str(root.resolve()) and r.get("source") == record["source"]
                                   and r.get("output") == active["output"]), {})
            result.update(display_output=active["output"], display_hash=active.get("output_hash", ""),
                          display_source_hash=active.get("source_hash", ""), display_prompt=parameters.get("prompt", ""))
        return result

    def current(self, root: Path) -> dict[str, dict]:
        latest = self.latest(root)
        self._success_by_page = {rel:[] for rel in latest}
        for record in self.entries.values():
            if record.get('root') == str(root.resolve()) and record.get('status') == 'success':
                self._success_by_page.setdefault(record['source'],[]).append(record)
        try:
            return {rel: self.display_record(root, record) for rel, record in latest.items()}
        finally:
            del self._success_by_page

    def original_selected(self, root: Path, source: Path) -> bool:
        choice = self.data.get('original_pages', {}).get(self.page_key(root, source.resolve().relative_to(root.resolve()).as_posix()))
        return bool(choice and choice == fingerprint(source))

    def use_original(self, root: Path, source: Path) -> dict:
        root, source = root.resolve(), source.resolve()
        if self.output == root or root.is_relative_to(self.output) or source.is_relative_to(self.output):
            raise ValueError('输出目录必须独立于输入目录')
        if not valid_image(source):
            raise ValueError('原图损坏或无法解码')
        rel = source.relative_to(root).as_posix()
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if fingerprint(source) != digest:
            raise ValueError('原图正在变化，请重试')
        identity = uuid.uuid4().hex
        destination = Path(rel).parent / (source.stem[:100] + '.original-' + identity[:12] + source.suffix)
        atomic_bytes(self.output / destination, content)
        record = dict(root=str(root), source=rel, source_hash=digest, parameters={'use_original': True},
                      output=destination.as_posix(), output_hash=digest, status='success', error='',
                      updated=datetime.now(timezone.utc).isoformat())
        self.entries[identity] = record
        self.data.setdefault('original_pages', {})[self.page_key(root, rel)] = digest
        self.activate(root, record)
        self.save()
        return self.display_record(root, record)


@dataclass(frozen=True)
class JobConfig:
    input_dir: Path
    output_dir: Path
    files: tuple[Path, ...]
    api: ApiConfig
    delay: float = 2.0
    allow_uncertain: bool = False
    force: bool = False
    project: object = None


class BatchWorker(threading.Thread):
    def __init__(self, config: JobConfig, events: queue.Queue, api: ImageApi | None = None):
        super().__init__(daemon=False)
        self.config = config
        self.events = events
        self.api = api or ImageApi()
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()

    def emit(self, kind: str, **payload) -> None:
        self.events.put((kind, payload))

    def stop(self) -> None:
        self.stop_event.set()
        self.resume_event.set()

    def pause(self) -> None:
        self.resume_event.clear()

    def resume(self) -> None:
        self.resume_event.set()

    def checkpoint(self) -> bool:
        if not self.resume_event.is_set():
            self.emit("paused")
        while not self.resume_event.wait(0.1):
            if self.stop_event.is_set():
                return False
        return not self.stop_event.is_set()

    def run(self) -> None:
        summary = {"success": 0, "failed": 0, "skipped": 0, "uncertain": 0}
        try:
            self.config.api.validate()
            root = self.config.input_dir.resolve()
            output = self.config.output_dir.resolve()
            if output == root or root.is_relative_to(output):
                raise ValueError("输出目录必须独立于输入目录，不能等于输入目录或其上级目录")
            with output_lock(output):
                store = TaskStore(output, self.config.project)
                store.recover()
                reference_hashes = [(str(path.resolve()), fingerprint(path)) for path in self.config.api.references]
                params = {"url": self.config.api.url, "model": self.config.api.model,
                          "prompt": self.config.api.prompt, "mode": self.config.api.mode,
                          "denoise": self.config.api.denoise, "ratio": self.config.api.ratio,
                          "references": reference_hashes}
                if self.config.api.preserve_text:
                    from .text_protection import TextOptions
                    params["text_protection"] = TextOptions().values()
                # Credentials in the URL query are not persisted; validation forbids query/fragment.
                encoded = json.dumps(params, ensure_ascii=False, sort_keys=True)
                for index, source in enumerate(self.config.files):
                    if not self.checkpoint():
                        break
                    source = source.resolve()
                    rel = source.relative_to(root).as_posix()
                    if source.is_relative_to(output):
                        raise ValueError("不能处理输出目录内的图片，请刷新图片列表")
                    record = None
                    try:
                        source_hash = fingerprint(source)
                        if self.config.force:
                            store.data.get('original_pages', {}).pop(store.page_key(root, rel), None)
                            store.save()
                        if store.original_selected(root, source):
                            current = store.current(root).get(rel)
                            record = current if current and store.reusable(current) and current.get('parameters', {}).get('use_original') else store.use_original(root, source)
                            summary['skipped'] += 1
                            self.emit('item', source=rel, status='success', record=record)
                            self.emit('progress', index=index + 1, total=len(self.config.files), summary=dict(summary))
                            continue
                        page_params = dict(params)
                        text_data = None
                        if self.config.api.preserve_text:
                            from .text_protection import detection
                            self.emit('stage',message='准备当前页原稿文字（有效缓存优先）')
                            text_data = detection(source, store.cache_root)
                            page_params["text_regions"] = text_data
                        page_encoded = json.dumps(page_params, ensure_ascii=False, sort_keys=True)
                        identity = hashlib.sha256(json.dumps([str(root), rel, source_hash, page_encoded], ensure_ascii=False).encode("utf-8")).hexdigest()
                        if self.config.project:
                            identity = hashlib.sha256((self.config.project.id + identity).encode()).hexdigest()
                        if self.config.force:
                            identity = hashlib.sha256((identity + uuid.uuid4().hex).encode()).hexdigest()
                        record = store.entries.get(identity)
                        if record and record.get("status") == "needs_review":
                            summary["uncertain"] += 1
                            self.emit("item", source=rel, status="needs_review", record=store.display_record(root, record))
                            self.emit("progress", index=index + 1, total=len(self.config.files), summary=dict(summary))
                            continue
                        if record and store.reusable(record):
                            summary["skipped"] += 1
                            self.emit("item", source=rel, status="skipped", record=store.display_record(root, record))
                            self.emit("progress", index=index + 1, total=len(self.config.files), summary=dict(summary))
                            continue
                        if record and record.get("status") == "uncertain" and not self.config.allow_uncertain:
                            summary["uncertain"] += 1
                            self.emit("item", source=rel, status="uncertain", record=store.display_record(root, record))
                            self.emit("progress", index=index + 1, total=len(self.config.files), summary=dict(summary))
                            continue
                        # A unique variant name protects results produced by different parameters.
                        destination = Path(rel).parent / (source.name[:120] + ".colored-" + identity[:12] + ".png")
                        if (output / destination).exists():
                            destination = destination.with_name(destination.stem + "-" + uuid.uuid4().hex[:8] + ".png")
                        record = {"root": str(root), "source": rel, "source_hash": source_hash,
                                  "parameters": page_params, "output": destination.as_posix(), "status": "pending"}
                        store.entries[identity] = record
                        if not valid_image(source):
                            raise ValueError("输入图片损坏或无法解码")
                        for path, expected in reference_hashes:
                            if fingerprint(Path(path)) != expected or not valid_image(Path(path)):
                                raise ValueError("参考图已改变或损坏，请停止后重新选择")
                        record.update(status="running", error="", updated=datetime.now(timezone.utc).isoformat())
                        store.save()
                        self.emit("item", source=rel, status="running", record=store.display_record(root, record))
                        content = self.api.color(source, self.config.api)
                        if self.config.api.preserve_text:
                            raw, raw_path = store.asset('raw/' + identity + '.png')
                            atomic_bytes(raw_path, content)
                            record.update(raw_output=raw, raw_hash=fingerprint(raw_path))
                            store.save()
                            from .text_protection import fuse
                            content, mask, info = fuse(source, content, text_data, self.config.api.ratio,
                                                     progress=lambda message:self.emit('stage',message=message))
                            mask_ref, mask_path = store.asset('masks/' + identity + '.png')
                            cleanup_ref, cleanup_path = store.asset('masks/' + identity + '-cleanup.png')
                            atomic_bytes(cleanup_path, info.pop('cleanup_png'))
                            info['cleanup_mask'] = cleanup_ref
                            info['cleanup_hash'] = fingerprint(cleanup_path)
                            atomic_bytes(mask_path, mask)
                            record.update(text_metadata=info, text_mask=mask_ref)
                        if fingerprint(source) != source_hash:
                            raise ValueError("请求期间原图发生变化，结果未写入，请核对后重试")
                        path = store.output_path(record)
                        atomic_bytes(path, content)
                        record.update(status="success", output_hash=fingerprint(path), error="")
                        store.activate(root, record)
                        summary["success"] += 1
                    except Exception as error:
                        if record is None:
                            identity = hashlib.sha256(json.dumps([str(root), rel, encoded], ensure_ascii=False).encode("utf-8")).hexdigest()
                            record = {"root": str(root), "source": rel, "output": "", "parameters": params}
                            store.entries[identity] = record
                        needs_review = bool(record.get("raw_output"))
                        record.update(status="needs_review" if needs_review else "failed", error=safe_error(error, self.config.api.key))
                        summary["uncertain" if needs_review else "failed"] += 1
                    record["updated"] = datetime.now(timezone.utc).isoformat()
                    store.save()
                    self.emit("item", source=rel, status=record["status"], record=store.display_record(root, record))
                    self.emit("progress", index=index + 1, total=len(self.config.files), summary=dict(summary))
                    if index + 1 < len(self.config.files) and self.stop_event.wait(self.config.delay):
                        break
        except Exception as error:
            self.emit("error", message=safe_error(error, self.config.api.key))
        finally:
            self.emit("finished", summary=summary, stopped=self.stop_event.is_set())
