"""Local-only repair workers. No API calls or credentials."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import uuid

from .config import ORIGINAL, atomic_bytes
from .imaging import fingerprint
from .jobs import TaskStore, output_lock
from .review import result_path
from .text_protection import VERSION, detection, fuse


def repair_page(root: Path, output: Path, source: Path, store: TaskStore, progress=None) -> dict:
    root, output, source = root.resolve(), output.resolve(), source.resolve()
    rel = source.relative_to(root).as_posix()
    previous = store.current(root).get(rel, {})
    if store.original_selected(root, source):
        return previous
    record = {"root": str(root), "source": rel, "source_hash": fingerprint(source),
              "parameters": dict(previous.get("parameters", {})), "output": "",
              "status": "needs_review", "updated": datetime.now(timezone.utc).isoformat()}
    identity = uuid.uuid4().hex
    store.entries[identity] = record
    try:
        if previous.get("raw_output"):
            if previous.get("source_hash") != record["source_hash"]:
                raise ValueError("原稿已改变，请重新生成上色图；旧原稿的生成图不能复用")
            raw = store.output_path({"output": previous["raw_output"]})
            if fingerprint(raw) != previous.get("raw_hash"):
                raise ValueError("保存的原始生成图校验失败")
        else:
            if previous.get("text_metadata") or previous.get("text_mask"):
                raise ValueError("此页只有旧融合图且缺少原始生成图，无法可靠恢复白框底色；请恢复原始生成图")
            raw = result_path(root, output, source, previous)
            if raw is None:
                raise ValueError("没有可供文字修复的生成图")
            expected = previous.get("display_hash") or previous.get("output_hash")
            if expected and fingerprint(raw) != expected:
                raise ValueError("已有上色图被修改，请核对后再处理")
            expected_source = previous.get("display_source_hash") or previous.get("source_hash")
            if expected_source and expected_source != record["source_hash"]:
                raise ValueError("原稿已改变，不能修复旧原稿的上色图")
        raw_content = raw.read_bytes()
        raw_ref, raw_copy = store.asset('raw/' + identity + '.png')
        atomic_bytes(raw_copy, raw_content)
        record.update(raw_output=raw_ref, raw_hash=fingerprint(raw_copy))
        store.save()
        if progress:
            progress('准备当前页原稿文字（有效缓存优先）')
        data = detection(source, store.cache_root)
        record["parameters"]["text_regions"] = data
        record["parameters"]["text_protection"] = {"enabled": True, "algorithm": VERSION}
        record["text_fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        if progress:
            content, mask, info = fuse(source, raw_content, data, record['parameters'].get('ratio', ORIGINAL),progress=progress)
        else:
            content, mask, info = fuse(source, raw_content, data, record['parameters'].get('ratio', ORIGINAL))
        if fingerprint(source) != record["source_hash"]:
            raise ValueError("处理期间原稿改变，结果未采用，请重新检测")
        destination = Path(rel).parent / (source.name[:100] + ".text-" + identity[:12] + ".png")
        mask_ref, mask_path = store.asset('masks/' + identity + '.png')
        cleanup_ref, cleanup_path = store.asset('masks/' + identity + '-cleanup.png')
        atomic_bytes(cleanup_path, info.pop('cleanup_png'))
        info.update(cleanup_mask=cleanup_ref, cleanup_hash=fingerprint(cleanup_path))
        atomic_bytes(mask_path, mask)
        atomic_bytes(output / destination, content)
        record.update(status="success", output=destination.as_posix(), output_hash=fingerprint(output / destination),
                      error="", text_mask=mask_ref, text_metadata=info)
        store.activate(root, record)
    except Exception as error:
        record.update(status="needs_review", error=str(error))
    record["updated"] = datetime.now(timezone.utc).isoformat()
    store.save()
    return store.display_record(root, record)


class RepairWorker(threading.Thread):
    def __init__(self, root, output, files, events, project=None):
        super().__init__(daemon=False)
        self.root, self.output, self.files, self.events = root, output, files, events
        self.project = project
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()

    def stop(self):
        self.stop_event.set()
        self.resume_event.set()

    def pause(self):
        self.resume_event.clear()

    def resume(self):
        self.resume_event.set()

    def run(self):
        summary = {"success": 0, "failed": 0, "skipped": 0, "uncertain": 0}
        try:
            with output_lock(self.output):
                store = TaskStore(self.output, self.project)
                store.recover()
                for i, source in enumerate(self.files):
                    while not self.resume_event.wait(.1) and not self.stop_event.is_set():
                        pass
                    if self.stop_event.is_set():
                        break
                    record = repair_page(self.root, self.output, source, store,
                                         progress=lambda message:self.events.put(('stage',{'message':message})))
                    summary["success" if record["status"] == "success" else "uncertain"] += 1
                    self.events.put(("item", {"source": record["source"], "status": record["status"], "record": record}))
                    self.events.put(("progress", {"index": i + 1, "total": len(self.files), "summary": dict(summary)}))
        except Exception as error:
            self.events.put(("error", {"message": str(error)}))
        finally:
            self.events.put(("finished", {"summary": summary, "stopped": self.stop_event.is_set()}))
