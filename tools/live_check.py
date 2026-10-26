"""Explicit one-page integration check using legacy credentials in memory only."""
from __future__ import annotations

from pathlib import Path
import queue
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comic_color.api import ApiConfig, safe_error
from comic_color.config import DEFAULT_PROMPT, EDIT_MODE, ORIGINAL, atomic_json, read_json
from comic_color.imaging import valid_image, EXTENSIONS
from comic_color.jobs import BatchWorker, JobConfig, TaskStore


def main():
    root = Path(__file__).resolve().parent.parent
    legacy = read_json(root / "local_settings.json")
    candidates = sorted(path for path in (root / "inputs/Girls' Last Tour").iterdir() if path.suffix.lower() in EXTENSIONS)
    source = candidates[1] if len(candidates) > 1 else candidates[0]
    reference = root / "outputs/Girls' Last Tour/colored_001.png"
    destination = root / "outputs/integration-check"
    report = {"success": False, "source": str(source), "reference": str(reference)}
    try:
        api = ApiConfig(legacy["api_url"], legacy["api_key"], legacy["model"], DEFAULT_PROMPT,
                        EDIT_MODE, ratio=ORIGINAL, references=(reference,))
        api.validate()
        events = queue.Queue()
        worker = BatchWorker(JobConfig(source.parent, destination, (source,), api, delay=0), events)
        worker.run()
        store = TaskStore(destination)
        record = next(iter(store.latest(source.parent).values()))
        if record.get("status") != "success":
            raise ValueError(record.get("error", "生成未成功"))
        result = store.output_path(record)
        assert valid_image(result)
        report.update(success=True, output=str(result), model=api.model)
    except Exception as error:
        report["error"] = safe_error(error, legacy.get("api_key", ""))
    atomic_json(destination / "report.json", report)
    print(__import__("json").dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
