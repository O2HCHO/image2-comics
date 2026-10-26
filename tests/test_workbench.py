from dataclasses import replace
import base64
from io import BytesIO
import json
from pathlib import Path
import queue
import threading

from PIL import Image
import pytest
import requests

from comic_color.api import ApiConfig, ImageApi, image_value
from comic_color.config import EDIT_MODE, JSON_MODE, atomic_json, protect, save_settings, load_settings
from comic_color.imaging import scan_images, png_bytes, fingerprint
from comic_color.jobs import BatchWorker, JobConfig, TaskStore, output_lock


@pytest.fixture
def picture():
    stream = BytesIO()
    Image.new("RGB", (40, 60), "#ab8923").save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def source(tmp_path, picture):
    root = tmp_path / "原稿 空格"
    root.mkdir()
    path = root / "第2页.png"
    path.write_bytes(picture)
    return path


def config():
    return ApiConfig("https://test.invalid/v1/images/edits", "sk-test-secret", "image-model", "preserve text")


def response(data, status=200):
    value = requests.Response()
    value.status_code = status
    value._content = json.dumps(data).encode()
    value.headers["Content-Type"] = "application/json"
    return value


class FakeApi:
    def __init__(self, picture):
        self.picture = picture
        self.calls = 0

    def color(self, source, config):
        self.calls += 1
        return self.picture


def run_job(source, output, api, cfg=None, **kw):
    events = queue.Queue()
    job = BatchWorker(JobConfig(source.parent, output, (source,), cfg or config(), delay=0, **kw), events, api)
    job.run()
    return list(events.queue)


def test_recursive_scan_order_and_exclusion(tmp_path, picture):
    for name in ("chapter10/page1.jpg", "chapter2/page10.png", "chapter2/page2.png", "上色/generated.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(picture)
    paths = scan_images(tmp_path, tmp_path / "上色")
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == ["chapter2/page2.png", "chapter2/page10.png", "chapter10/page1.jpg"]
    with pytest.raises(ValueError):
        scan_images(tmp_path, tmp_path)


def test_multipart_references_order_and_fields(source, picture, monkeypatch):
    ref = source.with_name("参考.png")
    ref.write_bytes(picture)
    def post(url, **kw):
        assert [field for field, _ in kw["files"]] == ["image[]", "image[]"]
        assert kw["files"][0][1][0] == source.name
        assert kw["files"][1][1][0] == ref.name
        assert "第一张图片" in kw["data"]["prompt"]
        assert "denoising_strength" not in kw["data"]
        assert kw["allow_redirects"] is False
        return response({"data": [{"b64_json": base64.b64encode(picture).decode()}]})
    monkeypatch.setattr(requests, "post", post)
    assert Image.open(BytesIO(ImageApi().color(source, replace(config(), references=(ref,))))).format == "PNG"


def test_single_image_and_custom_json(source, picture, monkeypatch):
    calls = []
    def post(url, **kw):
        calls.append(kw)
        return response({"images": [base64.b64encode(picture).decode()]})
    monkeypatch.setattr(requests, "post", post)
    ImageApi().color(source, config())
    assert calls[-1]["files"][0][0] == "image"
    ImageApi().color(source, replace(config(), mode=JSON_MODE))
    assert base64.b64decode(calls[-1]["json"]["image"]) == picture
    assert calls[-1]["json"]["denoising_strength"] == .55
    with pytest.raises(ValueError, match="参考图"):
        ImageApi().color(source, replace(config(), mode=JSON_MODE, references=(source,)))


def test_url_download_never_receives_api_key(source, picture, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: response({"data": [{"url": "https://cdn.invalid/result.png?signature=foo"}]}))
    def get(url, **kw):
        assert "headers" not in kw and "auth" not in kw
        result = response({})
        result._content = picture
        return result
    monkeypatch.setattr(requests, "get", get)
    assert ImageApi().color(source, config())


@pytest.mark.parametrize("data", [{"image": "data:image/png;base64,eA=="}, {"data": {"images": [{"b64_json": "eA=="}]}}, {"data": [{"b64_json": "eA=="}]}])
def test_response_shapes(data):
    assert image_value(data)[0] == "base64"


@pytest.mark.parametrize("mode", ["timeout", "bad-image", "http", "html"])
def test_api_errors_and_no_automatic_retry(mode, source, monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        if mode == "timeout":
            raise requests.Timeout()
        if mode == "http":
            return response({"error": {"message": "sk-test-secret failed"}}, 429)
        if mode == "html":
            value = response({})
            value._content = b"<html>docs</html>"
            return value
        return response({"image": base64.b64encode(b"not-image").decode()})
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ValueError) as error:
        ImageApi().color(source, config())
    assert config().key not in str(error.value)
    assert len(calls) == 1


def test_resume_changed_parameters_and_lost_output(source, tmp_path, picture):
    api = FakeApi(picture)
    output = tmp_path / "results"
    run_job(source, output, api)
    first = TaskStore(output)
    record = next(iter(first.entries.values()))
    original_output = first.output_path(record)
    run_job(source, output, api)
    assert api.calls == 1
    run_job(source, output, api, replace(config(), prompt="different"))
    assert api.calls == 2 and original_output.exists()
    original_output.unlink()
    run_job(source, output, api)
    assert api.calls == 3
    source.write_bytes(png_bytes(picture, "1:1（透明补边）"))
    run_job(source, output, api)
    assert api.calls == 4
    manifest = (output / ".comic-color/tasks.json").read_text(encoding="utf-8")
    assert config().key not in manifest


def test_corrupt_result_reprocessed(source, tmp_path, picture):
    output, api = tmp_path / "results", FakeApi(picture)
    run_job(source, output, api)
    store = TaskStore(output)
    store.output_path(next(iter(store.entries.values()))).write_bytes(b"broken")
    run_job(source, output, api)
    assert api.calls == 2


def test_uncertain_requests_require_explicit_retry(source, tmp_path, picture):
    output, api = tmp_path / "results", FakeApi(picture)
    run_job(source, output, api)
    store = TaskStore(output)
    record = next(iter(store.entries.values()))
    record["status"] = "running"
    store.save()
    run_job(source, output, api)
    assert api.calls == 1
    assert next(iter(TaskStore(output).entries.values()))["status"] == "uncertain"
    run_job(source, output, api, allow_uncertain=True)
    assert api.calls == 2


def test_failed_retry_and_extension_collision(source, tmp_path, picture):
    output = tmp_path / "results"
    class Flaky(FakeApi):
        def color(self, source, cfg):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("failure")
            return picture
    api = Flaky(picture)
    run_job(source, output, api)
    assert next(iter(TaskStore(output).entries.values()))["status"] == "failed"
    run_job(source, output, api)
    assert next(iter(TaskStore(output).entries.values()))["status"] == "success"
    other = source.with_suffix(".jpg")
    other.write_bytes(picture)
    run_job(other, output, api)
    assert len(list(output.glob("*.png"))) == 2


def test_stop_waits_for_inflight_save(source, tmp_path, picture):
    entered, release = threading.Event(), threading.Event()
    class BlockingApi(FakeApi):
        def color(self, *args):
            entered.set()
            assert release.wait(5)
            return picture
    events = queue.Queue()
    output = tmp_path / "results"
    worker = BatchWorker(JobConfig(source.parent, output, (source, source), config(), delay=0), events, BlockingApi(picture))
    worker.start()
    assert entered.wait(5)
    worker.stop()
    assert worker.is_alive()
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert next(iter(TaskStore(output).entries.values()))["status"] == "success"
    assert len(list(output.glob("*.png"))) == 1


def test_pause_before_next_request(source, tmp_path, picture):
    api = FakeApi(picture)
    worker = BatchWorker(JobConfig(source.parent, tmp_path / "out", (source,), config(), delay=0), queue.Queue(), api)
    worker.pause()
    worker.start()
    assert not worker.resume_event.is_set() and api.calls == 0
    worker.resume()
    worker.join(5)
    assert api.calls == 1


def test_output_lock_and_corrupt_manifest(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(ValueError, match="另一个"):
            with output_lock(tmp_path):
                pass
    path = tmp_path / ".comic-color/tasks.json"
    path.parent.mkdir()
    path.write_text("invalid")
    with pytest.raises(ValueError, match="损坏"):
        TaskStore(tmp_path)


def test_padding_keeps_original_pixels(picture):
    result = Image.open(BytesIO(png_bytes(picture, "1:1（透明补边）")))
    assert result.size == (60, 60)
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((10, 0))[:3] == (171, 137, 35)


def test_dpapi_and_settings_do_not_store_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    value = "private-secret-中文"
    assert protect(protect(value.encode()), decrypt=True).decode() == value
    save_settings({"api_key": value, "remember_api_key": True})
    assert value not in (tmp_path / "Image2Comics/settings.json").read_text()
    assert load_settings()["api_key"] == value
    save_settings({"api_key": value, "remember_api_key": False})
    assert "api_key" not in load_settings()


def test_legacy_migrates_only_nonsensitive_settings(tmp_path, monkeypatch):
    import comic_color.config as module
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "profile"))
    monkeypatch.setattr(module, "__file__", str(tmp_path / "comic_color/config.py"))
    atomic_json(tmp_path / "local_settings.json", {"api_key": "legacy-secret", "remember_api_key": True,
                                                  "api_url": config().url, "model": "legacy-model"})
    result = load_settings()
    assert "api_key" not in result
    assert not result["remember_api_key"]
    assert result["model"] == "legacy-model"
    assert "legacy-secret" in (tmp_path / "local_settings.json").read_text()


def test_manifest_cannot_write_outside_output(tmp_path):
    store = TaskStore(tmp_path / "out")
    with pytest.raises(ValueError, match="超出"):
        store.output_path({"output": "../../outside.png"})


def test_reference_changes_create_new_variant(source, tmp_path, picture):
    reference = source.with_name("reference.png")
    reference.write_bytes(picture)
    api, output = FakeApi(picture), tmp_path / "out"
    cfg = replace(config(), references=(reference,))
    run_job(source, output, api, cfg)
    run_job(source, output, api, cfg)
    assert api.calls == 1
    reference.write_bytes(png_bytes(picture, "1:1（透明补边）"))
    run_job(source, output, api, cfg)
    assert api.calls == 2


def test_input_and_output_cannot_overlap_destructively(source, picture):
    events = run_job(source, source.parent, FakeApi(picture))
    assert any(kind == "error" for kind, _ in events)
    assert source.read_bytes() == picture
