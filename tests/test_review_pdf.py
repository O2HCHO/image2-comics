from dataclasses import replace
from io import BytesIO
from pathlib import Path
import queue

from PIL import Image
from pypdf import PdfReader
import pytest

from comic_color.api import ApiConfig
from comic_color.jobs import BatchWorker, JobConfig, TaskStore
from comic_color.pdf_export import export_pdf
from comic_color.review import result_path, export_pages
from comic_color.imaging import scan_images


def picture(color, size=(80, 120)):
    stream = BytesIO()
    Image.new("RGBA", size, color).save(stream, "PNG")
    return stream.getvalue()


class FakeApi:
    def __init__(self):
        self.calls = []
        self.fail = False

    def color(self, path, api):
        self.calls.append((path, api.prompt))
        if self.fail:
            raise ValueError("revision failed")
        return picture("red" if api.prompt == "red" else "blue")


def run(root, output, files, api, prompt="blue", force=False):
    cfg = ApiConfig("https://test.invalid/v1/images/edits", "key", "model", prompt)
    events = queue.Queue()
    BatchWorker(JobConfig(root, output, tuple(files), cfg, delay=0, force=force), events, api).run()
    assert not any(kind == "error" for kind, payload in events.queue)
    return TaskStore(output)


def test_single_page_revision_updates_only_its_active_result(tmp_path):
    root, output = tmp_path / "输入", tmp_path / "输出"
    root.mkdir()
    files = [root / "page2.png", root / "page10.png"]
    for path in files:
        path.write_bytes(picture("white"))
    api = FakeApi()
    store = run(root, output, files, api)
    before = store.current(root)
    store = run(root, output, [files[0]], api, "red", force=True)
    after = store.current(root)
    assert api.calls[-1] == (files[0], "red")
    assert after[files[0].name]["display_output"] != before[files[0].name]["display_output"]
    assert after[files[1].name]["display_output"] == before[files[1].name]["display_output"]
    assert store.output_path({"output": before[files[0].name]["display_output"]}).exists()
    store = run(root, output, [files[0]], api, "red", force=True)
    assert len(api.calls) == 4, "Explicit regeneration must bypass resume skipping"
    assert len(store.entries) == 4
    api.fail = True
    previous = store.current(root)[files[0].name]["display_output"]
    store = run(root, output, [files[0]], api, "failed prompt", force=True)
    failed = store.current(root)[files[0].name]
    assert failed["status"] == "failed" and failed["display_output"] == previous
    assert result_path(root, output, files[0], failed).exists()
    pages, missing = export_pages(root, output, tuple(files), store.current(root))
    assert len(pages) == 2 and not missing


def test_old_manifest_success_still_displays_after_failed_revision(tmp_path):
    root, output = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    source = root / "1.png"
    source.write_bytes(picture("white"))
    api = FakeApi()
    store = run(root, output, [source], api)
    store.data.pop("active_outputs")
    store.save()
    old_output = store.current(root)[source.name]["display_output"]
    api.fail = True
    store = run(root, output, [source], api, force=True)
    assert store.current(root)[source.name]["display_output"] == old_output


def test_legacy_output_mapping_and_missing_pages(tmp_path):
    root, output = tmp_path / "in", tmp_path / "out"
    (root / "chapter2").mkdir(parents=True)
    (output / "chapter2").mkdir(parents=True)
    for name in ("page10.jpg", "page2.jpg", "page3.jpg"):
        (root / "chapter2" / name).write_bytes(picture("white"))
    (output / "chapter2/colored_page10.png").write_bytes(picture("blue"))
    (output / "chapter2/colored_page2.png").write_bytes(picture("red"))
    files = tuple(scan_images(root, output))
    pages, missing = export_pages(root, output, files, {})
    assert [p.name for p in pages] == ["colored_page2.png", "colored_page10.png"]
    assert missing == ["chapter2/page3.jpg"]


def test_export_validates_current_outputs_and_originals(tmp_path):
    root, output = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    source = root / "1.png"
    source.write_bytes(picture("white"))
    store = run(root, output, [source], FakeApi())
    records = store.current(root)
    result = result_path(root, output, source, records[source.name])
    result.write_bytes(picture("red"))
    with pytest.raises(ValueError, match="外部修改"):
        export_pages(root, output, (source,), records)
    store = run(root, output, [source], FakeApi(), force=True)
    source.write_bytes(picture("black"))
    with pytest.raises(ValueError, match="原稿已改变"):
        export_pages(root, output, (source,), store.current(root))


def test_pdf_page_order_aspect_transparency_and_atomic_failure(tmp_path):
    first, second = tmp_path / "page2.png", tmp_path / "page10.png"
    first.write_bytes(picture("red", (80, 120)))
    second.write_bytes(picture((0, 0, 0, 0), (120, 80)))
    pdf = tmp_path / "上色.pdf"
    assert export_pdf([first, second], pdf) == 2
    reader = PdfReader(pdf)
    assert len(reader.pages) == 2
    assert (float(reader.pages[0].mediabox.width), float(reader.pages[0].mediabox.height)) == (40, 60)
    assert (float(reader.pages[1].mediabox.width), float(reader.pages[1].mediabox.height)) == (60, 40)
    assert reader.pages[0].images[0].image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    assert reader.pages[1].images[0].image.convert("RGB").getpixel((0, 0)) == (255, 255, 255)
    original_pdf = pdf.read_bytes()
    second.write_bytes(b"broken")
    with pytest.raises(OSError):
        export_pdf([first, second], pdf)
    assert pdf.read_bytes() == original_pdf
    assert not list(tmp_path.glob(".comic-pdf-*.tmp"))


def test_pdf_does_not_allow_image_overwrite(tmp_path):
    page = tmp_path / "1.png"
    page.write_bytes(picture("red"))
    with pytest.raises(ValueError):
        export_pdf([page], page)
    with pytest.raises(ValueError):
        export_pdf([], tmp_path / "empty.pdf")


def test_legacy_same_stem_different_extensions_are_not_guessed(tmp_path):
    root, output = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    output.mkdir()
    for suffix in ("png", "jpg"):
        (root / f"1.{suffix}").write_bytes(picture("white"))
    (output / "colored_1.png").write_bytes(picture("red"))
    assert result_path(root, output, root / "1.png", {}) is None
    assert result_path(root, output, root / "1.jpg", {}) is None


def test_resuming_old_batch_does_not_revert_adopted_revision(tmp_path):
    root, output = tmp_path / "in", tmp_path / "out"
    root.mkdir()
    source = root / "1.png"
    source.write_bytes(picture("white"))
    api = FakeApi()
    run(root, output, [source], api)
    store = run(root, output, [source], api, "red", force=True)
    chosen = store.current(root)[source.name]["display_output"]
    store = run(root, output, [source], api)
    assert len(api.calls) == 2
    assert store.current(root)[source.name]["display_output"] == chosen
    assert store.current(root)[source.name]["display_prompt"] == "red"
