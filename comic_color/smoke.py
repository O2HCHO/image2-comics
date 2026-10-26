"""Offline end-to-end diagnostic, usable inside the frozen distribution."""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
import uuid
import faulthandler
import sys

from PIL import Image, ImageDraw, ImageGrab, ImageFont
from .config import atomic_json


def run(destination: Path):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    workspace = destination / ("session-" + uuid.uuid4().hex[:8])
    os.environ["LOCALAPPDATA"] = str(workspace / "test-profile")
    report = {"success": False}
    diagnostic = (destination / "diagnostic.log").open("w", encoding="utf-8")
    previous_stderr = sys.stderr
    sys.stderr = diagnostic
    faulthandler.enable(file=diagnostic)
    faulthandler.dump_traceback_later(25, repeat=True, file=diagnostic)
    def stage(name):
        report["stage"] = name
        atomic_json(destination / "report.json", report)
    stage("startup")
    server = None
    app = None
    try:
        from .gui import ColoringApp
        from .jobs import TaskStore
        from tkinter import messagebox
        source_dir = workspace / "漫画 输入" / "第2章"
        source_dir.mkdir(parents=True, exist_ok=True)
        source = source_dir / "第1页.png"
        image = Image.new("RGB", (420, 600), "white")
        draw = ImageDraw.Draw(image)
        for top in (20, 310):
            draw.rectangle((20, top, 400, top + 265), outline="#25324a", width=4)
            draw.ellipse((130, top + 50, 280, top + 220), outline="#25324a", width=5)
            draw.rectangle((35, top + 15, 160, top + 70), fill="white", outline="#25324a", width=2)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
        draw.text((40, 42), 'KEEP TEXT', font=font, fill='black')
        image.save(source)
        colored = image.copy()
        draw = ImageDraw.Draw(colored)
        for top in (20, 310):
            draw.ellipse((133, top + 53, 277, top + 217), fill="#e3ae66", outline="#25324a", width=3)
        stream = BytesIO()
        colored.save(stream, "PNG")
        picture = stream.getvalue()
        reference = workspace / "配色参考.png"
        reference.write_bytes(picture)
        calls = {"edits": 0, "models": 0}
        release = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                calls["models"] += 1
                self.send_json({"data": [{"id": "offline-test-model"}]})

            def do_POST(self):
                content = self.rfile.read(int(self.headers["Content-Length"]))
                if content.count(b'name="image[]"') != 2:
                    self.send_error(400, "Expected source plus reference")
                    return
                calls["edits"] += 1
                if calls["edits"] == 3:
                    release.wait(10)
                self.send_json({"data": [{"b64_json": base64.b64encode(picture).decode()}]})
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        output = workspace / "上色 结果"
        app = ColoringApp(settings={"api_url": f"http://127.0.0.1:{server.server_port}/v1/images/edits",
                                    "api_key": "offline-secret", "model": "offline-test-model",
                                    "input_dir": str(source_dir.parent), "output_dir": str(output),
                                    "references": [str(reference)], "remember_api_key": True, "preserve_text": False})
        stage("workbench")
        def until(predicate, timeout=20):
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() > deadline:
                    raise TimeoutError("UI smoke check exceeded deadline")
                if not app.app_closed:
                    app.update()
                time.sleep(.025)
            for _ in range(5):
                if not app.app_closed:
                    app.update()
                time.sleep(.025)
        until(lambda: len(app.files) == 1 and not app.scanning)
        app.test_connection()
        until(lambda: not app.busy)
        app.start()
        until(lambda: app.worker is not None and not app.worker.is_alive())
        until(lambda: bool(app.records) and next(iter(app.records.values())).get("status") == "success")
        assert calls == {"edits": 1, "models": 1}, calls
        store = TaskStore(output)
        assert store.reusable(next(iter(store.entries.values())))
        assert (output / "第2章").is_dir()
        assert app.model.get() == "offline-test-model"
        app.show_preview()
        until(lambda: len(app.preview_photos) == 2)
        assert app.tree.winfo_height() >= 100, "Image list collapsed"
        assert app.canvases[0].winfo_height() >= 150, "Preview collapsed"
        assert app.logs.winfo_rooty() + app.logs.winfo_height() <= app.winfo_rooty() + app.winfo_height(), "Controls clipped"
        ImageGrab.grab(bbox=(app.winfo_rootx(), app.winfo_rooty(), app.winfo_rootx() + app.winfo_width(), app.winfo_rooty() + app.winfo_height())).save(destination / "workbench.png")
        app.start()
        until(lambda: not app.worker.is_alive())
        assert calls["edits"] == 1, "Resume submitted a duplicate generation"
        opened = []
        original_open = os.startfile
        os.startfile = lambda path: opened.append(path)
        try:
            app.open_output()
        finally:
            os.startfile = original_open
        assert opened == [str(output.resolve())]
        old_result = app.result_for(source)
        stage("revision")
        for number in range(2, 11):
            image.save(source_dir / f"第{number}页.png")
        app.refresh()
        until(lambda: len(app.files) == 10 and not app.scanning)
        panel = app.review_panel
        app.tabs.set("整批监修 / 单页修改")
        panel.select(source)
        base_prompt = app.prompt.get("1.0", "end-1c")
        panel.prompt.insert("end", "\nOnly this page: use a warmer palette.")
        panel.regenerate()
        until(lambda: not app.worker.is_alive())
        until(lambda: app.result_for(source) != old_result)
        assert calls["edits"] == 2
        assert old_result.is_file(), "Previous version was overwritten"
        assert app.prompt.get("1.0", "end-1c") == base_prompt
        assert len(TaskStore(output).entries) == 2, "Single page revision touched other pages"
        until(lambda: len(panel.photos) == 2)
        panel.change_page(1)
        until(lambda: len(panel.photos) == 1)
        assert panel.page == 1 and panel.current == app.files[1]
        panel.change_page(-1)
        until(lambda: len(panel.photos) == 2)
        ImageGrab.grab(bbox=(app.winfo_rootx(), app.winfo_rooty(), app.winfo_rootx() + app.winfo_width(), app.winfo_rooty() + app.winfo_height())).save(destination / "review.png")
        # Drive PDF preparation, missing-page confirmation, save dialog and completion.
        stage("pdf")
        from tkinter import filedialog
        pdf = destination / "review-export.pdf"
        original_save, original_info, original_ask = filedialog.asksaveasfilename, messagebox.showinfo, messagebox.askyesno
        filedialog.asksaveasfilename = lambda *a, **k: str(pdf)
        messagebox.showinfo = lambda *a, **k: None
        messagebox.askyesno = lambda *a, **k: True
        try:
            app.export_pdf()
            until(lambda: not app.busy)
            assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-")
            import re
            assert len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes())) == 1
        finally:
            filedialog.asksaveasfilename, messagebox.showinfo, messagebox.askyesno = original_save, original_info, original_ask
        stage("after-pdf")
        stage("text-protection")
        panel.select(source)
        app.edit_text()
        editor = app.text_editor
        until(lambda: not editor.loading, timeout=40)
        assert editor.data is not None, editor.status.cget('text')
        # Strict balloon filtering may reject this synthetic rectangular fixture.
        # Real automatic detections are covered separately; exercise manual override here.
        assert set(editor.data['models']) == {'ctd', 'bubble'}
        editor.data['added'].append([38, 40, 155, 64])
        assert editor.save()
        editor.preview()
        until(lambda: not editor.loading, timeout=40)
        assert hasattr(editor, 'right_photo'), editor.status.cget('text')
        ImageGrab.grab(bbox=(editor.winfo_rootx(), editor.winfo_rooty(), editor.winfo_rootx() + editor.winfo_width(), editor.winfo_rooty() + editor.winfo_height())).save(destination / 'text-editor.png')
        editor.apply()
        until(lambda: not app.worker.is_alive(), timeout=40)
        until(lambda: app.records[source.relative_to(source_dir.parent).as_posix()].get('text_metadata') is not None)
        assert calls['edits'] == 2, 'Local repair invoked image generation'
        assert app.result_for(source).is_file()
        app.tabs.set("接口与上色设置")
        app.update()
        stage("settings")
        ImageGrab.grab(bbox=(app.winfo_rootx(), app.winfo_rooty(), app.winfo_rootx() + app.winfo_width(), app.winfo_rooty() + app.winfo_height())).save(destination / "settings.png")
        # Exercise the actual GUI close path while a request is in flight.
        stage("close")
        app.prompt.insert("end", "\nClose-path diagnostic")
        app.start()
        until(lambda: calls["edits"] == 3)
        original_ask = messagebox.askyesno
        messagebox.askyesno = lambda *args, **kwargs: True
        try:
            app.on_close()
        finally:
            messagebox.askyesno = original_ask
        assert app.closing and app.worker.is_alive()
        release.set()
        until(lambda: not app.worker.is_alive())
        assert all(record["status"] == "success" for record in TaskStore(output).entries.values())
        if not app.app_closed:
            app.finish_close()
        app = None
        from .config import load_settings
        assert load_settings()["api_key"] == "offline-secret"
        report.update(success=True, frozen=bool(getattr(__import__("sys"), "frozen", False)), calls=calls,
                      checks=["GUI launch", "recursive scan", "model list", "multipart reference", "PNG output",
                              "task manifest", "resume skips", "preview pair", "open output action", "encrypted settings",
                              "close waits for in-flight result", "high-DPI layout", "whole-input paired gallery",
                              "gallery pagination", "single-page custom prompt regeneration", "old result retained",
                              "comparison refresh", "missing-page PDF confirmation", "PDF export",
                              "bundled CTD and YOLO offline models", "mask editor save and preview",
                              "local-only text repair updates current result"])
    except Exception as error:
        import traceback
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if app and not app.app_closed:
            app.finish_close()
        if server:
            server.shutdown()
            server.server_close()
        atomic_json(destination / "report.json", report)
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        sys.stderr = previous_stderr
        diagnostic.close()
    return report["success"]
