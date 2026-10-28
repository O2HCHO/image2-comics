"""Main-thread mask editing, background detection and fusion preview."""
import gc
import copy
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from .config import ORIGINAL, atomic_json
from .imaging import fingerprint
from .jobs import TaskStore, output_lock
from .text_protection import cache_path, detection, fuse, restoration_mask, regions, rgb


class TextEditor(ctk.CTkToplevel):
    def __init__(self, app, source: Path):
        super().__init__(app)
        self.app, self.source = app, source
        self.root, self.output = Path(app.input_var.get()).resolve(), Path(app.output_var.get()).resolve()
        self.project = app.project
        self.cache_root = self.project.folder if self.project else self.output
        self.data = None
        self.original = None
        self.load_failed = False
        self.photos = []
        self.events = queue.Queue()
        self.loading = True
        self.closed = False
        self.preview_revision = 0
        self.pending_close = False
        self.pending_preview = False
        self.verified_options = None
        self.error_details = ""
        self.right_image = None
        self.drag = None
        self.rect = None
        self.scale = 1.
        self.title("文字保护 · " + app.relative(source))
        self.geometry("1080x740")
        self.minsize(850, 600)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(self, text="仅自动处理封闭气泡/旁白框。拖框补选可处理框外文字；绿色为文字，蓝色为气泡，橙色为跳过候选。红色为实际掩码。", wraplength=950).grid(row=0, column=0, columnspan=2, padx=12, pady=10)
        bar = ctk.CTkFrame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)
        self.mode = tk.StringVar(value="补选区域")
        ctk.CTkSegmentedButton(bar, variable=self.mode, values=["补选区域", "取消区域"]).pack(side="left", padx=8, pady=8)
        ctk.CTkLabel(bar, text="外扩 px").pack(side="left", padx=4)
        self.expand = ctk.CTkEntry(bar, width=48)
        self.expand.insert(0, "1")
        self.expand.pack(side="left")
        ctk.CTkLabel(bar, text="羽化 px").pack(side="left", padx=4)
        self.feather = ctk.CTkEntry(bar, width=48)
        self.feather.insert(0, "1")
        self.feather.pack(side="left")
        self.empty = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(bar, text="确认无须保护文字", variable=self.empty).pack(side="left", padx=12)
        self.canvases = []
        for col in range(2):
            frame = ctk.CTkFrame(self)
            frame.grid(row=2, column=col, sticky="nsew", padx=8, pady=10)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(1, weight=1)
            label = ctk.CTkLabel(frame, text="原稿与保护掩码" if col == 0 else "点击预览融合，查看上色图与文字恢复结果")
            label.grid(row=0, column=0)
            if col == 1:
                self.right_label = label
            canvas = tk.Canvas(frame, bg="#e8edf4", highlightthickness=0)
            canvas.grid(row=1, column=0, sticky="nsew")
            self.canvases.append(canvas)
        self.canvases[0].bind("<ButtonPress-1>", self.press)
        self.canvases[0].bind("<B1-Motion>", self.motion)
        self.canvases[0].bind("<ButtonRelease-1>", self.release)
        self.canvases[0].bind("<Configure>", lambda _: self.draw())
        self.canvases[1].bind("<Configure>", lambda _: self.draw_right())
        footer = ctk.CTkFrame(self)
        footer.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=8)
        ctk.CTkButton(footer, text="预览融合", command=self.preview).pack(side="left", padx=6, pady=10)
        ctk.CTkButton(footer, text="保存区域", command=self.save).pack(side="left", padx=6)
        ctk.CTkButton(footer, text="保存并修复此页", command=self.apply).pack(side="left", padx=6)
        self.help_button = ctk.CTkButton(footer, text="查看原因与修复方法", command=self.show_error, state="disabled")
        self.help_button.pack(side="left", padx=6)
        self.status = ctk.CTkLabel(self, text="本机检测中…", wraplength=980, justify="left")
        self.status.grid(row=5, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))
        self.mask_mode = tk.StringVar(value="文字恢复")
        ctk.CTkSegmentedButton(self, variable=self.mask_mode, values=["文字恢复", "清字范围"], command=lambda _: self.draw()).grid(row=4, column=0, columnspan=2, pady=4)
        self.display_mask = None
        self.mask_cache = {}
        self.drawing_mask = False
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(80, self.poll)
        gc.collect()
        threading.Thread(target=self.load, daemon=True).start()

    def signature(self, data):
        import json
        return json.dumps(data, sort_keys=True)

    def show_error(self):
        if self.error_details:
            messagebox.showinfo("未完成文字恢复 · 原因与修复方法", self.error_details, parent=self)

    def draw_right(self):
        if self.right_image is None:
            return
        canvas = self.canvases[1]
        image = self.right_image.copy()
        image.thumbnail((max(1, canvas.winfo_width()-12), max(1, canvas.winfo_height()-12)))
        self.right_photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(0, 0, image=self.right_photo, anchor="nw")

    def load(self):
        try:
            with output_lock(self.output):
                data = detection(self.source, self.cache_root)
            self.events.put(("loaded", (data, rgb(self.source))))
        except Exception as error:
            self.events.put(("error", str(error)))

    def options(self):
        if self.data is None:
            raise ValueError("文字检测尚未完成")
        expand, feather = int(self.expand.get()), int(self.feather.get())
        if not 0 <= expand <= 30 or not 0 <= feather <= 12:
            raise ValueError("外扩范围 0–30，羽化范围 0–12 像素")
        return {**copy.deepcopy(self.data), "expand": expand, "feather": feather, "confirmed_empty": self.empty.get()}

    def draw(self):
        if self.original is None:
            return
        try:
            data = self.options()
        except ValueError:
            data = self.data
        image = self.original.copy()
        boxes = regions(data)
        import json
        key = json.dumps(data, sort_keys=True)
        if self.mask_mode.get() == "清字范围":
            mask = self.mask_cache.get((key, "cleanup"), np.zeros(image.shape[:2], np.uint8))
        else:
            mask = self.mask_cache.get((key, "restore"))
            if mask is None:
                mask = np.zeros(image.shape[:2], np.uint8)
                if not self.drawing_mask:
                    self.drawing_mask = True
                    def extract():
                        try:
                            self.events.put(("mask", (key, restoration_mask(self.original, data))))
                        except Exception as error:
                            self.events.put(("mask_error", str(error)))
                    threading.Thread(target=extract, daemon=True).start()
        amount = mask.astype(np.float32)[..., None] / 255 * .3
        overlay = np.round(image * (1-amount) + np.array([255, 60, 60]) * amount).astype(np.uint8)
        canvas = self.canvases[0]
        self.scale = min(max(200, canvas.winfo_width()-12) / image.shape[1], max(200, canvas.winfo_height()-12) / image.shape[0])
        preview = Image.fromarray(overlay).resize((max(1, int(image.shape[1]*self.scale)), max(1, int(image.shape[0]*self.scale))))
        self.left_photo = ImageTk.PhotoImage(preview)
        canvas.delete("all")
        canvas.create_image(0, 0, image=self.left_photo, anchor="nw")
        for box in boxes:
            canvas.create_rectangle(*[coordinate*self.scale for coordinate in box], outline="#00ad72", width=2)
        for bubble in data.get("bubbles", []):
            coordinates = [v*self.scale for point in bubble["contour"] for v in point]
            canvas.create_polygon(*coordinates, outline="#3985ff", fill="", width=2)
        for skipped in data.get("skipped", []):
            canvas.create_rectangle(*[v*self.scale for v in skipped["box"]], outline="#df9b32", dash=(3,3))

    def press(self, event):
        if self.loading or self.data is None:
            return
        x, y = event.x/self.scale, event.y/self.scale
        if self.mode.get() == "取消区域":
            for i in reversed(range(len(self.data["added"]))):
                x1,y1,x2,y2 = self.data["added"][i]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.data["added"].pop(i)
                    self.draw()
                    return
            for i, (x1,y1,x2,y2) in enumerate(self.data["automatic"]):
                if x1 <= x <= x2 and y1 <= y <= y2 and i not in self.data["removed"]:
                    self.data["removed"].append(i)
            self.draw()
            return
        self.drag = (x, y)
        self.rect = self.canvases[0].create_rectangle(event.x, event.y, event.x, event.y, outline="#ff8800", width=2)

    def motion(self, event):
        if self.drag is not None:
            self.canvases[0].coords(self.rect, self.drag[0]*self.scale, self.drag[1]*self.scale, event.x, event.y)

    def release(self, event):
        if self.drag is None:
            return
        h,w = self.original.shape[:2]
        x1,x2 = sorted([self.drag[0], event.x/self.scale])
        y1,y2 = sorted([self.drag[1], event.y/self.scale])
        box = [max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))]
        if box[2]-box[0] >= 3 and box[3]-box[1] >= 3:
            self.data["added"].append(box)
        self.drag = None
        self.draw()

    def save(self):
        if self.loading:
            return False
        try:
            data = self.options()
            if fingerprint(self.source) != data["source_hash"]:
                raise ValueError("原稿已改变，请关闭并重新检测")
            with output_lock(self.output):
                atomic_json(cache_path(self.cache_root, self.source), data)
            self.data = data
            self.status.configure(text="区域已保存，仅作用于此页")
            self.draw()
            return True
        except Exception as error:
            messagebox.showerror("保存文字区域", str(error), parent=self)
            return False

    def preview(self):
        if self.loading:
            self.pending_preview = True
            return
        try:
            data = self.options()
        except ValueError as error:
            messagebox.showerror("融合参数", str(error), parent=self)
            return
        self.loading = True
        self.verified_options = None
        self.error_details = ""
        self.help_button.configure(state="disabled")
        self.status.configure(text="本机对齐与融合中…")
        def work():
            try:
                with output_lock(self.output):
                    record = TaskStore(self.output,self.project).current(self.root).get(self.source.relative_to(self.root).as_posix(), {})
                    if record.get("raw_output"):
                        store = TaskStore(self.output,self.project)
                        path = store.output_path({"output": record["raw_output"]})
                        if record.get("source_hash") != fingerprint(self.source) or fingerprint(path) != record.get("raw_hash"):
                            raise ValueError("原稿或原始生成图已改变，请核对后重新生成")
                    else:
                        if record.get("text_metadata") or record.get("text_mask"):
                            raise ValueError("缺少原始生成图，旧白框底色无法可靠恢复")
                        from .review import result_path
                        path = result_path(self.root, self.output, self.source, record)
                    if path is None:
                        raise ValueError("没有已有结果，可先保存区域，再上色")
                    raw = path.read_bytes()
                    self.events.put(("raw_preview", Image.fromarray(rgb(raw))))
                    content, restored, info = fuse(self.source, raw, data, ORIGINAL,
                                                  progress=lambda message:self.events.put(('stage',message)))
                from io import BytesIO
                with Image.open(BytesIO(content)) as image:
                    self.events.put(("preview", (image.copy(), info, data, rgb(restored)[:,:,0], rgb(info.pop("cleanup_png"))[:,:,0])))
            except Exception as error:
                self.events.put(("error", str(error)))
        gc.collect()
        threading.Thread(target=work, daemon=True).start()

    def apply(self):
        if self.loading:
            return
        try:
            verified = self.verified_options == self.signature(self.options())
        except ValueError as error:
            messagebox.showerror("融合参数", str(error), parent=self)
            return
        if not verified:
            self.preview()
            return
        if self.save():
            source = self.source
            self.close()
            self.app.repair_text((source,))

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind in ("loaded", "preview", "error"):
                    self.loading = False
                if kind == "loaded":
                    self.data, self.original = value
                    self.expand.delete(0, "end"); self.expand.insert(0, str(self.data.get("expand", 1)))
                    self.feather.delete(0, "end"); self.feather.insert(0, str(self.data.get("feather", 1)))
                    self.empty.set(self.data.get("confirmed_empty", False))
                    self.draw()
                    self.status.configure(text=f"保护 {len(regions(self.data))} 个区域 · 跳过 {len(self.data.get('skipped', []))} 个 · 疑似漏字 {len(self.data.get('issues', []))} 个；清字范围需先预览融合")
                elif kind == "mask":
                    key, mask = value
                    self.drawing_mask = False
                    if len(self.mask_cache) >= 6:
                        self.mask_cache.clear()
                    self.mask_cache[(key, "restore")] = mask
                    self.draw()
                elif kind == "mask_error":
                    self.drawing_mask = False
                    self.status.configure(text=value)
                elif kind == 'stage':
                    self.status.configure(text=value)
                elif kind == "raw_preview":
                    self.right_image = value
                    self.right_label.configure(text="原始上色图（尚未恢复文字，未采用）")
                    self.draw_right()
                elif kind == "preview":
                    image, info, data, restored, cleanup = value
                    import json
                    key = json.dumps(data, sort_keys=True)
                    if key != json.dumps(self.options(),sort_keys=True):
                        self.pending_preview = True
                        continue
                    self.mask_cache[(key, "restore")] = restored
                    self.mask_cache[(key, "cleanup")] = cleanup
                    self.draw()
                    self.right_image = image
                    self.right_label.configure(text="融合预览（已恢复文字，保存前请核对）")
                    self.draw_right()
                    self.verified_options = key
                    self.status.configure(text=f"融合预览完成 · {info['regions']} 个区域；请核对文字位置，再点击保存并修复此页")
                    # Leave the completed preview visible for human verification.
                else:
                    self.load_failed = self.data is None
                    self.verified_options = None
                    self.error_details = value
                    self.help_button.configure(state="normal")
                    self.status.configure(text=value.splitlines()[0] + " 点击“查看原因与修复方法”；本次未保存修复结果。")
        except queue.Empty:
            pass
        if self.pending_close and not self.loading:
            self.close()
            return
        if self.pending_preview and not self.loading:
            self.pending_preview = False
            self.after_idle(self.preview)
        self.after(80, self.poll)

    def close(self):
        # Background work holds an immutable source/project context and only
        # posts to this editor's private queue. It may finish after UI closure.
        self.closed = True
        self.app.busy = False
        self.destroy()
