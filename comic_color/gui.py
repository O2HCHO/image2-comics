from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
import gc
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from PIL import ImageTk

from .api import ApiConfig, ImageApi, safe_error
from .config import DEFAULT_PROMPT, EDIT_MODE, JSON_MODE, ORIGINAL, RATIOS, load_settings, save_settings
from .imaging import scan_images, thumbnail
from .jobs import BatchWorker, JobConfig, LABELS, TaskStore, output_lock
from .review import result_path, export_pages
from .review_ui import ReviewPanel
from .project_ui import ProjectUI
from .projects import EDIT_SETTINGS


class ColoringApp(ProjectUI, ctk.CTk):
    def __init__(self, settings: dict | None = None):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("漫画上色工作台 · Image2 Comics")
        import sys
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        icon = base / "assets/app.ico"
        if icon.is_file():
            self.iconbitmap(str(icon))
        # CTk geometry uses logical pixels; account for Windows display scaling.
        scaling = self._get_window_scaling()
        width = min(1320, int(self.winfo_screenwidth() / scaling) - 40)
        height = min(880, int(self.winfo_screenheight() / scaling) - 90)
        self.geometry(f"{width}x{height}+20+20")
        self.minsize(min(1000, width), min(600, height))
        self.configure(fg_color="#f1f4f9")
        self.settings = load_settings() if settings is None else settings
        self.events = queue.Queue()
        self.worker = None
        self.single_queue = []
        self.busy = False
        self.io_thread = None
        self.closing = False
        self.app_closed = False
        self.scanning = False
        self.files: list[Path] = []
        self.records: dict[str, dict] = {}
        self.references: tuple[Path, ...] = tuple(Path(p) for p in self.settings.get("references", []) if Path(p).is_file())
        self.preview_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
        self.preview_token = 0
        self.preview_future = None
        self.preview_photos = []
        self.input_var = tk.StringVar(value=self.settings.get("input_dir", ""))
        self.output_var = tk.StringVar(value=self.settings.get("output_dir", ""))
        self.recursive = tk.BooleanVar(value=self.settings.get("recursive", True))
        self.zoom = tk.StringVar(value="适应窗口")
        self.url = tk.StringVar(value=self.settings.get("api_url", ""))
        self.key = tk.StringVar(value=self.settings.get("api_key", ""))
        self.model = tk.StringVar(value=self.settings.get("model", "gpt-image-2"))
        mode = self.settings.get("api_mode", EDIT_MODE)
        self.mode = tk.StringVar(value=mode if mode in (EDIT_MODE, JSON_MODE) else EDIT_MODE)
        self.ratio = tk.StringVar(value=ORIGINAL)
        self.remember = tk.BooleanVar(value=self.settings.get("remember_api_key", False))
        try:
            denoise = min(1., max(0., float(self.settings.get("denoising_strength", .55))))
        except (ValueError, TypeError):
            denoise = .55
        self.denoise = tk.DoubleVar(value=denoise)
        self.preserve_text = tk.BooleanVar(value=self.settings.get("preserve_text", True))
        self._build()
        self.init_projects(legacy=settings is not None)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, lambda: self.state("zoomed"))
        self.after(80, self.poll)
        if self.settings.get("key_error"):
            self.log(self.settings["key_error"])
        if settings is not None and self.input_var.get() and Path(self.input_var.get()).is_dir():
            self.after(200, self.refresh)

    def about(self):
        from . import __version__
        messagebox.showinfo("关于 Image2 Comics", f"漫画上色工作台 {__version__}\nCTD 字形分割与严格气泡文字保护\n\n本版本按 GNU AGPL-3.0 分发，允许依许可证使用、修改和再分发；不提供担保。\n完整许可证及第三方声明见发布包 LICENSE、licenses 和 THIRD_PARTY_NOTICES.txt；对应应用源码在随包 source.zip 中。", parent=self)

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(self, fg_color="#16243c", corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(header, text="漫画上色工作台", text_color="white", font=ctk.CTkFont(size=25, weight="bold")).pack(side="left", padx=24, pady=16)
        ctk.CTkLabel(header, text="保留原稿 · 参考配色 · 批量处理", text_color="#bacbe6").pack(side="left", padx=12)
        ctk.CTkButton(header, text="关于 / 许可证", width=110, command=self.about).pack(side="right", padx=16)
        self.tabs = ctk.CTkTabview(self, fg_color="transparent")
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 12))
        work = self.tabs.add("上色工作台")
        review = self.tabs.add("整批监修 / 单页修改")
        settings = self.tabs.add("接口与上色设置")
        work.grid_columnconfigure(0, weight=1)
        work.grid_rowconfigure(1, weight=1)
        paths = ctk.CTkFrame(work, fg_color="white")
        paths.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        paths.grid_columnconfigure(1, weight=1)
        for row, (label, variable, command) in enumerate((("输入目录", self.input_var, self.choose_input), ("输出目录", self.output_var, self.choose_output))):
            ctk.CTkLabel(paths, text=label).grid(row=row, column=0, padx=14, pady=8)
            ctk.CTkEntry(paths, textvariable=variable, state="readonly").grid(row=row, column=1, sticky="ew", padx=8)
            ctk.CTkButton(paths, text="选择文件夹", width=100, command=command).grid(row=row, column=2, padx=12, pady=8)
        center = ctk.CTkFrame(work, fg_color="transparent")
        center.grid(row=1, column=0, sticky="nsew")
        center.grid_columnconfigure(1, weight=1)
        center.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(center, fg_color="white", width=335)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(2, weight=1, minsize=150)
        left.grid_columnconfigure(0, weight=1)
        tools = ctk.CTkFrame(left, fg_color="transparent")
        tools.grid(row=0, column=0, sticky="ew", padx=8, pady=10)
        ctk.CTkCheckBox(tools, text="包含子目录", variable=self.recursive, command=self.refresh, width=120).pack(side="left", padx=4)
        ctk.CTkButton(tools, text="刷新 / 恢复任务", width=115, command=self.refresh).pack(side="right")
        actions = ctk.CTkFrame(left, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew", padx=10)
        for label, command in (("全选", lambda: self.tree.selection_set(self.tree.get_children())),
                               ("取消选择", lambda: self.tree.selection_remove(self.tree.selection())),
                               ("仅选失败", self.select_failed)):
            ctk.CTkButton(actions, text=label, width=85, command=command).pack(side="left", padx=3, pady=4)
        table = ttk.Frame(left)
        table.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)
        table.grid_rowconfigure(0, weight=1)
        table.grid_columnconfigure(0, weight=1)
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=29, font=("Microsoft YaHei UI", 10))
        self.tree = ttk.Treeview(table, columns=("file", "state"), show="headings", selectmode="extended", height=10)
        self.tree.heading("file", text="漫画页（Ctrl / Shift 多选）")
        self.tree.heading("state", text="状态")
        self.tree.column("file", width=240, minwidth=150)
        self.tree.column("state", width=70, stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.selection_changed)
        self.selection_label = ctk.CTkLabel(left, text="选择输入目录开始", anchor="w")
        self.selection_label.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 6))
        ctk.CTkButton(left, text='选中单页：不需上色 / 输出原图', command=self.use_original).grid(row=4, column=0, sticky='ew', padx=14, pady=6)
        right = ctk.CTkFrame(center, fg_color="white")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure((0, 1), weight=1, uniform="preview")
        right.grid_rowconfigure(1, weight=1, minsize=150)
        for column, text in enumerate(("原始漫画", "上色结果")):
            ctk.CTkLabel(right, text=text, font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=column, pady=10)
        self.canvases = []
        for column in range(2):
            frame = ttk.Frame(right)
            frame.grid(row=1, column=column, sticky="nsew", padx=8)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(0, weight=1)
            canvas = tk.Canvas(frame, bg="#e8edf4", highlightthickness=0)
            canvas.grid(row=0, column=0, sticky="nsew")
            vertical = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            vertical.grid(row=0, column=1, sticky="ns")
            horizontal = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
            horizontal.grid(row=1, column=0, sticky="ew")
            canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
            self.canvases.append(canvas)
        footer = ctk.CTkFrame(right, fg_color="transparent")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=8)
        ctk.CTkLabel(footer, text="预览缩放").pack(side="left", padx=6)
        ctk.CTkOptionMenu(footer, variable=self.zoom, values=["适应窗口", "25%", "50%", "100%"], width=110, command=lambda _: self.show_preview()).pack(side="left")
        ctk.CTkButton(footer, text="打开输出目录", width=120, command=self.open_output).pack(side="right")
        ctk.CTkButton(footer, text="监修 / 改本页提示词", width=150, command=self.review_focused).pack(side="right", padx=8)
        self.detail = ctk.CTkLabel(right, text="保真与跨页配色效果取决于模型，请抽查文字和人物细节。", wraplength=650, anchor="w")
        self.detail.grid(row=3, column=0, columnspan=2, sticky="ew", padx=14, pady=6)
        controls = ctk.CTkFrame(work, fg_color="white")
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        controls.grid_columnconfigure(5, weight=1)
        self.start_button = ctk.CTkButton(controls, text="开始 / 继续上色", fg_color="#187a65", hover_color="#126451", command=self.start)
        self.start_button.grid(row=0, column=0, padx=10, pady=12)
        self.retry_button = ctk.CTkButton(controls, text="重试失败项", width=110, command=lambda: self.start(failed_only=True))
        self.retry_button.grid(row=0, column=1, padx=4)
        self.pause_button = ctk.CTkButton(controls, text="暂停", width=80, state="disabled", command=self.toggle_pause)
        self.pause_button.grid(row=0, column=2, padx=4)
        self.stop_button = ctk.CTkButton(controls, text="停止", width=80, state="disabled", command=self.stop)
        self.stop_button.grid(row=0, column=3, padx=4)
        self.status = ctk.CTkLabel(controls, text="就绪")
        self.status.grid(row=0, column=5, sticky="e", padx=16)
        self.progress = ctk.CTkProgressBar(controls)
        self.progress.grid(row=1, column=0, columnspan=6, sticky="ew", padx=14, pady=(0, 8))
        self.progress.set(0)
        self.counts = ctk.CTkLabel(controls, text="成功 0   失败 0   跳过 0   待确认 0   待处理 0")
        self.counts.grid(row=2, column=0, columnspan=6, sticky="w", padx=14)
        self.logs = ctk.CTkTextbox(work, height=85, fg_color="#17243b", text_color="#d3e2f4", font=("Microsoft YaHei UI", 11))
        self.logs.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self._build_settings(settings)
        self.review_panel = ReviewPanel(review, self)
        self.review_panel.pack(fill="both", expand=True)

    def result_for(self, source):
        if not self.output_var.get():
            return None
        return result_path(Path(self.input_var.get()), Path(self.output_var.get()), source,
                           self.records.get(self.relative(source), {}))

    def review_focused(self):
        item = self.tree.focus()
        if not item or int(item) >= len(self.files):
            return
        source = self.files[int(item)]
        self.review_panel.page = int(item) // self.review_panel.PAGE_SIZE
        self.review_panel.select(source)
        self.review_panel.refresh()
        self.tabs.set("整批监修 / 单页修改")

    def _build_settings(self, parent):
        panel = ctk.CTkScrollableFrame(parent, fg_color="white")
        panel.pack(fill="both", expand=True)
        panel.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(panel, text="连接你的图像编辑服务", font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=18)
        for row, label, variable in ((1, "API URL", self.url), (2, "API Key", self.key)):
            ctk.CTkLabel(panel, text=label).grid(row=row, column=0, padx=20, pady=10, sticky="w")
            ctk.CTkEntry(panel, textvariable=variable, show="*" if row == 2 else "").grid(row=row, column=1, sticky="ew", padx=20, pady=10)
        ctk.CTkCheckBox(panel, text="记住 API Key（仅本 Windows 账户可解密）", variable=self.remember).grid(row=3, column=1, sticky="w", padx=20, pady=10)
        ctk.CTkLabel(panel, text="模型").grid(row=4, column=0, padx=20, sticky="w")
        models = ctk.CTkFrame(panel, fg_color="transparent")
        models.grid(row=4, column=1, sticky="ew", padx=20, pady=10)
        models.grid_columnconfigure(0, weight=1)
        self.model_box = ctk.CTkComboBox(models, variable=self.model, values=[self.model.get()])
        self.model_box.grid(row=0, column=0, sticky="ew")
        self.test_button = ctk.CTkButton(models, text="测试连接 / 获取模型", command=self.test_connection)
        self.test_button.grid(row=0, column=1, padx=(12, 0))
        ctk.CTkLabel(panel, text="接口模式").grid(row=5, column=0, sticky="w", padx=20)
        ctk.CTkOptionMenu(panel, variable=self.mode, values=[EDIT_MODE, JSON_MODE]).grid(row=5, column=1, sticky="ew", padx=20, pady=10)
        ctk.CTkLabel(panel, text="上色提示词").grid(row=6, column=0, sticky="nw", padx=20, pady=12)
        self.prompt = ctk.CTkTextbox(panel, height=150, border_width=1)
        self.prompt.grid(row=6, column=1, sticky="ew", padx=20, pady=10)
        self.prompt.insert("1.0", self.settings.get("prompt") or DEFAULT_PROMPT)
        ctk.CTkButton(panel, text="恢复保真提示词", command=self.reset_prompt).grid(row=7, column=1, sticky="w", padx=20)
        ctk.CTkLabel(panel, text="配色参考图").grid(row=8, column=0, sticky="nw", padx=20, pady=20)
        refs = ctk.CTkFrame(panel, fg_color="transparent")
        refs.grid(row=8, column=1, sticky="ew", padx=20, pady=10)
        ctk.CTkButton(refs, text="选择参考图（多选）", command=self.choose_references).pack(side="left")
        ctk.CTkButton(refs, text="移除参考图", width=100, command=self.clear_references).pack(side="left", padx=10)
        self.ref_label = ctk.CTkLabel(panel, text="", wraplength=750, anchor="w", justify="left")
        self.ref_label.grid(row=9, column=1, sticky="ew", padx=20)
        self.update_references()
        ctk.CTkLabel(panel, text="输出画布比例\n原始比例将严格锁定").grid(row=10, column=0, sticky="w", padx=20)
        ctk.CTkLabel(panel, text="始终与原稿一致，可提高分辨率；不裁切、不补边", anchor="w").grid(row=10, column=1, sticky="ew", padx=20, pady=10)
        ctk.CTkLabel(panel, text="重绘幅度\n仅自定义 JSON 有效").grid(row=11, column=0, sticky="w", padx=20)
        ctk.CTkSlider(panel, variable=self.denoise, from_=0, to=1, number_of_steps=100).grid(row=11, column=1, sticky="ew", padx=20, pady=16)
        ctk.CTkLabel(panel, text="图像编辑地址示例：https://服务域名/v1/images/edits\n参考图需要服务支持多图编辑；上色会消耗你配置的 API 额度。", justify="left", text_color="#56657a").grid(row=12, column=1, sticky="w", padx=20, pady=12)
        ctk.CTkButton(panel, text="保存设置", command=self.save).grid(row=13, column=1, sticky="w", padx=20, pady=16)
        ctk.CTkCheckBox(panel, text="保留原稿文字（本机检测＋掩码融合，优先对白旁白）", variable=self.preserve_text).grid(row=14, column=1, sticky="w", padx=20, pady=12)
        ctk.CTkLabel(panel, text="对齐失败或没有文字区域时保留生成图、标记需核对；可在监修页编辑区域并本地修复，不再请求 API。", wraplength=700, justify="left").grid(row=15, column=1, sticky="w", padx=20, pady=8)

    def log(self, message):
        logging.info("%s", safe_error(message, self.key.get()))
        self.logs.insert("end", message + "\n")
        self.logs.see("end")
        if int(self.logs.index("end").split(".")[0]) > 350:
            self.logs.delete("1.0", "80.0")

    def idle_required(self):
        if self.busy or self.scanning or self.single_queue or (self.worker and self.worker.is_alive()):
            messagebox.showinfo("任务正在运行", "请等待扫描、连接测试或上色任务结束后再操作。")
            return False
        return True

    def choose_input(self):
        if self.project:
            messagebox.showinfo('项目目录','项目输入目录已固定。请新建项目；目录移动后重新打开项目进行定位。')
            return
        if not self.idle_required():
            return
        value = filedialog.askdirectory(title="选择漫画输入目录")
        if value:
            self.input_var.set(value)
            if not self.output_var.get():
                self.output_var.set(str(Path(value).parent / (Path(value).name + "_上色")))
            self.refresh()

    def choose_output(self):
        if self.project:
            messagebox.showinfo('项目目录','项目输出目录已固定，避免混合结果。请通过项目首页管理其他漫画。')
            return
        if not self.idle_required():
            return
        value = filedialog.askdirectory(title="选择上色结果目录")
        if value:
            self.output_var.set(value)
            self.refresh()

    def refresh(self):
        if not self.idle_required():
            return
        root = Path(self.input_var.get())
        if not self.input_var.get() or not root.is_dir():
            return
        output = Path(self.output_var.get()) if self.output_var.get() else None
        recursive = self.recursive.get()
        project, events = self.project, self.events
        current_workspace = self.workspace_values() if self.files else (self.restore_workspace or {})
        self.scanning = True
        self.status.configure(text="扫描图片 / 恢复任务…")
        def scan():
            try:
                started = time.perf_counter()
                files = project.scan(recursive)[0] if project else scan_images(root, output, recursive)
                scan_seconds = time.perf_counter()-started
                records = {}
                if output and output.is_dir():
                    with output_lock(output):
                        started = time.perf_counter()
                        store = TaskStore(output,project)
                        store.recover()
                        records = store.current(root)
                        events.put(('stage',{'message':f'任务恢复 {time.perf_counter()-started:.2f}s；目录核对 {scan_seconds:.2f}s'}))
                events.put(("scan", {"files": files, "records": records,'workspace':current_workspace}))
            except Exception as error:
                events.put(("scan_error", {"message": str(error)}))
        gc.collect()
        threading.Thread(target=scan, daemon=True).start()

    def select_failed(self):
        self.tree.selection_set([str(i) for i, path in enumerate(self.files)
                                 if self.records.get(self.relative(path), {}).get("status") == "failed"])

    def relative(self, path):
        return path.relative_to(Path(self.input_var.get()).resolve()).as_posix()

    def selection_changed(self, _=None):
        self.selection_label.configure(text=f"已选择 {len(self.tree.selection())} / {len(self.files)} 张")
        self.show_preview()

    def show_preview(self):
        self.preview_token += 1
        token = self.preview_token
        item = self.tree.focus()
        if not item or not self.files or int(item) >= len(self.files):
            return
        source = self.files[int(item)]
        record = self.records.get(self.relative(source), {})
        result = self.result_for(source)
        self.detail.configure(text=record.get("error") or "保真与跨页配色效果取决于模型，请抽查文字和人物细节。")
        sizes = [(max(180, c.winfo_width() - 16), max(180, c.winfo_height() - 16)) for c in self.canvases]
        zoom = self.zoom.get()
        events = self.events
        if self.preview_future:
            self.preview_future.cancel()
        def load():
            images = []
            try:
                from PIL import Image
                for path, fit in zip((source, result), sizes):
                    if path is None:
                        images.append(None)
                        continue
                    if zoom != "适应窗口":
                        with Image.open(path) as image:
                            scale = min(float(zoom.rstrip("%")) / 100, 2400 / max(image.size))
                            fit = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                    images.append(thumbnail(path, fit))
                events.put(("preview", {"token": token, "images": images}))
            except Exception as error:
                events.put(("preview", {"token": token, "images": [None, None], "error": str(error)}))
        gc.collect()
        self.preview_future = self.preview_pool.submit(load)

    def choose_references(self):
        if not self.idle_required():
            return
        paths = filedialog.askopenfilenames(title="选择已上色参考图", filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if paths:
            self.references = tuple(Path(path) for path in paths)
            self.update_references()

    def clear_references(self):
        if self.idle_required():
            self.references = ()
            self.update_references()

    def update_references(self):
        self.ref_label.configure(text="\n".join(str(p) for p in self.references) or "未选择参考图；可逐页独立上色。")

    def reset_prompt(self):
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", DEFAULT_PROMPT)

    def collect_settings(self):
        return {"api_url": self.url.get().strip(), "api_key": self.key.get().strip(), "model": self.model.get().strip(),
                "api_mode": self.mode.get(), "prompt": self.prompt.get("1.0", "end-1c"),
                "remember_api_key": self.remember.get(), "denoising_strength": self.denoise.get(),
                "output_ratio": self.ratio.get(), "input_dir": self.input_var.get(), "output_dir": self.output_var.get(),
                "references": [str(p) for p in self.references], "recursive": self.recursive.get(),
                "preserve_text": self.preserve_text.get()}

    def save(self):
        try:
            settings = self.collect_settings()
            if self.project:
                self.save_workspace()
            if not self.legacy_workspace:
                settings = {k:v for k,v in settings.items() if k not in EDIT_SETTINGS | {'input_dir','output_dir'}}
            save_settings(settings)
            self.log("设置已保存在本机。")
            return True
        except Exception as error:
            messagebox.showerror("保存设置失败", safe_error(error, self.key.get()))
            return False

    def test_connection(self):
        if not self.idle_required():
            return
        url, key = self.url.get().strip(), self.key.get().strip()
        if not key:
            messagebox.showerror("缺少密钥", "请先填写 API Key。")
            return
        self.busy = True
        self.test_button.configure(state="disabled")
        events = self.events
        def test():
            try:
                events.put(("models", {"models": ImageApi().models(url, key)}))
            except Exception as error:
                events.put(("models", {"error": safe_error(error, key)}))
        gc.collect()
        threading.Thread(target=test, daemon=True).start()

    def start_single(self, source, prompt):
        if self.closing or self.busy or self.scanning or (self.worker and self.worker.is_alive() and not isinstance(self.worker, BatchWorker)):
            messagebox.showinfo('暂不能加入队列', '请等待扫描或本地操作完成。')
            return
        if source not in self.files or not self.output_var.get():
            return
        api = ApiConfig(self.url.get().strip(), self.key.get().strip(), self.model.get().strip(),
                        prompt, self.mode.get(), self.denoise.get(), self.ratio.get(), tuple(self.references),
                        preserve_text=self.preserve_text.get())
        try:
            api.validate()
        except ValueError as error:
            messagebox.showerror('接口配置', str(error))
            return
        cfg = JobConfig(Path(self.input_var.get()), Path(self.output_var.get()), (source,), api,
                        force=True, project=self.project)
        if any(item.files == cfg.files for item in self.single_queue):
            messagebox.showinfo('已在等待队列', '此页已等待生成；如需修改提示词，请清空等待队列后重新加入。')
            return
        self.single_queue.append(cfg)
        self.update_single_queue()
        self.log('加入单页生成队列：' + self.relative(source))
        self.after(100, self.next_single)

    def update_single_queue(self):
        names = [cfg.files[0].name for cfg in self.single_queue]
        self.review_panel.queue_status.configure(text=f'等待 {len(names)} 页：' + ' → '.join(names) if names else '等待队列为空')

    def clear_single_queue(self):
        self.single_queue.clear()
        self.update_single_queue()

    def next_single(self):
        if self.closing or not self.single_queue:
            return
        if self.worker and self.worker.is_alive():
            self.after(100, self.next_single)
            return
        if self.busy or self.scanning:
            return
        config = self.single_queue.pop(0)
        self.update_single_queue()
        self.launch_generation(config)

    def choose_version(self, source, identity):
        if not self.idle_required():
            return False
        try:
            output = Path(self.output_var.get())
            with output_lock(output):
                record = self.task_store(output).use_version(Path(self.input_var.get()), source, identity)
            self.events.put(('item', dict(source=self.relative(source), status=record['status'], record=record)))
            self.log('已采用历史版本：' + self.relative(source))
            return True
        except Exception as error:
            messagebox.showerror('采用历史版本失败', str(error))
            return False

    def use_original(self, source=None):
        if not self.idle_required():
            return
        if source is None:
            selected = self.tree.selection()
            if len(selected) != 1:
                messagebox.showinfo('保留原图', '请只选中一张图片。')
                return
            source = self.files[int(selected[0])]
        if not self.output_var.get():
            messagebox.showinfo('保留原图', '请先选择输出目录。')
            return
        try:
            output = Path(self.output_var.get())
            with output_lock(output):
                store = self.task_store(output)
                record = store.use_original(Path(self.input_var.get()), source)
            self.events.put(('item', dict(source=self.relative(source), status='success', record=record)))
            self.log('此页不需上色，原图已输出；批量续跑将保留原图。重新生成此页可恢复上色。')
        except Exception as error:
            messagebox.showerror('保留原图失败', str(error))

    def start(self, failed_only=False, single=None):
        if not self.idle_required():
            return
        selected = set(self.tree.selection())
        files = tuple(path for i, path in enumerate(self.files) if
                      (self.records.get(self.relative(path), {}).get("status") == "failed" if failed_only else str(i) in selected))
        if single:
            files = (single[0],) if single[0] in self.files else ()
        if not files:
            messagebox.showinfo("没有待处理图片", "请先选择图片。重试失败项仅处理当前目录中的失败记录。")
            return
        if not self.output_var.get():
            messagebox.showerror("缺少输出目录", "请选择输出目录。")
            return
        api = ApiConfig(self.url.get().strip(), self.key.get().strip(), self.model.get().strip(),
                        self.prompt.get("1.0", "end-1c").strip(), self.mode.get(), self.denoise.get(), self.ratio.get(), self.references)
        if single:
            api = replace(api, prompt=single[1])
        api = replace(api, preserve_text=self.preserve_text.get())
        try:
            api.validate()
        except ValueError as error:
            messagebox.showerror("接口配置", str(error))
            self.tabs.set("接口与上色设置")
            return
        uncertain = any(self.records.get(self.relative(path), {}).get("status") == "uncertain" for path in files)
        if uncertain and not messagebox.askyesno("确认重试中断请求", "所选图片包含上次中断的请求。服务可能已经计费，重试可能再次计费。是否重试这些图片？"):
            return
        if not self.save():
            return
        config = JobConfig(Path(self.input_var.get()), Path(self.output_var.get()), files, api,
                           allow_uncertain=uncertain, force=bool(single), project=self.project)
        self.launch_generation(config)

    def launch_generation(self, config):
        self.worker = BatchWorker(config, self.events)
        self.review_panel.regenerate_button.configure(state="normal")
        self.start_button.configure(state="disabled")
        self.retry_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="暂停")
        self.stop_button.configure(state="normal")
        self.progress.set(0)
        self.status.configure(text="正在处理…")
        self.log(f"开始处理 {len(config.files)} 张；单页修订将生成新版本。")
        gc.collect()
        self.worker.start()

    def edit_text(self):
        if not self.idle_required():
            return
        source = self.review_panel.current
        if source is None or not self.output_var.get():
            messagebox.showinfo("文字保护", "请先在监修页选中一页并选择输出目录。")
            return
        from .text_editor import TextEditor
        self.busy = True
        self.text_editor = TextEditor(self, source)
        self.text_editor.transient(self)
        self.text_editor.grab_set()

    def repair_text(self, files=None):
        if not self.idle_required():
            return
        if not self.output_var.get() or not self.input_var.get():
            return
        chosen = tuple(files) if files is not None else tuple(p for p in self.files if
                 self.result_for(p) is not None or self.records.get(self.relative(p), {}).get("raw_output"))
        if not chosen:
            messagebox.showinfo("文字修复", "没有可修复的已有结果。")
            return
        from .text_jobs import RepairWorker
        self.worker = RepairWorker(Path(self.input_var.get()), Path(self.output_var.get()), chosen, self.events,self.project)
        self.start_button.configure(state="disabled")
        self.retry_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="暂停")
        self.stop_button.configure(state="normal")
        self.review_panel.regenerate_button.configure(state="disabled")
        self.status.configure(text="本地文字修复中…")
        self.progress.set(0)
        self.log(f"本地修复 {len(chosen)} 页，不请求上色 API。")
        gc.collect()
        self.worker.start()

    def export_pdf(self):
        if not self.idle_required():
            return
        if not self.files or not self.output_var.get():
            messagebox.showinfo("导出 PDF", "请先选择输入、输出目录并生成上色结果。")
            return
        root, output, files = Path(self.input_var.get()), Path(self.output_var.get()), tuple(self.files)
        self.busy = True
        self.status.configure(text="检查全部上色结果…")
        self.review_panel.export_button.configure(state="disabled")
        self.review_panel.export_status.configure(text="检查全部上色结果…")
        project, events = self.project, self.events
        def prepare():
            try:
                with output_lock(output):
                    records = TaskStore(output,project).current(root)
                    pages, missing = export_pages(root, output, files, records)
                if not pages:
                    raise ValueError("没有可导出的上色结果")
                events.put(("pdf_plan", {"root": root, "output": output, "files": files,
                                             "pages": pages, "missing": missing,'project':project}))
            except Exception as error:
                events.put(("pdf_error", {"message": str(error)}))
        self.io_thread = threading.Thread(target=prepare, daemon=False)
        gc.collect()
        self.io_thread.start()

    def write_pdf(self, plan, destination):
        events = self.events
        def write():
            try:
                from .pdf_export import export_pdf
                with output_lock(plan["output"]):
                    pages, missing = export_pages(plan["root"], plan["output"], plan["files"],
                                                  TaskStore(plan["output"],plan.get('project')).current(plan["root"]))
                    if pages != plan["pages"] or missing != plan["missing"]:
                        raise ValueError("上色结果已发生变化，请重新导出")
                    count = export_pdf(pages, Path(destination),
                                       lambda done, total: events.put(("pdf_progress", {"done": done, "total": total})))
                events.put(("pdf_done", {"path": str(destination), "count": count}))
            except Exception as error:
                events.put(("pdf_error", {"message": str(error)}))
        self.io_thread = threading.Thread(target=write, daemon=False)
        gc.collect()
        self.io_thread.start()

    def toggle_pause(self):
        if not self.worker or not self.worker.is_alive():
            return
        if self.worker.resume_event.is_set():
            self.worker.pause()
            self.pause_button.configure(text="继续")
            self.status.configure(text="当前请求完成后暂停…")
        else:
            self.worker.resume()
            self.pause_button.configure(text="暂停")
            self.status.configure(text="正在处理…")

    def stop(self):
        self.clear_single_queue()
        if self.worker:
            self.worker.stop()
            self.stop_button.configure(state="disabled")
            self.pause_button.configure(state="disabled")
            self.status.configure(text="等待当前请求保存后停止…")

    def open_output(self):
        value = self.output_var.get()
        if value and Path(value).is_dir():
            os.startfile(str(Path(value).resolve()))
        else:
            messagebox.showinfo("输出目录", "输出目录尚未创建，请先完成一次上色。")

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if payload.get('_scope', self.events.scope) != self.events.scope:
                    continue
                if kind == 'stage':
                    self.status.configure(text=payload['message'])
                    self.log(payload['message'])
                elif kind == "scan":
                    self.scanning = False
                    workspace = self.workspace_values() if self.files else (self.restore_workspace or payload.get('workspace',{}))
                    self.files, self.records = payload["files"], payload["records"]
                    self.preview_token += 1
                    self.preview_photos = []
                    for canvas in self.canvases:
                        canvas.delete("all")
                    if self.project:
                        self.restore_workspace = workspace
                        self.populate_pages(restore=True)
                    else:
                        self.populate_pages()
                    self.status.configure(text=f"就绪 · {len(self.files)} 张")
                    self.counts.configure(text=f"已扫描 {len(self.files)} 张；开始后将校验可复用结果。")
                    self.log(f"扫描完成：{len(self.files)} 张。成功状态为历史记录，续跑时会重新校验。")
                elif kind == "scan_error":
                    self.scanning = False
                    self.status.configure(text="扫描失败")
                    self.log(payload["message"])
                elif kind == "preview" and payload["token"] == self.preview_token:
                    self.preview_photos = []
                    for canvas, image in zip(self.canvases, payload["images"]):
                        canvas.delete("all")
                        if image:
                            photo = ImageTk.PhotoImage(image)
                            self.preview_photos.append(photo)
                            canvas.create_image(0, 0, anchor="nw", image=photo)
                            canvas.configure(scrollregion=(0, 0, image.width, image.height))
                        else:
                            canvas.create_text(20, 30, anchor="nw", text="暂无可预览图片", fill="#66758c")
                    if payload.get("error"):
                        self.detail.configure(text="预览失败：" + payload["error"])
                    scroll = getattr(self,'pending_scroll',{}).pop('preview',None)
                    if scroll:
                        for canvas,(x,y) in zip(self.canvases,scroll):
                            canvas.xview_moveto(x);canvas.yview_moveto(y)
                elif kind == "models":
                    self.busy = False
                    self.test_button.configure(state="normal")
                    if payload.get("error"):
                        self.log(payload["error"])
                        if not self.closing:
                            messagebox.showerror("连接测试失败", payload["error"])
                    else:
                        models = payload["models"]
                        self.model_box.configure(values=models)
                        if self.model.get() not in models:
                            self.model.set(models[0])
                        self.log(f"连接成功，获取 {len(models)} 个模型。")
                elif kind == "item":
                    rel, state, record = payload["source"], payload["status"], payload["record"]
                    self.records[rel] = record
                    for i, path in enumerate(self.files):
                        if self.relative(path) == rel:
                            self.tree.set(str(i), "state", LABELS[state])
                            break
                    self.log(f"{LABELS[state]}  {rel}" + (" · " + record["error"] if record.get("error") else ""))
                    if state in ("success", "failed", "skipped", "needs_review"):
                        self.show_preview()
                        self.review_panel.refresh()
                elif kind == "gallery":
                    self.review_panel.loaded(payload)
                elif kind == "pdf_plan":
                    if self.closing:
                        self.busy = False
                        self.review_panel.export_button.configure(state="normal")
                        continue
                    missing = payload["missing"]
                    if missing and not messagebox.askyesno("部分页面尚未上色", f"共 {len(payload['files'])} 页，其中 {len(missing)} 页没有结果。仅导出已有的 {len(payload['pages'])} 页吗？\n\n缺失示例：" + "、".join(missing[:5])):
                        self.busy = False
                        self.status.configure(text="已取消导出")
                        self.review_panel.export_button.configure(state="normal")
                        self.review_panel.export_status.configure(text="已取消导出")
                        continue
                    destination = filedialog.asksaveasfilename(title="按漫画页序导出当前上色结果", defaultextension=".pdf",
                                                              initialfile=payload["root"].name + "_上色.pdf",
                                                              filetypes=[("PDF", "*.pdf")])
                    if destination:
                        self.write_pdf(payload, destination)
                    else:
                        self.busy = False
                        self.status.configure(text="已取消导出")
                        self.review_panel.export_button.configure(state="normal")
                        self.review_panel.export_status.configure(text="已取消导出")
                elif kind == "pdf_progress":
                    self.status.configure(text=f"导出 PDF：{payload['done']} / {payload['total']}")
                    self.review_panel.export_status.configure(text=f"导出 PDF：{payload['done']} / {payload['total']}")
                elif kind in ("pdf_done", "pdf_error"):
                    self.busy = False
                    self.review_panel.export_button.configure(state="normal")
                    if kind == "pdf_done":
                        self.review_panel.export_status.configure(text=f"PDF 导出完成 · {payload['count']} 页")
                        self.status.configure(text=f"PDF 导出完成 · {payload['count']} 页")
                        self.log("PDF 已保存：" + payload["path"])
                        if not self.closing:
                            messagebox.showinfo("PDF 导出完成", f"已按顺序导出 {payload['count']} 页。\n{payload['path']}")
                    else:
                        self.review_panel.export_status.configure(text="PDF 导出失败")
                        self.status.configure(text="PDF 导出失败")
                        self.log(payload["message"])
                        if not self.closing:
                            messagebox.showerror("PDF 导出失败", payload["message"])
                elif kind == "progress":
                    s = payload["summary"]
                    self.progress.set(payload["index"] / payload["total"])
                    self.counts.configure(text=f"成功 {s['success']}   失败 {s['failed']}   跳过 {s['skipped']}   待确认 {s['uncertain']}   待处理 {payload['total'] - payload['index']}")
                elif kind == "paused":
                    self.status.configure(text="已暂停")
                elif kind == "error":
                    self.log("任务停止：" + payload["message"])
                    self.status.configure(text="任务出错，查看日志")
                elif kind == "finished":
                    if payload['stopped']:
                        self.clear_single_queue()
                    elif not self.closing:
                        self.after(150, self.next_single)
                    self.review_panel.regenerate_button.configure(state="normal")
                    self.start_button.configure(state="normal")
                    self.retry_button.configure(state="normal")
                    self.pause_button.configure(state="disabled", text="暂停")
                    self.stop_button.configure(state="disabled")
                    self.status.configure(text="已停止" if payload["stopped"] else "本轮结束 · 查看统计和日志")
                    self.log("本轮任务结束。失败与待确认图片不会被标记为成功。")
        except queue.Empty:
            pass
        if self.closing and not self.busy and not (self.worker and self.worker.is_alive()):
            self.finish_close()
            return
        self.after(80, self.poll)

    def on_close(self):
        if self.closing:
            return
        editor = getattr(self, "text_editor", None)
        if editor is not None and not editor.closed:
            editor.close()
            if not editor.closed:
                return
        if self.busy or (self.worker and self.worker.is_alive()):
            if not messagebox.askyesno("安全关闭", "任务正在运行。是否等待当前请求或 PDF 导出保存后关闭？"):
                return
            self.closing = True
            self.stop()
            self.save()
        else:
            self.save()
            self.finish_close()

    def finish_close(self):
        if not self.app_closed:
            self.app_closed = True
            self.preview_pool.shutdown(wait=False, cancel_futures=True)
            for timer in self.tk.call("after", "info"):
                self.tk.call("after", "cancel", timer)
            self.destroy()
