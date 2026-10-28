"""Single-page paired canvases with keyboard navigation and prompt drafts."""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import gc
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image

import customtkinter as ctk

from .config import DEFAULT_PROMPT
from .imaging import thumbnail
from .jobs import LABELS


class ReviewPanel(ctk.CTkFrame):
    PAGE_SIZE = 1

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.page = 0
        self.token = 0
        self.photos = []
        self.labels = []
        self.current = None
        self.drafts = {}
        self.future = None
        self.image_cache = OrderedDict()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        navigation = ctk.CTkFrame(self, fg_color="white")
        navigation.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(navigation, text="大图监修 · 原稿与当前结果", font=ctk.CTkFont(size=17, weight="bold")).pack(side="left", padx=14, pady=12)
        self.export_button = ctk.CTkButton(navigation, text="导出全部上色图为 PDF", command=app.export_pdf, width=180)
        self.export_button.pack(side="right", padx=12)
        ctk.CTkButton(navigation, text="历史版本 / 采用旧版", command=self.history, width=150).pack(side="right", padx=6)
        self.gallery = ctk.CTkFrame(self, fg_color="#e8edf4")
        self.gallery.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.gallery.grid_columnconfigure((0, 1), weight=1, uniform="pair")
        self.gallery.grid_rowconfigure(1, weight=1)
        self.canvases = []
        for column, title in enumerate(("原始漫画", "当前采用结果")):
            ctk.CTkLabel(self.gallery, text=title).grid(row=0, column=column, pady=6)
            frame = ctk.CTkFrame(self.gallery)
            frame.grid(row=1, column=column, sticky="nsew", padx=3)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(0, weight=1)
            canvas = tk.Canvas(frame, bg="#e8edf4", highlightthickness=0)
            canvas.grid(row=0, column=0, sticky="nsew")
            vertical = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            vertical.grid(row=0, column=1, sticky="ns")
            horizontal = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
            horizontal.grid(row=1, column=0, sticky="ew")
            canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
            canvas.bind("<MouseWheel>", lambda event, c=canvas: c.yview_scroll(-int(event.delta/120), "units"))
            canvas.bind("<ButtonPress-1>", lambda e, c=canvas: c.scan_mark(e.x,e.y))
            canvas.bind("<B1-Motion>", lambda e, c=canvas: c.scan_dragto(e.x,e.y,gain=1))
            self.canvases.append(canvas)
        self.zoom = tk.StringVar(value="适应窗口")
        ctk.CTkOptionMenu(self.gallery, variable=self.zoom, values=["适应窗口", "50%", "100%", "150%"], command=lambda _: self.refresh()).grid(row=2,column=0,columnspan=2,pady=6)
        app.bind("<Left>", lambda e: self.keyboard_page(e,-1), add="+")
        app.bind("<Right>", lambda e: self.keyboard_page(e,1), add="+")
        editor = ctk.CTkFrame(self, fg_color="white", width=330)
        editor.grid(row=1, column=1, sticky="ns")
        editor.grid_propagate(False)
        editor.grid_columnconfigure(0, weight=1)
        editor.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(editor, text="单页监修", font=ctk.CTkFont(size=19, weight="bold")).grid(row=0, column=0, padx=16, pady=16, sticky="w")
        self.title = ctk.CTkLabel(editor, text="点击左侧任意一组对照图", wraplength=290, anchor="w", justify="left")
        self.title.grid(row=1, column=0, padx=16, pady=8, sticky="ew")
        ctk.CTkLabel(editor, text="本页完整提示词（不改动全局设置）", anchor="w").grid(row=2, column=0, padx=16, pady=8, sticky="ew")
        self.prompt = ctk.CTkTextbox(editor, height=160, border_width=1, wrap="word")
        self.prompt.grid(row=3, column=0, padx=16, pady=8, sticky="nsew")
        ctk.CTkLabel(editor, text="从原始漫画整页重新上色；沿用接口、模型和配色参考图。成功后更新对照，上一版文件保留。", wraplength=290, text_color="#56657a", justify="left").grid(row=4, column=0, padx=16, pady=8, sticky="ew")
        self.regenerate_button = ctk.CTkButton(editor, text="重新生成此页 / 加入队列", fg_color="#187a65", command=self.regenerate)
        self.regenerate_button.grid(row=5, column=0, padx=16, pady=10, sticky="ew")
        ctk.CTkButton(editor, text="此页不需上色，直接输出原图", command=lambda: app.use_original(self.current) if self.current else None).grid(row=6, column=0, padx=16, pady=8, sticky="ew")
        self.notice = ctk.CTkLabel(editor, text="", wraplength=290, justify="left", anchor="w")
        self.notice.grid(row=10, column=0, padx=16, pady=8, sticky="ew")
        self.queue_status = ctk.CTkLabel(editor, text="等待队列为空", wraplength=290, justify="left")
        self.queue_status.grid(row=11, column=0, padx=16, sticky="ew")
        ctk.CTkButton(editor, text="清空等待队列（不停止当前页）", command=app.clear_single_queue).grid(row=12, column=0, padx=16, pady=6, sticky="ew")
        ctk.CTkButton(editor, text="文字区域 / 掩码预览", command=app.edit_text).grid(row=7, column=0, padx=16, pady=4, sticky="ew")
        ctk.CTkButton(editor, text="本地修复当前页文字", command=lambda: app.repair_text((self.current,)) if self.current else None).grid(row=8, column=0, padx=16, pady=4, sticky="ew")
        ctk.CTkButton(editor, text="本地修复全部已有结果", command=app.repair_text).grid(row=9, column=0, padx=16, pady=4, sticky="ew")
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)
        ctk.CTkButton(footer, text="← 上一页", width=100, command=lambda: self.change_page(-1)).pack(side="left", padx=5)
        self.page_label = ctk.CTkLabel(footer, text="尚未扫描输入目录")
        self.page_label.pack(side="left", padx=15)
        ctk.CTkButton(footer, text="下一页 →", width=100, command=lambda: self.change_page(1)).pack(side="left", padx=5)
        self.export_status = ctk.CTkLabel(footer, text="")
        self.export_status.pack(side="right", padx=14)

    def save_draft(self):
        if self.current is not None:
            self.drafts[self.app.relative(self.current)] = self.prompt.get("1.0", "end-1c")

    def history(self):
        if self.current is None or not self.app.idle_required():
            return
        source = self.current
        store = self.app.task_store()
        versions = store.versions(Path(self.app.input_var.get()), source)
        if not versions:
            messagebox.showinfo('历史版本', '此页还没有成功保存的版本。', parent=self)
            return
        window = ctk.CTkToplevel(self)
        window.title('选择历史版本 · ' + source.name)
        window.geometry('900x720')
        window.transient(self.app)
        window.grab_set()
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(1, weight=1)
        active = self.app.records.get(self.app.relative(source), {}).get('display_output')
        labels = {f"版本 {len(versions)-i} · {r.get('updated','')[:19]}" + (' · 当前采用' if r['output'] == active else ''): (key,r)
                  for i,(key,r) in enumerate(versions)}
        selection = tk.StringVar(value=next(iter(labels)))
        preview = ctk.CTkLabel(window, text='')
        preview.grid(row=1,column=0,sticky='nsew',padx=12,pady=8)
        detail = ctk.CTkLabel(window,text='',wraplength=850,justify='left')
        detail.grid(row=2,column=0,sticky='ew',padx=12)
        def show(label):
            _, record = labels[label]
            try:
                image = thumbnail(store.output_path(record),(820,480))
                window.photo = ImageTk.PhotoImage(image)
                preview.configure(image=window.photo,text='')
            except Exception:
                preview.configure(image=None,text='版本文件缺失或无法显示，采用时会再次校验。')
            detail.configure(text='本版提示词：' + record.get('parameters',{}).get('prompt','保留原图 / 无提示词'))
        ctk.CTkOptionMenu(window,variable=selection,values=list(labels),command=show).grid(row=0,column=0,sticky='ew',padx=12,pady=12)
        def adopt():
            if self.app.choose_version(source,labels[selection.get()][0]):
                window.destroy()
        ctk.CTkButton(window,text='采用所选版本（不重新生成）',command=adopt).grid(row=3,column=0,pady=12)
        show(selection.get())

    def select(self, source: Path, reload=True):
        if self.current != source:
            self.save_draft()
            self.current = source
            record = self.app.records.get(self.app.relative(source), {})
            saved_prompt = record.get("parameters", {}).get("prompt") if record.get("status") == "failed" else record.get("display_prompt")
            text = self.drafts.get(self.app.relative(source)) or saved_prompt or record.get("parameters", {}).get("prompt") or self.app.prompt.get("1.0", "end-1c") or DEFAULT_PROMPT
            self.prompt.delete("1.0", "end")
            self.prompt.insert("1.0", text)
        record = self.app.records.get(self.app.relative(source), {})
        self.title.configure(text=self.app.relative(source))
        self.notice.configure(text=record.get("error") or ('此页保留原图，批量续跑不调用 API。点击重新生成可恢复上色。' if record.get('parameters', {}).get('use_original') else "提示词仅作用于本页。即使提示词相同，也会发起新生成。"))
        i = self.app.files.index(source)
        self.app.tree.focus(str(i))
        self.app.tree.selection_set(str(i))
        self.app.tree.see(str(i))
        if reload:
            self.page = i
            self.refresh()

    def regenerate(self):
        if self.current is None:
            return
        self.save_draft()
        self.app.start_single(self.current, self.prompt.get("1.0", "end-1c").strip())

    def open_detail(self):
        if self.current:
            self.app.tabs.set("上色工作台")
            self.app.show_preview()

    def change_page(self, delta):
        count = max(1, (len(self.app.files) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page = max(0, min(count - 1, self.page + delta))
        self.refresh()

    def keyboard_page(self, event, delta):
        if self.app.tabs.get() != "整批监修 / 单页修改":
            return
        if isinstance(event.widget, (tk.Text, tk.Entry)):
            return
        self.change_page(delta)
        return "break"

    def refresh(self):
        self.token += 1
        token = self.token
        if self.future:
            self.future.cancel()
        count = len(self.app.files)
        self.page = min(self.page, max(0,count-1))
        self.page_label.configure(text=f"第 {self.page+1 if count else 0} / {count} 页 · 左右键翻页，滚轮滚动，拖拽平移")
        if not count:
            self.current = None
            for canvas in self.canvases:
                canvas.delete("all")
            self.photos = []
            return
        source = self.app.files[self.page]
        self.select(source, reload=False)
        paths = (source, self.app.result_for(source))
        sizes = [(max(200,c.winfo_width()-12),max(250,c.winfo_height()-12)) for c in self.canvases]
        zoom = self.zoom.get()
        events = self.app.events
        neighbors = [self.app.files[i] for i in (self.page-1,self.page+1) if 0 <= i < count]
        neighbor_pairs = [(p,self.app.result_for(p)) for p in neighbors]
        def cached_image(path, fit):
            if path is None:
                return None
            stat = path.stat()
            key = (str(path),stat.st_mtime_ns,stat.st_size,fit,zoom)
            if key not in self.image_cache:
                target = fit
                if zoom != '适应窗口':
                    with Image.open(path) as image:
                        scale = min(float(zoom[:-1])/100,5000/max(image.size))
                        target = (max(1,int(image.width*scale)),max(1,int(image.height*scale)))
                self.image_cache[key] = thumbnail(path,target)
            self.image_cache.move_to_end(key)
            result = self.image_cache[key]
            while len(self.image_cache) > 6:
                self.image_cache.popitem(last=False)
            return result
        def load():
            images = []
            for path, fit in zip(paths,sizes):
                try:
                    images.append(cached_image(path,fit))
                except Exception:
                    images.append(None)
            events.put(("gallery", {"token":token,"images":images}))
            if zoom == '适应窗口':
                for pair in neighbor_pairs:
                    if token != self.token:
                        break
                    for path,fit in zip(pair,sizes):
                        try:
                            cached_image(path,fit)
                        except Exception:
                            pass
        self.future = self.app.preview_pool.submit(load)

    def loaded(self, payload):
        if payload["token"] != self.token:
            return
        self.photos = []
        for canvas, image in zip(self.canvases,payload["images"]):
            canvas.delete("all")
            canvas.xview_moveto(0)
            canvas.yview_moveto(0)
            if image is not None:
                photo = ImageTk.PhotoImage(image, master=canvas)
                self.photos.append(photo)
                canvas.create_image(0,0,image=photo,anchor="nw")
                canvas.configure(scrollregion=(0,0,image.width,image.height))
            else:
                canvas.create_text(20,30,text="未上色 / 图片不可用",anchor="nw")
        scroll = getattr(self.app,'pending_scroll',{}).pop('gallery',None)
        if scroll:
            for canvas,(x,y) in zip(self.canvases,scroll):
                canvas.xview_moveto(x);canvas.yview_moveto(y)
