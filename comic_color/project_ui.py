"""Project home and workspace persistence; no image work on home startup."""
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .config import DEFAULT_PROMPT, ORIGINAL
from .projects import EDIT_SETTINGS, Project, ScopedEvents, recent_projects


class ProjectUI:
    def init_projects(self, legacy=False):
        self.project = None
        self.workspace_generation = 0
        self.event_queue = self.events
        self.events = ScopedEvents(self.event_queue, (None,0))
        self.workspace_timer = None
        self.restore_workspace = None
        self.legacy_workspace = legacy
        self.project_bar = ctk.CTkFrame(self,fg_color='white')
        self.project_bar.grid(row=1,column=0,sticky='new',padx=15)
        self.project_title = ctk.CTkLabel(self.project_bar,text='未打开项目',anchor='w')
        self.project_title.pack(side='left',padx=12,pady=6)
        ctk.CTkButton(self.project_bar,text='项目首页 / 切换项目',command=self.show_project_home).pack(side='right',padx=12)
        # The existing workspace fills row 2; row 1 becomes the project toolbar.
        self.tabs.grid_configure(row=2)
        self.grid_rowconfigure(1,weight=0)
        self.grid_rowconfigure(2,weight=1)
        self.home = ctk.CTkFrame(self,fg_color='#f1f4f9')
        for variable in (self.ratio,self.denoise,self.preserve_text,self.recursive):
            variable.trace_add('write',lambda *_: self.schedule_workspace_save())
        self.prompt.bind('<KeyRelease>',lambda _:self.schedule_workspace_save(),add='+')
        self.bind('<ButtonRelease-1>',lambda _:self.schedule_workspace_save(),add='+')
        self.bind('<MouseWheel>',lambda _:self.schedule_workspace_save(),add='+')
        self.after(1500,self.periodic_workspace_save)
        if not legacy:
            self.input_var.set('')
            self.output_var.set('')
            self.show_project_home()
        else:
            self.project_bar.grid_remove()

    def task_store(self, output=None):
        from .jobs import TaskStore
        return TaskStore(Path(output or self.output_var.get()),self.project)

    def switch_allowed(self):
        editor = getattr(self,'text_editor',None)
        if editor is not None and not editor.closed:
            editor.close()
        if self.busy or (self.worker and self.worker.is_alive()):
            messagebox.showinfo('正在写入结果','请等待上色、文字修复、导出或文字编辑结束后切换项目。')
            return False
        return True

    def clear_workspace(self):
        self.workspace_generation += 1
        self.events = ScopedEvents(self.event_queue,(self.project.id if self.project else None,self.workspace_generation))
        self.preview_token += 1
        self.review_panel.token += 1
        for future in (self.preview_future,self.review_panel.future):
            if future:
                future.cancel()
        self.review_panel.image_cache.clear()
        for canvas in self.canvases+self.review_panel.canvases:
            canvas.delete('all')
        self.preview_photos.clear()
        self.review_panel.photos.clear()
        self.files,self.records = [],{}
        self.tree.delete(*self.tree.get_children())
        self.review_panel.current = None
        self.review_panel.drafts = {}
        self.review_panel.page = 0
        self.scanning = False
        self.restore_workspace = None

    def show_project_home(self):
        if not self.switch_allowed():
            return
        self.save_workspace()
        self.project = None
        self.clear_workspace()
        self.tabs.grid_remove()
        self.project_bar.grid_remove()
        self.home.grid(row=1,column=0,rowspan=2,sticky='nsew',padx=30,pady=20)
        for widget in self.home.winfo_children():
            widget.destroy()
        ctk.CTkLabel(self.home,text='项目工作区',font=ctk.CTkFont(size=25,weight='bold')).pack(anchor='w',pady=12)
        buttons = ctk.CTkFrame(self.home,fg_color='transparent')
        buttons.pack(fill='x',pady=10)
        ctk.CTkButton(buttons,text='新建项目 / 接管旧漫画',command=self.create_project_dialog).pack(side='left',padx=5)
        ctk.CTkButton(buttons,text='打开项目',command=self.open_project_dialog).pack(side='left',padx=5)
        ctk.CTkLabel(self.home,text='最近项目 · 打开前不扫描图片，不检测或融合文字').pack(anchor='w',pady=12)
        recent = ctk.CTkScrollableFrame(self.home)
        recent.pack(fill='both',expand=True)
        try:
            items = recent_projects()
        except ValueError as error:
            self.log(str(error)); items = []
        for item in items:
            ctk.CTkButton(recent,text=item['name']+'   '+item['path'],anchor='w',
                          command=lambda p=item['path']:self.open_project(p)).pack(fill='x',pady=4,padx=8)

    def create_project_dialog(self):
        folder = filedialog.askdirectory(title='选择新的项目工作区目录（保存 project.json）')
        if not folder:
            return
        source = filedialog.askdirectory(title='选择现有漫画原图目录（不复制图片）')
        if not source:
            return
        use_existing = messagebox.askyesno('结果目录','是否接管已有的上色结果目录？\n选择“否”将使用项目目录中的 results 文件夹。')
        output = filedialog.askdirectory(title='选择已有上色结果目录') if use_existing else str(Path(folder)/'results')
        if not output:
            return
        try:
            settings = {'prompt':DEFAULT_PROMPT,'references':[],'recursive':True,
                        'preserve_text':True,'denoising_strength':.55,'output_ratio':ORIGINAL}
            project = Project.create(folder,source,output,settings)
            self.open_project(project.path)
        except Exception as error:
            messagebox.showerror('创建项目失败',str(error))

    def open_project_dialog(self):
        path = filedialog.askopenfilename(title='打开项目',filetypes=[('漫画项目','project.json')])
        if path:
            self.open_project(path)

    def open_project(self, path):
        if not self.switch_allowed():
            return
        try:
            project = Project(path)
            if not project.input.is_dir() or not project.output.is_dir():
                if not messagebox.askyesno('需要重新定位','项目目录不可用。是否重新选择原图与结果目录并核对内容？'):
                    return
                source = filedialog.askdirectory(title='重新定位原图目录')
                output = filedialog.askdirectory(title='重新定位结果目录') if source else ''
                if not source or not output:
                    return
                project.relocate(source,output)
            project.claim()
            from .jobs import TaskStore
            stored = TaskStore(project.output,project)
            if any(r.get('root') != str(project.input) for r in stored.entries.values()):
                # A whole project tree may have moved while relative paths
                # still resolve. Verify content before rebinding old roots.
                project.relocate(project.input,project.output)
            pages = project.index()
            self.save_workspace()
            self.project = project
            self.clear_workspace()
            self.input_var.set(str(project.input))
            self.output_var.set(str(project.output))
            settings = project.data.get('settings',{})
            self.prompt.delete('1.0','end');self.prompt.insert('1.0',settings.get('prompt',DEFAULT_PROMPT))
            self.references = tuple(project.resolve(p) for p in settings.get('references',[]))
            self.update_references()
            self.ratio.set(settings.get('output_ratio',ORIGINAL))
            self.denoise.set(settings.get('denoising_strength',.55))
            self.preserve_text.set(settings.get('preserve_text',True))
            self.recursive.set(settings.get('recursive',True))
            self.restore_workspace = project.data.get('workspace',{})
            self.review_panel.drafts = dict(self.restore_workspace.get('drafts',{}))
            self.review_panel.zoom.set(self.restore_workspace.get('review_zoom','适应窗口'))
            self.zoom.set(self.restore_workspace.get('zoom','适应窗口'))
            self.home.grid_remove()
            self.tabs.grid()
            self.project_bar.grid()
            self.project_title.configure(text=project.data['name']+' · '+str(project.folder))
            tab = self.restore_workspace.get('tab')
            if tab in ('上色工作台','整批监修 / 单页修改','接口与上色设置'):
                self.tabs.set(tab)
            recent_projects(project)
            if pages:
                self.files = [project.input/p['path'] for p in pages]
                self.records = self.task_store().current(project.input)
                self.populate_pages(restore=True)
            self.refresh()
        except Exception as error:
            messagebox.showerror('打开项目失败',str(error))

    def populate_pages(self, restore=False):
        from .jobs import LABELS
        workspace = self.restore_workspace if restore and self.restore_workspace else self.workspace_values()
        self.tree.delete(*self.tree.get_children())
        for i,path in enumerate(self.files):
            rel = self.relative(path)
            record = self.records.get(rel,{})
            self.tree.insert('', 'end',iid=str(i),values=(rel,LABELS.get(record.get('status'),'待处理')))
        selected = set(workspace.get('selection',[]))
        self.tree.selection_set([str(i) for i,p in enumerate(self.files) if self.relative(p) in selected])
        if (not selected and 'selection' not in workspace) or (not restore and self.legacy_workspace):
            self.tree.selection_set(self.tree.get_children())
        batch_selection = self.tree.selection()
        page = workspace.get('page')
        index = next((i for i,p in enumerate(self.files) if self.relative(p) == page),0)
        self.review_panel.page = index
        if self.files:
            self.tree.focus(str(index))
        self.selection_changed()
        self.review_panel.refresh()
        # Selecting the displayed page must not discard the persisted batch selection.
        self.tree.selection_set(batch_selection)
        if restore:
            self.pending_scroll = workspace.get('scroll',{})
        self.restore_workspace = None

    def workspace_values(self):
        self.review_panel.save_draft()
        page = self.review_panel.current
        return {'tab':self.tabs.get(),'page':self.relative(page) if page else None,
                'selection':[self.relative(self.files[int(i)]) for i in self.tree.selection() if int(i)<len(self.files)],
                'drafts':dict(self.review_panel.drafts),'zoom':self.zoom.get(),'review_zoom':self.review_panel.zoom.get(),
                'scroll':{name:[(c.xview()[0],c.yview()[0]) for c in canvases] for name,canvases in
                          (('preview',self.canvases),('gallery',self.review_panel.canvases))}}

    def save_workspace(self):
        if self.project is None:
            return
        settings = {k:v for k,v in self.collect_settings().items() if k in EDIT_SETTINGS}
        settings['references'] = [self.project.encode(p) for p in self.references]
        workspace = self.workspace_values()
        if self.project.data.get('settings') != settings or self.project.data.get('workspace') != workspace:
            self.project.data.update(settings=settings,workspace=workspace)
            self.project.save()

    def schedule_workspace_save(self):
        if self.workspace_timer:
            self.after_cancel(self.workspace_timer)
        self.workspace_timer = self.after(600,self.persist_workspace)

    def persist_workspace(self):
        self.workspace_timer = None
        try:
            self.save_workspace()
        except Exception as error:
            self.log('保存项目失败：'+str(error))

    def periodic_workspace_save(self):
        if not self.app_closed:
            self.persist_workspace()
            self.after(1500,self.periodic_workspace_save)
