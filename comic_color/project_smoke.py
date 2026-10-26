"""Project GUI regression using local fixture files and zero API calls."""
from pathlib import Path
import json
import os
import time
import traceback
from unittest.mock import patch

from PIL import Image


def run(destination):
    from .config import atomic_json
    destination = Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    os.environ['LOCALAPPDATA'] = str(destination/'profile')
    from .projects import Project
    from .gui import ColoringApp
    from .jobs import TaskStore
    app = None
    report = {'success':False,'api_calls':0}
    try:
        projects = []
        for name in ('A','B'):
            root = destination/name/'input';root.mkdir(parents=True,exist_ok=True)
            for i in range(3):
                Image.new('RGB',(120,180),('white' if i%2 else 'gray')).save(root/f'page{i}.png')
            project = Project.create(destination/name/'workspace',root,destination/name/'output',{'prompt':name})
            store = TaskStore(project.output,project)
            store.use_original(root,root/'page0.png')
            projects.append(project)
        with patch('comic_color.gui.load_settings',return_value={}),patch('comic_color.gui.scan_images',side_effect=AssertionError('home scanned images')):
            app = ColoringApp()
            for _ in range(10):app.update();time.sleep(.02)
            assert app.project is None and not app.files and not app.scanning
        def until(predicate):
            deadline = time.monotonic()+15
            while time.monotonic()<deadline:
                app.update()
                if predicate():return
                time.sleep(.02)
            raise AssertionError('GUI timeout')
        a,b = projects
        with patch('comic_color.api.ImageApi.color',side_effect=AssertionError('project operation invoked API')):
            app.open_project(a.path)
            until(lambda:len(app.files)==3 and not app.scanning)
            app.review_panel.select(app.files[1])
            app.review_panel.prompt.insert('end',' draft-A')
            app.tabs.set('整批监修 / 单页修改')
            app.review_panel.zoom.set('100%')
            app.save_workspace()
            old_events = app.events
            app.open_project(b.path)
            until(lambda:len(app.files)==3 and not app.scanning)
            assert app.prompt.get('1.0','end-1c')=='B'
            assert 'draft-A' not in json.dumps(app.review_panel.drafts), 'Previous project draft leaked'
            old_events.put(('scan',{'files':[],'records':{}}))
            app.update();time.sleep(.12);app.update()
            assert len(app.files)==3 and app.project.id==b.id
            app.open_project(a.path)
            until(lambda:len(app.files)==3 and not app.scanning)
            assert app.review_panel.current.name=='page1.png'
            assert 'draft-A' in app.review_panel.prompt.get('1.0','end-1c')
            assert app.review_panel.zoom.get()=='100%'
            assert app.result_for(a.input/'page0.png').is_file()
            app.show_project_home()
            assert not app.files and app.project is None
        report.update(success=True,checks=['home does not scan','legacy adoption','isolated settings','stale callback discarded','current page and draft restored','current result retained','zero generation API calls'])
        try:
            from PIL import ImageGrab
            ImageGrab.grab().save(destination/'home.png')
        except OSError:
            report['screenshot']='desktop capture unavailable'
    except Exception as error:
        report.update(error=str(error),traceback=traceback.format_exc())
    finally:
        if app:
            app.finish_close()
        atomic_json(destination/'report.json',report)
    return report['success']
