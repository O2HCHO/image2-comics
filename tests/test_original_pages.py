import queue
from PIL import Image
import numpy as np
import cv2
from comic_color.api import ApiConfig
from comic_color.jobs import TaskStore, BatchWorker, JobConfig
from comic_color.review import export_pages
from comic_color.text_geometry import strokes, rectangle_mask


def test_original_page_exact_copy_resume_and_export(tmp_path):
    root = tmp_path/'in'; root.mkdir()
    output = tmp_path/'out'
    source = root/'page.jpg'
    Image.new('RGB',(80,90),'gray').save(source)
    store = TaskStore(output)
    original = store.use_original(root,source)
    assert store.output_path(original).read_bytes() == source.read_bytes()
    class NoApi:
        def color(self,*args):
            raise AssertionError('Original page must never request API')
    config = JobConfig(root,output,(source,),ApiConfig('https://test/images/edits','key','model','prompt',preserve_text=True),delay=0)
    events = queue.Queue()
    BatchWorker(config,events,NoApi()).run()
    assert not any(kind == 'error' for kind,payload in list(events.queue))
    store = TaskStore(output)
    assert len(store.entries) == 1
    pages,missing = export_pages(root,output,[source],store.current(root))
    assert not missing and pages[0].read_bytes() == source.read_bytes()
    Image.new('RGB',(80,90),'white').save(source)
    assert not store.original_selected(root,source)


def test_polygon_alone_cannot_select_background_line():
    image = np.full((100,150,3),255,np.uint8)
    cv2.line(image,(20,45),(120,45),(0,0,0),2)
    allowed = rectangle_mask(image.shape,[[5,5,145,95]])
    polygon = [[10,30],[130,30],[130,60],[10,60]]
    probability = np.zeros(image.shape[:2],np.float32)
    probability[40:50,20:45] = .95
    core,_ = strokes(image,probability,allowed,[polygon])
    assert not core.any()
    # Even an overconfident model should not select a long straight speed line.
    probability[:] = .99
    core,_ = strokes(image,probability,allowed,[polygon])
    assert not core.any()
