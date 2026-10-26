from io import BytesIO
from pathlib import Path
from dataclasses import replace
import queue

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from comic_color.config import ORIGINAL, atomic_json
from comic_color.imaging import fingerprint
from comic_color.text_protection import VERSION, NeedsReview, align, cache_path, detection, fuse, make_mask, png, regions, rgb
from comic_color.jobs import TaskStore, BatchWorker, JobConfig
from comic_color.api import ApiConfig
from comic_color.text_jobs import repair_page


def drawing():
    rng = np.random.default_rng(42)
    a = np.full((700, 500, 3), 255, np.uint8)
    for _ in range(170):
        x, y = rng.integers([0, 0], [490, 690])
        cv2.circle(a, (int(x), int(y)), int(rng.integers(3, 14)), (30, 30, 30), 2)
    cv2.rectangle(a, (30, 60), (190, 200), (255,255,255), -1)
    cv2.putText(a, 'TEXT', (48, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return a


def metadata(source):
    return {'source_hash': fingerprint(source), 'algorithm': VERSION, 'size': [500,700],
            'automatic': [], 'added': [[40,90,175,145]], 'removed': [], 'expand': 1, 'feather': 1}


@pytest.fixture(autouse=True)
def deterministic_models(monkeypatch, request):
    if 'bundled' in request.node.name:
        return
    from comic_color import comic_models
    def infer(image, include_bubbles=True):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return (gray < 100).astype(np.float32), [], [], []
    monkeypatch.setattr(comic_models, 'infer', infer)
    monkeypatch.setattr(comic_models, 'model_manifest', lambda: {k:{'sha256':'test','revision':'test'} for k in ('ctd','bubble')})


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / '原稿.png'
    source.write_bytes(png(drawing()))
    return source


def test_mask_follows_strokes_and_never_fills_box():
    image = np.full((100,100,3),255,np.uint8)
    cv2.putText(image,'HI',(32,55),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),1)
    mask = make_mask((100,100), [[25,25,75,70]], 1, 1, image)
    assert mask[image[:,:,0] == 0].min() == 255
    assert mask[30,30] == 0 and mask[68,70] == 0
    assert np.count_nonzero(mask) < 500


@pytest.mark.parametrize('transform', ['same', 'crop', 'rotate', 'scale'])
def test_registration_and_exact_original_text(sample, transform):
    original = rgb(sample)
    colored = original.copy()
    colored[:, :, 0] = np.minimum(colored[:,:,0], 200)
    colored[90:145,40:175] = (200,100,50)
    if transform == 'crop':
        colored = colored[25:675,20:480]
    elif transform == 'rotate':
        colored = cv2.warpAffine(colored, cv2.getRotationMatrix2D((250,350), 2, 1), (500,700), borderValue=(255,255,255))
    elif transform == 'scale':
        colored = cv2.resize(colored, (750,1050))
    content, mask, info = fuse(sample, png(colored), metadata(sample), ORIGINAL)
    result = rgb(content)
    assert result.shape == original.shape
    ink = np.zeros(original.shape[:2],bool)
    ink[90:145,40:175] = original[90:145,40:175,0] == 0
    assert np.array_equal(result[ink], original[ink])
    assert not np.array_equal(result[90,40], original[90,40])
    assert info['edge_score'] > .43


def test_reject_wrong_layout_and_empty_mask(sample):
    with pytest.raises(NeedsReview):
        fuse(sample, png(np.random.default_rng(9).integers(0,255,(700,500,3),dtype=np.uint8)), metadata(sample), ORIGINAL)
    data = metadata(sample)
    data['added'] = []
    data['issues'] = [{'bubble':0,'reason':'疑似漏字'}]
    with pytest.raises(NeedsReview, match='疑似文字'):
        fuse(sample, sample.read_bytes(), data, ORIGINAL)
    data['confirmed_empty'] = True
    assert fuse(sample, sample.read_bytes(), data, ORIGINAL)[2]['regions'] == 0


def test_cache_manual_edits_invalidated_when_source_changes(sample, tmp_path, monkeypatch):
    import comic_color.text_pipeline as module
    calls = []
    def detect(image):
        calls.append(1)
        mask = np.zeros(image.shape[:2],np.uint8)
        return {'automatic':[[40,90,175,145]],'details':[],'bubbles':[],'issues':[]},mask,mask
    monkeypatch.setattr(module, 'detect_page', detect)
    data = detection(sample, tmp_path / 'out')
    data.update(added=[[10,10,20,20]], removed=[0])
    atomic_json(cache_path(tmp_path / 'out', sample), data)
    assert regions(detection(sample, tmp_path / 'out')) == [[10,10,20,20]]
    assert len(calls) == 1
    sample.write_bytes(png(255 - rgb(sample)))
    assert detection(sample, tmp_path / 'out')['added'] == []
    assert len(calls) == 2


def test_raw_retained_and_local_repair_never_regenerates(sample, tmp_path, monkeypatch):
    import comic_color.text_pipeline as module
    def detect(image):
        mask = np.zeros(image.shape[:2],np.uint8)
        return {'automatic':[],'details':[],'bubbles':[],'issues':[{'bubble':0}]},mask,mask
    monkeypatch.setattr(module, 'detect_page', detect)
    class Api:
        def __init__(self): self.calls = 0
        def color(self, source, cfg):
            self.calls += 1
            return source.read_bytes()
    api = Api()
    output = tmp_path / 'out'
    cfg = ApiConfig('https://test.invalid/v1/images/edits', 'key', 'model', 'prompt', preserve_text=True)
    job = JobConfig(sample.parent, output, (sample,), cfg, delay=0)
    BatchWorker(job, queue.Queue(), api).run()
    store = TaskStore(output)
    rec = next(iter(store.entries.values()))
    assert rec['status'] == 'needs_review' and (output / rec['raw_output']).is_file()
    assert not store.data.get('active_outputs')
    BatchWorker(job, queue.Queue(), api).run()
    assert api.calls == 1
    data = detection(sample, output)
    data['added'] = [[40,90,175,145]]
    data['issues'] = []
    atomic_json(cache_path(output, sample), data)
    result = repair_page(sample.parent, output, sample, TaskStore(output))
    assert result['status'] == 'success' and result.get('display_output')
    assert api.calls == 1
    old_output = result['display_output']
    data['added'] = []
    data['issues'] = [{'bubble':0}]
    atomic_json(cache_path(output, sample), data)
    result = repair_page(sample.parent, output, sample, TaskStore(output))
    assert result['status'] == 'needs_review' and result['display_output'] == old_output


def test_local_repair_original_change_rejected(sample, tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    legacy = output / ('colored_' + sample.name)
    legacy.write_bytes(sample.read_bytes())
    # Explicit old manifest identifies the source hash.
    store = TaskStore(output)
    store.entries['old'] = {'root': str(sample.parent.resolve()), 'source': sample.name, 'source_hash': fingerprint(sample),
                            'output': legacy.name, 'output_hash': fingerprint(legacy), 'parameters': {}, 'status': 'success'}
    store.save()
    sample.write_bytes(png(255 - rgb(sample)))
    rec = repair_page(sample.parent, output, sample, store)
    assert rec['status'] == 'needs_review' and '原稿已改变' in rec['error']


def test_closed_white_balloon_does_not_fill_interior():
    image = np.full((300,300,3), 180, np.uint8)
    cv2.ellipse(image, (100,100), (48,70), 0, 0, 360, (0,0,0), 3)
    cv2.ellipse(image, (100,100), (45,67), 0, 0, 360, (255,255,255), -1)
    cv2.putText(image, 'HI', (75,100), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
    mask = make_mask((300,300), [[70,75,120,110]], 2, 2, image)
    assert mask[130,100] == 0
    assert mask[95,80] > 0 or np.any(mask[85:105,75:100])
    assert mask[0,0] == 0 and mask[100,160] == 0


def test_bundled_detector_horizontal_vertical_and_blank(tmp_path):
    from comic_color.text_protection import detect_regions
    page = Image.new('RGB', (700,700), 'white')
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 28)
    draw.ellipse((20,20,280,105),outline='black',width=3)
    draw.ellipse((465,85,590,365),outline='black',width=3)
    draw.text((40,40), '保留原稿文字', font=font, fill='black')
    for i, char in enumerate('こんにちは'):
        draw.text((500,120+i*38), char, font=font, fill='black')
    detected = detect_regions(np.asarray(page))
    assert any(b[0] > 450 and b[1] > 80 for b in detected)
    from comic_color.comic_models import engine
    _, lines, blocks = engine().text(np.asarray(page))
    assert any(b['box'][0] < 100 and b['box'][1] < 100 for b in blocks)
    assert len(lines) >= 2
    assert detect_regions(np.full((300,300,3),255,np.uint8)) == []
    low = page.resize((350,350))
    assert isinstance(detect_regions(np.asarray(low)),list)
