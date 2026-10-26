"""Pinned, CPU-only comic models. No downloads or torch imports at runtime."""
from pathlib import Path
import json
import threading
import time
import logging

import cv2
import numpy as np

from .imaging import fingerprint

MODEL_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'models'
_engine = None
_lock = threading.RLock()
_verified = {}
EXPECTED = {'ctd':'1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f',
            'bubble':'68e7e1dc2fa3128c9b014074fe6b7c3f3d94a391b378e63fd51778d9dfd5f4ea'}


def model_manifest():
    started = time.perf_counter()
    try:
        data = json.loads((MODEL_DIR / 'manifest.json').read_text(encoding='utf-8'))
        for name in ('ctd', 'bubble'):
            item = data[name]
            path = MODEL_DIR / item['file']
            if item['sha256'] != EXPECTED[name] or path.parent != MODEL_DIR or not path.is_file():
                raise ValueError(name)
            stat = path.stat()
            key = (str(path.resolve()),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,item['sha256'])
            with _lock:
                if key not in _verified:
                    if fingerprint(path) != item['sha256']:
                        raise ValueError(name)
                    _verified[key] = True
        logging.getLogger(__name__).info('model validation %.3fs (verified files reused)',time.perf_counter()-started)
        return data
    except (OSError, ValueError, KeyError) as error:
        raise ValueError('本地漫画检测模型缺失或校验失败，请使用完整发布包或运行 tools/prepare_comic_models.py') from error


def letterbox(image, side, centered=False, color=0):
    h, w = image.shape[:2]
    scale = min(side / w, side / h)
    nw, nh = round(w * scale), round(h * scale)
    left = (side - nw) // 2 if centered else 0
    top = (side - nh) // 2 if centered else 0
    view = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    view = cv2.copyMakeBorder(view, top, side - nh - top, left, side - nw - left, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return view, scale, left, top, nw, nh


def nms(items, threshold=.4):
    if not items:
        return []
    boxes = [[b['box'][0], b['box'][1], b['box'][2]-b['box'][0], b['box'][3]-b['box'][1]] for b in items]
    indices = cv2.dnn.NMSBoxes(boxes, [float(b['score']) for b in items], .0, threshold)
    return [items[int(i)] for i in np.asarray(indices).reshape(-1)]


class ComicModels:
    def __init__(self):
        import onnxruntime as ort
        started = time.perf_counter()
        cv2.setNumThreads(2)
        self.manifest = model_manifest()
        self.ctd = cv2.dnn.readNetFromONNX(str(MODEL_DIR / self.manifest['ctd']['file']))
        self.ctd.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.ctd.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.bubble = ort.InferenceSession(str(MODEL_DIR / self.manifest['bubble']['file']), options, providers=['CPUExecutionProvider'])
        self.load_seconds = time.perf_counter() - started
        logging.getLogger(__name__).info('model load %.3fs',self.load_seconds)

    def text_tile(self, image):
        h, w = image.shape[:2]
        view, scale, left, top, nw, nh = letterbox(image, 1024)
        # CTD's OpenCV path consumes RGB, with right/bottom black padding.
        self.ctd.setInput(cv2.dnn.blobFromImage(view, 1/255., (1024, 1024)))
        outputs = self.ctd.forward(self.ctd.getUnconnectedOutLayersNames())
        predictions = next((x for x in outputs if x.ndim == 3), None)
        maps = [x for x in outputs if x.ndim == 4]
        stroke = next((x for x in maps if x.shape[1] == 1), None)
        lines = next((x for x in maps if x.shape[1] == 2), None)
        if predictions is None or stroke is None or lines is None:
            raise ValueError('CTD 模型输出与固定版本不兼容')
        def restore(probability):
            full = cv2.resize(probability, (1024, 1024))[:nh, :nw]
            return cv2.resize(full, (w, h))
        probability = restore(stroke[0, 0])
        line_probability = restore(lines[0, 0])
        blocks = []
        for row in predictions[0]:
            score = float(row[4] * row[5:].max())
            if score < .35:
                continue
            cx, cy, bw, bh = row[:4] / scale
            box = [max(0, int(cx-bw/2)), max(0, int(cy-bh/2)), min(w, int(cx+bw/2)), min(h, int(cy+bh/2))]
            if box[2] > box[0] and box[3] > box[1]:
                blocks.append({'box': box, 'score': score})
        polygons = []
        contours, _ = cv2.findContours((line_probability > .3).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 3:
                continue
            rect = cv2.minAreaRect(contour)
            if min(rect[1]) < 1:
                continue
            # Expand the DB line map slightly; final masks still follow strokes.
            rect = (rect[0], (rect[1][0]+4, rect[1][1]+4), rect[2])
            polygons.append(np.rint(cv2.boxPoints(rect)).astype(int).tolist())
        return probability, polygons, nms(blocks)

    def text(self, image):
        h, w = image.shape[:2]
        probability = np.zeros((h, w), np.float32)
        polygons, blocks = [], []
        # Overlap and final anchored tiles avoid thin partial tiles at page edges.
        def starts(length):
            return sorted(set(list(range(0, max(1, length-1024), 832)) + [max(0, length-1024)]))
        for y in starts(h):
            for x in starts(w):
                tile = image[y:y+1024, x:x+1024]
                local, lines, found = self.text_tile(tile)
                probability[y:y+tile.shape[0], x:x+tile.shape[1]] = np.maximum(probability[y:y+tile.shape[0], x:x+tile.shape[1]], local)
                polygons.extend((np.asarray(p)+[x, y]).tolist() for p in lines)
                blocks.extend({**b, 'box': (np.asarray(b['box'])+[x,y,x,y]).tolist()} for b in found)
        return probability, polygons, nms(blocks)

    def bubble_tile(self, image):
        side = int(self.bubble.get_inputs()[0].shape[-1])
        view, scale, left, top, _, _ = letterbox(image, side, centered=True, color=114)
        tensor = np.ascontiguousarray(view.transpose(2,0,1)[None], dtype=np.float32) / 255
        output = self.bubble.run(None, {self.bubble.get_inputs()[0].name: tensor})[0][0].T
        h, w = image.shape[:2]
        items = []
        for row in output:
            category = int(row[4:].argmax())
            score = float(row[4+category])
            if category not in (0, 1) or score < .55:
                continue
            cx, cy, bw, bh = row[:4]
            box = [max(0, int((cx-bw/2-left)/scale)), max(0, int((cy-bh/2-top)/scale)), min(w, int((cx+bw/2-left)/scale)+1), min(h, int((cy+bh/2-top)/scale)+1)]
            if box[2] > box[0] and box[3] > box[1]:
                items.append({'box': box, 'score': score, 'category': category})
        return nms(items)

    def bubbles(self, image):
        h,w = image.shape[:2]
        items = self.bubble_tile(image)
        if max(h,w)>1800 or max(h,w)/min(h,w)>2:
            def starts(length):
                return sorted(set(list(range(0,max(1,length-1280),1024))+[max(0,length-1280)]))
            for y in starts(h):
                for x in starts(w):
                    for item in self.bubble_tile(image[y:y+1280,x:x+1280]):
                        items.append({**item,'box':(np.asarray(item['box'])+[x,y,x,y]).tolist()})
        return nms(items)


def engine():
    global _engine
    with _lock:
        if _engine is None:
            _engine = ComicModels()
        return _engine


def infer(image, include_bubbles=True):
    # OpenCV Net and ORT sessions share a serialized model instance.
    with _lock:
        model = engine()
        probability, lines, blocks = model.text(image)
        bubbles = model.bubbles(image) if include_bubbles else []
        return probability, lines, blocks, bubbles
