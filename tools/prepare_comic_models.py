"""Build-only YOLO export and tensor parity validation. Never imported by the app."""
from pathlib import Path
import functools
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'build/model-prep/ultralytics')
os.environ['ULTRALYTICS_AUTOINSTALL'] = 'false'


def main():
    import torch
    import numpy as np
    import onnxruntime as ort
    from ultralytics import YOLO
    from comic_color.comic_models import letterbox
    from PIL import Image
    weight = ROOT/'build/model-prep/bubble.pt'
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(weight) != 'eb1329ddc9b83b3e7e057b3f0c31d12b5fbcbc438f5baff20a154aa5b360e177':
        raise ValueError('Bubble checkpoint checksum mismatch')
    if digest(ROOT/'assets/models/comictextdetector.onnx') != '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f':
        raise ValueError('CTD checksum mismatch')
    model = YOLO(str(weight))
    assert model.names == {0:'bubble',1:'narration',2:'other',3:'text',4:'ui'}
    # Explicit legacy exporter keeps the export stable across torch versions.
    torch.onnx.export = functools.partial(torch.onnx.export,dynamo=False)
    exported = Path(model.export(format='onnx',imgsz=512,opset=12,simplify=False,dynamic=False,device='cpu'))
    session = ort.InferenceSession(str(exported),providers=['CPUExecutionProvider'])
    reference = YOLO(str(weight)).model.eval().float()
    samples = list((ROOT/'inputs').rglob('*.png'))[:3]
    tensors = [np.random.default_rng(42).random((1,3,512,512),dtype=np.float32)]
    for sample in samples:
        view,*_ = letterbox(np.array(Image.open(sample).convert('RGB')),512,centered=True,color=114)
        tensors.append(np.ascontiguousarray(view.transpose(2,0,1)[None],dtype=np.float32)/255.)
    errors = []
    for tensor in tensors:
        with torch.no_grad():
            expected = reference(torch.from_numpy(tensor))[0].numpy()
        actual = session.run(None,{session.get_inputs()[0].name:tensor})[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=2e-3)
        # Class choices above confidence cutoff must agree before NMS.
        good = expected[:,4:,:].max(axis=1)>.55
        assert np.array_equal(expected[:,4:,:].argmax(axis=1)[good],actual[:,4:,:].argmax(axis=1)[good])
        errors.append(float(np.abs(actual-expected).max()))
    destination = ROOT/'assets/models/bubble.onnx'
    shutil.copy2(exported,destination)
    manifest = {
        'ctd':{'file':'comictextdetector.onnx','sha256':digest(ROOT/'assets/models/comictextdetector.onnx'),'revision':'440b978563c71b758e31aaa315d100faba1efa2f','source':'https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1'},
        'bubble':{'file':'bubble.onnx','sha256':digest(destination),'checkpoint_sha256':digest(weight),'revision':'1035c7ffcb91ce12f484ace1b541471575911cae','source':'https://github.com/ebhon/YOLO-manga-bubble-detector','export':{'ultralytics':'8.3.0','torch':torch.__version__,'opset':12,'side':512,'parity_max_abs_errors':errors}}}
    (ROOT/'assets/models/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
