from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comic_color.text_protection import detection, fuse, rgb, regions, make_mask, png
from comic_color.config import ORIGINAL, atomic_json
from PIL import Image, ImageOps, ImageDraw

root = Path(__file__).resolve().parent.parent
source = next(p for p in (root / 'inputs').rglob('*009*.png') if 'Last Tour' in str(p))
colored = next((root / 'outputs').rglob(source.name + '.colored-*.png'))
out = root / 'build/text-real'
out.mkdir(parents=True, exist_ok=True)
data = detection(source, out)
print(json.dumps(data, ensure_ascii=False))
mask = make_mask(tuple(data['size']), regions(data), data['expand'], data['feather'], rgb(source))
(out / 'mask.png').write_bytes(png(mask))
original = Image.open(source).convert('RGB')
draw = ImageDraw.Draw(original)
for box in regions(data): draw.rectangle(box, outline='red', width=2)
original.save(out / 'detected.png')
try:
    content, mask, info = fuse(source, colored.read_bytes(), data, ORIGINAL)
    (out / 'fused.png').write_bytes(content)
    atomic_json(out / 'report.json', {'success': True, **info})
    print(json.dumps(info))
except Exception as error:
    atomic_json(out / 'report.json', {'success': False, 'error': str(error)})
    print(str(error))

