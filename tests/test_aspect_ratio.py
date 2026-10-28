from dataclasses import replace
from io import BytesIO
import base64
import math

from PIL import Image
import pytest

from comic_color.api import ImageApi, generation_size
from comic_color.config import JSON_MODE, ORIGINAL
from comic_color.imaging import exact_source_size, image_size, source_ratio_png
from comic_color.jobs import TaskStore
from test_workbench import config, response, run_job, FakeApi


def picture(size, mode='RGB'):
    stream = BytesIO()
    Image.new(mode, size, (30, 60, 90, 120) if mode == 'RGBA' else (30, 60, 90)).save(stream, 'PNG')
    return stream.getvalue()


@pytest.mark.parametrize('original,generated', [
    ((695,1080),(1041,1510)), ((1001,1600),(2048,3072)),
    ((1080,695),(1510,1041)), ((40,60),(20,30)),
    ((695,1080),(1390,2160)), ((1600,1001),(3072,2048)),
])
def test_exact_ratio_without_separate_axis_rounding(original, generated):
    width, height = exact_source_size(original, generated)
    assert width * original[1] == height * original[0]
    assert width >= original[0] and height >= original[1]
    divisor = math.gcd(*original)
    unit = original[0] // divisor, original[1] // divisor
    k = width // unit[0]
    area_error = abs(width * height - generated[0] * generated[1])
    for other in (k-1,k+1):
        if other >= divisor:
            assert area_error <= abs(unit[0] * unit[1] * other * other - generated[0] * generated[1])


def test_normalization_keeps_exact_high_resolution_and_alpha(tmp_path):
    source = tmp_path/'original.png'
    source.write_bytes(picture((695,1080)))
    output = source_ratio_png(picture((1041,1510),'RGBA'),source)
    with Image.open(BytesIO(output)) as image:
        assert image.size == (973,1512)
        assert image.width * 1080 == image.height * 695
        pixel = image.getpixel((0,0))
        assert pixel[3] == 120
        assert all(abs(a-b) <= 2 for a,b in zip(pixel[:3],(30,60,90)))
    assert image_size(source) == (695,1080)


def test_exif_orientation_is_the_source_coordinate_system(tmp_path):
    source = tmp_path/'rotated.jpg'
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB',(60,40),'white').save(source,exif=exif)
    assert image_size(source) == (40,60)
    with Image.open(BytesIO(source_ratio_png(picture((90,100)),source))) as result:
        assert result.width * 60 == result.height * 40


@pytest.mark.parametrize('mode', [config().mode, JSON_MODE])
def test_api_requests_source_ratio_and_corrects_service_reply(tmp_path,monkeypatch,mode):
    source = tmp_path/'source.png'
    source.write_bytes(picture((695,1080)))
    reply = picture((1041,1510))
    def post(url,**kw):
        payload = kw.get('data',kw.get('json'))
        assert '695×1080' in payload['prompt'] and '695:1080' in payload['prompt']
        if mode != JSON_MODE:
            assert payload['size'] == '992x1536'
        return response({'data':[{'b64_json':base64.b64encode(reply).decode()}]})
    monkeypatch.setattr('requests.post',post)
    result = ImageApi().color(source,replace(config(),mode=mode,model='gpt-image-2.5-sunburst'))
    with Image.open(BytesIO(result)) as image:
        assert image.width * 1080 == image.height * 695


def test_text_protection_retains_actual_service_image_for_registration(tmp_path,monkeypatch):
    source = tmp_path/'source.png'
    source.write_bytes(picture((40,60)))
    reply = picture((80,100))
    monkeypatch.setattr('requests.post',lambda *a,**k: response({'image':base64.b64encode(reply).decode()}))
    content = ImageApi().color(source,replace(config(),preserve_text=True))
    assert Image.open(BytesIO(content)).size == (80,100)


def test_worker_enforces_ratio_even_for_custom_api_adapter(tmp_path):
    source = tmp_path/'source.png'
    source.write_bytes(picture((40,60)))
    out = tmp_path/'output'
    api = FakeApi(picture((80,100)))
    run_job(source,out,api)
    record = next(iter(TaskStore(out).entries.values()))
    assert record['status'] == 'success'
    assert record['parameters']['aspect_policy'] == 'source-aspect-v1'
    width,height = image_size(out/record['output'])
    assert width * 60 == height * 40
    assert api.calls == 1


def test_unknown_model_gets_no_incompatible_size_field():
    assert generation_size((695,1080),'custom-image-model') is None
    assert generation_size((695,1080),'gpt-image-1.5') is None
    assert generation_size((400,2000),'gpt-image-2.5-flare') is None
    size = generation_size((5000,7000),'gpt-image-2.5-flare')
    w,h = map(int,size.split('x'))
    assert w % 16 == h % 16 == 0
    assert max(w,h) <= 3840 and w*h <= 8_294_400


def test_fixed_canvas_cannot_override_locked_source_ratio():
    with pytest.raises(ValueError,match='锁定原稿长宽比'):
        replace(config(),ratio='1:1（透明补边）').validate()
