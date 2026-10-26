import cv2
import numpy as np
import pytest

from comic_color import comic_models, text_pipeline as pipeline
from comic_color.config import ORIGINAL, atomic_json
from comic_color.imaging import fingerprint
from comic_color.text_protection import NeedsReview, cache_path, png, rgb


@pytest.fixture
def page(tmp_path,monkeypatch):
    image = np.full((300,400,3),220,np.uint8)
    cv2.ellipse(image,(140,150),(90,100),0,0,360,(255,255,255),-1)
    cv2.ellipse(image,(140,150),(90,100),0,0,360,(0,0,0),3)
    cv2.putText(image,'Hi.',(105,150),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
    cv2.putText(image,'OUT',(280,150),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
    source = tmp_path/'source.png';source.write_bytes(png(image))
    models = {k:{'sha256':'fixed','revision':'fixed'} for k in ('ctd','bubble')}
    monkeypatch.setattr(comic_models,'model_manifest',lambda:models)
    def infer(image,include_bubbles=True):
        probability = np.zeros(image.shape[:2],np.float32)
        probability[115:180,85:200] = (image[115:180,85:200].min(axis=2)<150)
        probability[120:160,275:350] = (image[120:160,275:350].min(axis=2)<150)
        return probability,[],[{'box':[280,130,335,155],'score':.99}],[{'box':[48,48,233,253],'score':.99,'category':0}] if include_bubbles else []
    monkeypatch.setattr(comic_models,'infer',infer)
    return source,tmp_path/'out'


def test_only_closed_bubble_and_cache_assets(page):
    source,out = page
    data = pipeline.detection(source,out)
    assert len(data['automatic']) == 1 and data['skipped']
    mask = pipeline.restoration_mask(rgb(source),data)
    assert mask[:,260:].max() == 0
    assert np.count_nonzero(mask) < 1000
    assert all((out/a['path']).is_file() for a in data['assets'].values())


def test_migrate_edits_and_rebuild_corrupt_asset(page):
    source,out = page
    old = {'source_hash':fingerprint(source),'algorithm':'old','automatic':[[90,115,160,170]],'removed':[0],'added':[[280,125,350,165]],'expand':3,'feather':2,'confirmed_empty':True}
    atomic_json(cache_path(out,source),old)
    data = pipeline.detection(source,out)
    assert data['removed']==[0] and data['added']==old['added']
    assert data['expand']==3 and data['confirmed_empty']
    asset = out/data['assets']['strokes']['path']
    asset.write_bytes(b'corrupt')
    fresh = pipeline.detection(source,out)
    assert fingerprint(asset)==fresh['assets']['strokes']['sha256']


def test_shifted_generated_text_cleared_and_only_masks_change(page,monkeypatch):
    source,out = page
    original = rgb(source)
    data = pipeline.detection(source,out)
    generated = original.copy()
    allowed = pipeline.safe_interior(original.shape,data['bubbles'][0])
    generated[allowed>0] = (210,230,250)
    cv2.putText(generated,'BAD',(110,170),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
    monkeypatch.setattr('comic_color.text_protection.align',lambda a,b,c:(b,{'edge_score':1}))
    result,mask,info = pipeline.fuse(source,png(generated),data,ORIGINAL)
    cleanup = rgb(info['cleanup_png'])[:,:,0]
    restore = rgb(mask)[:,:,0]
    combined = (cleanup>0)|(restore>0)
    assert np.array_equal(rgb(result)[~combined],generated[~combined])
    assert np.array_equal(restore,pipeline.restoration_mask(original,data))
    assert cleanup[:,260:].max()==0 and restore[:,260:].max()==0
    ghosts = (generated[:,:,0]<50)&(allowed>0)&(restore==0)
    assert np.min(rgb(result)[ghosts])>170
    assert np.mean(rgb(result)[145:175,100:180,2])>np.mean(rgb(result)[145:175,100:180,0])


def test_complex_background_requires_review(page,monkeypatch):
    source,out = page
    data=pipeline.detection(source,out)
    generated=rgb(source)
    allowed=pipeline.safe_interior(generated.shape,data['bubbles'][0])
    yy,xx=np.indices(allowed.shape)
    generated[(allowed>0)&((xx+yy)%2==0)] = (180,160,140)
    monkeypatch.setattr('comic_color.text_protection.align',lambda a,b,c:(b,{}))
    with pytest.raises(NeedsReview,match='底纹复杂'):
        pipeline.fuse(source,png(generated),data,ORIGINAL)


def test_missing_and_tampered_models_are_errors(tmp_path,monkeypatch):
    monkeypatch.setattr(comic_models,'MODEL_DIR',tmp_path)
    with pytest.raises(ValueError,match='模型缺失或校验失败'):
        comic_models.model_manifest()
    atomic_json(tmp_path/'manifest.json',{k:{'file':k+'.onnx','sha256':'fake'} for k in ('ctd','bubble')})
    for k in ('ctd','bubble'):(tmp_path/(k+'.onnx')).write_bytes(b'invalid')
    with pytest.raises(ValueError,match='模型缺失或校验失败'):
        comic_models.model_manifest()

def test_fusion_stage_cache_skips_models_and_rebuilds_corruption(page,monkeypatch):
    source,out = page
    original = rgb(source)
    data = pipeline.detection(source,out)
    calls = {'align':0,'infer':0}
    infer = comic_models.infer
    def align(a,b,c):
        calls['align'] += 1
        return b,{'edge_score':1}
    def detect(*args,**kwargs):
        calls['infer'] += 1
        return infer(*args,**kwargs)
    monkeypatch.setattr('comic_color.text_protection.align',align)
    monkeypatch.setattr(comic_models,'infer',detect)
    first = pipeline.fuse(source,png(original),data,ORIGINAL)
    second = pipeline.fuse(source,png(original),data,ORIGINAL)
    assert first[:2] == second[:2] and second[2]['cache_hit']
    assert calls == {'align':1,'infer':1}
    changed = dict(data,expand=2)
    pipeline.fuse(source,png(original),changed,ORIGINAL)
    assert calls == {'align':1,'infer':1}, 'mask-only changes must reuse alignment and detection'
    for path in (out/'.comic-color/fusion-cache').rglob('result.png'):
        path.write_bytes(b'corrupt')
    rebuilt = pipeline.fuse(source,png(original),data,ORIGINAL)
    assert first[:2] == rebuilt[:2]
    assert calls == {'align':1,'infer':1}

def test_project_repair_uses_isolated_state_and_private_assets(page,monkeypatch):
    from comic_color.projects import Project
    from comic_color.jobs import TaskStore
    from comic_color.text_jobs import repair_page
    source,out = page
    project = Project.create(source.parent/'workspace',source.parent,out)
    store = TaskStore(out,project)
    generated = out/'generated.png'; generated.write_bytes(source.read_bytes())
    record = {'root':str(source.parent),'source':source.name,'source_hash':fingerprint(source),
              'parameters':{},'output':generated.name,'output_hash':fingerprint(generated),'status':'success','updated':'2020'}
    store.entries['old'] = record;store.activate(source.parent,record);store.save()
    monkeypatch.setattr('comic_color.text_protection.align',lambda a,b,c:(b,{'edge_score':1}))
    result = repair_page(source.parent,out,source,store)
    assert result['status']=='success',result.get('error')
    assert result['raw_output'].startswith('project://')
    assert store.output_path({'output':result['raw_output']}).is_relative_to(project.state)
    assert (project.state/'text').is_dir()
    assert not (out/'.comic-color/tasks.json').exists()
    assert not (out/'.comic-color/text').exists()
