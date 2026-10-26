import copy
import json
import queue
from pathlib import Path

from PIL import Image
import pytest
from comic_color.projects import Project, ScopedEvents
from comic_color.jobs import TaskStore
from comic_color.config import atomic_json
from comic_color.imaging import fingerprint
from comic_color.text_protection import cache_path


def make_project(tmp_path, name):
    source = tmp_path/name/'input';source.mkdir(parents=True)
    Image.new('RGB',(80,100),'white').save(source/'page.png')
    return Project.create(tmp_path/name/'workspace',source,tmp_path/name/'output',{'prompt':name,'api_key':'secret'})


def test_projects_isolate_state_and_original_choices(tmp_path):
    a,b = make_project(tmp_path,'a'),make_project(tmp_path,'b')
    sa,sb = TaskStore(a.output,a),TaskStore(b.output,b)
    ra = sa.use_original(a.input,a.input/'page.png')
    assert sa.original_selected(a.input,a.input/'page.png')
    assert not sb.original_selected(b.input,b.input/'page.png')
    assert not sb.current(b.input)
    assert 'api_key' not in a.path.read_text()
    assert not (a.output/'.comic-color/tasks.json').exists()
    assert fingerprint(sa.output_path(ra)) == ra['source_hash']
    ref,path = sa.asset('raw/test.png')
    assert path.is_relative_to(a.state)
    assert sa.output_path({'output':ref}) == path
    with pytest.raises(ValueError):
        sa.output_path({'output':'project://../../outside'})
    with pytest.raises(ValueError):
        Project.create(tmp_path/'other',b.input,a.output)


def test_legacy_import_keeps_files_and_manual_edits(tmp_path):
    root = tmp_path/'input';root.mkdir()
    source = root/'page.png';Image.new('RGB',(80,100),'white').save(source)
    output = tmp_path/'out'
    old = TaskStore(output);record = old.use_original(root,source)
    legacy = old.path.read_bytes()
    atomic_json(cache_path(output,source),{'algorithm':'old','source_hash':fingerprint(source),'added':[[1,2,10,20]],'removed':[]})
    project = Project.create(tmp_path/'project',root,output)
    assert old.path.read_bytes() == legacy
    store = TaskStore(output,project)
    assert store.original_selected(root,source)
    assert store.current(root)['page.png']['display_output'] == record['output']
    imported = json.loads(cache_path(project.folder,source).read_text())
    assert imported['added'] == [[1,2,10,20]] and imported['algorithm'] == 'imported-manual-edits'
    assert not (project.state/'pages.json').exists()
    files,_ = project.scan()
    assert files == [source]
    assert project.index()[0]['path'] == 'page.png'


def test_scoped_events_keep_dispatch_generation():
    q = queue.Queue()
    old,new = ScopedEvents(q,('a',1)),ScopedEvents(q,('b',2))
    old.put(('scan',{'files':[]}));new.put(('scan',{'files':[]}))
    assert q.get()[1]['_scope'] != new.scope
    assert q.get()[1]['_scope'] == new.scope


def test_relocate_rejects_different_content(tmp_path):
    p = make_project(tmp_path,'a');store = TaskStore(p.output,p)
    store.use_original(p.input,p.input/'page.png')
    other = tmp_path/'other';other.mkdir()
    Image.new('RGB',(80,100),'black').save(other/'page.png')
    previous = p.path.read_bytes()
    with pytest.raises(ValueError,match='不一致'):
        p.relocate(other,p.output)
    assert p.path.read_bytes() == previous

def test_model_verification_reuses_unchanged_files_and_rejects_changes(tmp_path,monkeypatch):
    import hashlib
    from comic_color import comic_models as models
    folder = tmp_path/'models';folder.mkdir()
    manifest,expected = {},{}
    for name in ('ctd','bubble'):
        content = name.encode()
        (folder/(name+'.onnx')).write_bytes(content)
        expected[name] = hashlib.sha256(content).hexdigest()
        manifest[name] = {'file':name+'.onnx','sha256':expected[name],'revision':'test'}
    atomic_json(folder/'manifest.json',manifest)
    monkeypatch.setattr(models,'MODEL_DIR',folder)
    monkeypatch.setattr(models,'EXPECTED',expected)
    monkeypatch.setattr(models,'_verified',{})
    calls = []
    def digest(path):
        calls.append(path)
        return fingerprint(path)
    monkeypatch.setattr(models,'fingerprint',digest)
    models.model_manifest();models.model_manifest()
    assert len(calls)==2
    (folder/'ctd.onnx').write_bytes(b'broken')
    with pytest.raises(ValueError,match='校验失败'):
        models.model_manifest()

def test_relocate_retains_manual_regions_and_active_result(tmp_path):
    import shutil
    p = make_project(tmp_path,'a')
    store = TaskStore(p.output,p)
    source = p.input/'page.png'
    store.use_original(p.input,source)
    atomic_json(cache_path(p.folder,source),{'algorithm':'old','source_hash':fingerprint(source),'added':[[1,2,10,20]]})
    target = tmp_path/'new-input';shutil.copytree(p.input,target)
    p.relocate(target,p.output)
    current = TaskStore(p.output,p).current(target)
    assert current['page.png']['display_output']
    assert TaskStore(p.output,p).original_selected(target,target/'page.png')
    edits = json.loads(cache_path(p.folder,target/'page.png').read_text())
    assert edits['added'] == [[1,2,10,20]]
