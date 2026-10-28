from types import SimpleNamespace
from pathlib import Path

import pytest

from comic_color.gui import ColoringApp
from comic_color.jobs import TaskStore
from test_review_pdf import FakeApi, picture, run


def test_choose_old_version_survives_restart_and_checks_integrity(tmp_path):
    root, out = tmp_path/'in', tmp_path/'out'
    root.mkdir()
    page = root/'page.png'
    page.write_bytes(picture('white'))
    api = FakeApi()
    store = run(root,out,[page],api)
    old_id, old = store.versions(root,page)[0]
    store = run(root,out,[page],api,'red',force=True)
    latest = store.current(root)[page.name]['display_output']
    assert latest != old['output']
    store.use_version(root,page,old_id)
    assert TaskStore(out).current(root)[page.name]['display_output'] == old['output']
    assert len(api.calls) == 2
    (out/old['output']).write_bytes(picture('green'))
    with pytest.raises(ValueError,match='缺失或被修改'):
        store.use_version(root,page,old_id)
    assert len(store.entries) == 2


def test_history_cannot_select_another_page_or_changed_source(tmp_path):
    root, out = tmp_path/'in', tmp_path/'out'
    root.mkdir()
    pages = [root/'one.png', root/'two.png']
    for page in pages:
        page.write_bytes(picture('white'))
    store = run(root,out,pages,FakeApi())
    identity,_ = store.versions(root,pages[0])[0]
    with pytest.raises(ValueError,match='不属于当前页'):
        store.use_version(root,pages[1],identity)
    pages[0].write_bytes(picture('green'))
    with pytest.raises(ValueError,match='原稿已经改变'):
        store.use_version(root,pages[0],identity)


def test_queue_waits_for_worker_then_runs_in_order():
    calls = []
    worker = SimpleNamespace(is_alive=lambda: True)
    app = SimpleNamespace(closing=False,busy=False,scanning=False,worker=worker,
                          single_queue=['first','second'],next_single=lambda: None,
                          after=lambda *args: calls.append('wait'),
                          update_single_queue=lambda: None,
                          launch_generation=lambda cfg: calls.append(cfg))
    ColoringApp.next_single(app)
    assert calls == ['wait'] and app.single_queue == ['first','second']
    worker.is_alive = lambda: False
    ColoringApp.next_single(app)
    assert calls == ['wait','first'] and app.single_queue == ['second']
    ColoringApp.next_single(app)
    assert calls[-1] == 'second' and not app.single_queue


def test_stop_clears_pending_requests_but_stops_worker_safely():
    calls = []
    widget = SimpleNamespace(configure=lambda **kw: None)
    app = SimpleNamespace(worker=SimpleNamespace(stop=lambda: calls.append('stop')),
                          clear_single_queue=lambda: calls.append('clear'),
                          stop_button=widget,pause_button=widget,status=widget)
    ColoringApp.stop(app)
    assert calls == ['clear','stop']
