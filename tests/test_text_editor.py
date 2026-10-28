"""Editor recovery flow without a display server or paid image requests."""
from types import SimpleNamespace
import queue

from comic_color.text_editor import TextEditor


class Widget:
    def __init__(self):
        self.values = {}

    def configure(self, **values):
        self.values.update(values)


def test_apply_requires_preview_of_current_options():
    calls = []
    editor = SimpleNamespace(
        loading=False, verified_options=None, options=lambda: {'expand': 1},
        signature=lambda data: TextEditor.signature(None, data),
        preview=lambda: calls.append('preview'), save=lambda: calls.append('save'),
    )
    TextEditor.apply(editor)
    assert calls == ['preview']


def test_apply_uses_verified_preview_then_saves_and_repairs():
    calls = []
    editor = SimpleNamespace(
        loading=False, verified_options='{"expand": 1}', options=lambda: {'expand': 1},
        signature=lambda data: TextEditor.signature(None, data), source='page.png',
        save=lambda: calls.append('save') or True, close=lambda: calls.append('close'),
        app=SimpleNamespace(repair_text=lambda sources: calls.append(sources)),
    )
    TextEditor.apply(editor)
    assert calls == ['save', 'close', ('page.png',)]


def test_failed_preview_keeps_raw_visible_and_exposes_details():
    events = queue.Queue()
    raw = object()
    events.put(('raw_preview', raw))
    events.put(('error', '对齐失败\n检查气泡位置，重新预览'))
    editor = SimpleNamespace(
        closed=False, loading=True, events=events, data={}, verified_options='old',
        pending_close=False, pending_preview=False,
        status=Widget(), help_button=Widget(), right_label=Widget(),
        draw_right=lambda: None, after=lambda *args: None, poll=lambda: None,
    )
    TextEditor.poll(editor)
    assert editor.right_image is raw
    assert not editor.loading and editor.verified_options is None
    assert editor.help_button.values['state'] == 'normal'
    assert '检查气泡位置' in editor.error_details
    assert '查看原因与修复方法' in editor.status.values['text']
