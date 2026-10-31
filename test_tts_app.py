# -*- coding: utf-8 -*-
"""tts_app.py 的 GUI 冒烟测试。

跑法：
    python -m pytest test_tts_app.py -v

只做「建窗口 -> 切后端 -> 查控件状态 -> 销毁」，不发任何网络请求，
也不会真的加载 IndexTTS 模型。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

tk = pytest.importorskip("tkinter")

import minimax_tts
import tts_app


@pytest.fixture(scope="module")
def app():
    root = tk.Tk()
    root.withdraw()
    a = tts_app.TTSApp(root)
    yield a
    try:
        root.destroy()
    except Exception:
        pass


def _packed(w):
    """窗口 withdraw 时 winfo_ismapped 恒为 0，改用 winfo_manager 判是否被布局。"""
    return bool(w.winfo_manager())


def test_default_backend_is_minimax(app):
    assert app.backend == tts_app.BACKEND_MINIMAX


def test_minimax_panel_shown_by_default(app):
    assert _packed(app.frm_mm)
    assert not _packed(app.frm_it)


def test_switch_to_indextts(app):
    app.backend_var.set(tts_app.BACKEND_INDextts)
    app._apply_backend()
    assert app.backend == tts_app.BACKEND_INDextts
    assert _packed(app.frm_it)
    assert _packed(app.prm_frame)
    assert not _packed(app.frm_mm)


def test_switch_back_to_minimax(app):
    app.backend_var.set(tts_app.BACKEND_MINIMAX)
    app._apply_backend()
    assert _packed(app.frm_mm)
    assert not _packed(app.frm_it)
    assert not _packed(app.prm_frame)


def test_minimax_role_dropdown_has_30(app):
    labels = list(app.cb_mm_char.cget("values"))
    assert len(labels) == 30
    assert labels[0] == "原 · 原声"


def test_minimax_preset_lookup(app):
    app.cb_mm_char.set("童 · 萝莉")
    app._show_mm_preset()
    text = app.lbl_mm_preset.cget("text")
    assert "萝莉" in text


def test_missing_api_key_raises(app, tmp_path, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr(minimax_tts, "CONFIG_PATH", str(tmp_path / "nope.json"))
    with pytest.raises(minimax_tts.MiniMaxError) as ei:
        app._ensure_client()
    assert "API Key" in str(ei.value)


def test_index_worker_not_autostarted(app):
    """IndexTTS worker 不该在启动时自动拉起（要吃 ~7.5GB 内存）。"""
    assert app.worker is None
    assert not app.worker_loading


def test_playback_fallback_exists(app):
    assert callable(app._play_path)
    assert callable(app.on_stop_play)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
