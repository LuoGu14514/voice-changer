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
def custom_store(tmp_path_factory):
    """把自定义角色库指到临时文件，别读用户真实的 %APPDATA%。"""
    path = tmp_path_factory.mktemp("custom_voices") / "custom_voices.json"
    mp = pytest.MonkeyPatch()
    mp.setattr(minimax_tts, "CUSTOM_VOICES_PATH", str(path))
    yield path
    mp.undo()


@pytest.fixture(scope="module")
def app(custom_store):
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
    # 启动时 _load_mm_key() 会把配置里的 Key 填进输入框，这里也要清掉
    app.ent_key.delete(0, "end")
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


# ---------------------------------------------------------------- 自定义角色

def test_custom_panel_widgets_exist(app):
    assert hasattr(app, "tree_custom")
    assert hasattr(app, "ent_ref")
    assert hasattr(app, "ent_custom_name")
    assert callable(app.on_save_custom_ref)
    assert callable(app.on_fetch_remote_voices)
    assert callable(app.on_clone_voice)
    assert callable(app.on_design_voice)
    assert callable(app.on_use_custom)
    assert callable(app.on_delete_custom)


def test_custom_role_appears_in_index_dropdown(app, custom_store):
    """存一个 IndexTTS 零样本角色后，IndexTTS 角色下拉里要能看到它。"""
    minimax_tts.add_custom_voice(
        "测试零样本", backend="indextts", ref_audio=__file__, source="local")
    app._refresh_custom_roles()
    values = list(app.cb_char.cget("values"))
    assert "测试零样本" in values
    assert "三月七" in values


def test_custom_role_current_ref_uses_its_audio(app, custom_store, tmp_path):
    ref = tmp_path / "sample.wav"
    ref.write_bytes(b"RIFF")
    minimax_tts.add_custom_voice(
        "取样角色", backend="indextts", ref_audio=str(ref), source="local")
    app._refresh_custom_roles()
    app.cb_char.set("取样角色")
    app._load_characters()
    path, text = app._current_ref()
    assert path == str(ref)
    assert "零样本参考音频" in app.lbl_ref.cget("text")


def test_custom_minimax_role_appears_in_mm_dropdown(app, custom_store):
    minimax_tts.add_custom_voice(
        "云端角色", backend="minimax", voice_id="vc-test-1234", source="import")
    app._refresh_custom_roles()
    labels = list(app.cb_mm_char.cget("values"))
    assert "自定义 · 云端角色" in labels


def test_minimax_rejects_indextts_custom_role(app, custom_store):
    """IndexTTS 自定义角色没有云端音色 ID，不能直接丢给 MiniMax。"""
    minimax_tts.add_custom_voice(
        "本地角色", backend="indextts", ref_audio=__file__, source="local")
    app._refresh_custom_roles()
    app.cb_mm_char.set("自定义 · 本地角色")
    app._show_mm_preset()
    assert "IndexTTS" in app.lbl_mm_preset.cget("text")


def test_custom_tree_lists_roles(app, custom_store):
    minimax_tts.add_custom_voice(
        "树里的角色", backend="minimax", voice_id="vc-tree-1234", source="import")
    app._refresh_custom_roles()
    iids = list(app.tree_custom.get_children())
    assert "树里的角色" in iids
    row = app.tree_custom.item("树里的角色", "values")
    assert row[0] == "树里的角色"
    assert row[1] == "minimax"
    assert row[2] == "vc-tree-1234"


def test_name_from_label_handles_group_prefix():
    assert tts_app.name_from_label("童 · 萝莉") == "萝莉"
    assert tts_app.name_from_label("自定义 · 我的角色") == "我的角色"
    assert tts_app.name_from_label("") == ""


def test_safe_filename_strips_windows_bad_chars():
    assert tts_app.safe_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"
    assert tts_app.safe_filename("") == "role"
    assert tts_app.safe_filename("   ") == "role"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
