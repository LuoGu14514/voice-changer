# -*- coding: utf-8 -*-
"""minimax_tts.py 的离线测试。

全部用本地 mock HTTP server，**不需要真实 API Key、不发外网请求**。

运行：
    python -m pytest test_minimax_tts.py -v
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minimax_tts
from minimax_tts import MiniMaxError, MiniMaxTTS

# ---------------------------------------------------------------- mock server

#: 记录收到的请求，供断言
RECORD = {
    "requests": [],       # [(method, path, headers, body_bytes)]
    "responses": {},      # path -> (status, json_body)
    "default": (200, {"base_resp": {"status_code": 0, "status_msg": "success"},
                      "data": {"audio": "00112233"}, "trace_id": "t-1"}),
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 静音
        pass

    def _record(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        RECORD["requests"].append(
            (self.command, self.path, dict(self.headers), body)
        )
        return body

    def _respond(self):
        self._record()
        path = self.path.split("?")[0]
        status, payload = RECORD["responses"].get(path, RECORD["default"])
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = _respond
    do_POST = _respond
    do_DELETE = _respond


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _clear():
    RECORD["requests"].clear()
    RECORD["responses"].clear()
    yield


@pytest.fixture
def client(server):
    return MiniMaxTTS(api_key="test-key-12345678", base_url=server)


def _last(path_suffix):
    for m, p, h, b in reversed(RECORD["requests"]):
        if p.split("?")[0].endswith(path_suffix):
            return m, p, h, b
    raise AssertionError(f"没有收到指向 {path_suffix} 的请求")


# ---------------------------------------------------------------- 预设

def test_preset_count_is_30():
    assert len(minimax_tts.MINIMAX_PRESETS) == 30


def test_preset_groups_cover_all_five():
    groups = {p["group"] for p in minimax_tts.MINIMAX_PRESETS}
    assert groups == set(minimax_tts.PRESET_GROUPS)
    assert groups == {"原", "女", "男", "童", "特效"}


def test_preset_names_unique():
    names = [p["name"] for p in minimax_tts.MINIMAX_PRESETS]
    assert len(names) == len(set(names))


def test_preset_names_cover_original_project_roles():
    """原项目 30 个角色名一个都不能少（分组也必须一致）。"""
    expected = {
        "原声": "原",
        "真女声": "女", "御姐": "女", "客服女": "女", "播音女": "女",
        "少妇": "女", "老奶奶": "女", "萌妹": "女", "嗲嗲": "女",
        "真男声": "男", "大叔": "男", "磁性男": "男", "恶魔": "男",
        "客服男": "男", "播音男": "男", "老人": "男",
        "萝莉": "童", "正太": "童", "小孩": "童", "娃娃音": "童",
        "小猫": "童", "花栗鼠": "童", "汤姆猫": "童",
        "机器人": "特效", "电音女王": "特效", "外星人": "特效", "兽人": "特效",
        "幽灵": "特效", "回声": "特效", "电话音": "特效",
    }
    actual = {p["name"]: p["group"] for p in minimax_tts.MINIMAX_PRESETS}
    assert actual == expected


def test_preset_required_fields():
    for p in minimax_tts.MINIMAX_PRESETS:
        assert p["voice_id"], p["name"]
        assert 0.5 <= p["speed"] <= 2.0, p["name"]
        assert 0 < p["vol"] <= 10, p["name"]
        assert -12 <= p["pitch"] <= 12, p["name"]
        if p["emotion"] is not None:
            assert p["emotion"] in minimax_tts.EMOTIONS, p["name"]


def test_preset_modify_uses_known_sound_effects():
    for p in minimax_tts.MINIMAX_PRESETS:
        mod = p.get("modify")
        if mod and "sound_effects" in mod:
            assert mod["sound_effects"] in minimax_tts.SOUND_EFFECTS, p["name"]


def test_get_preset_and_grouped():
    p = minimax_tts.get_preset("萝莉")
    assert p is not None and p["group"] == "童"
    assert minimax_tts.get_preset("不存在的角色") is None
    groups = minimax_tts.grouped_presets()
    assert [g for g, _ in groups] == list(minimax_tts.PRESET_GROUPS)
    assert sum(len(items) for _, items in groups) == 30


def test_system_voice_lists_nonempty():
    assert len(minimax_tts.SYSTEM_VOICES_ZH) == 34
    assert len(minimax_tts.SYSTEM_VOICES_CANTONESE) == 6
    assert len(minimax_tts.SYSTEM_VOICES_JA) == 15
    # voice_id 不能重复
    ids = [v[0] for v in minimax_tts.SYSTEM_VOICES_ZH]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- 请求体

def test_build_payload_basic_shape(client):
    p = client._build_payload("你好", "Chinese (Mandarin)_Sweet_Lady")
    assert p["text"] == "你好"
    assert p["model"] == minimax_tts.DEFAULT_MODEL
    vs = p["voice_setting"]
    assert vs["voice_id"] == "Chinese (Mandarin)_Sweet_Lady"
    assert vs["speed"] == 1.0 and vs["vol"] == 1.0 and vs["pitch"] == 0
    assert "emotion" not in vs          # 没给就不塞
    au = p["audio_setting"]
    assert au["sample_rate"] == 32000
    assert au["format"] == "mp3"
    assert au["channel"] == 1
    assert au["bitrate"] == 128000
    assert p["output_format"] == "hex"


def test_build_payload_bitrate_only_for_mp3(client):
    p = client._build_payload("hi", "v", fmt="wav")
    assert "bitrate" not in p["audio_setting"]
    p = client._build_payload("hi", "v", fmt="mp3", bitrate=64000)
    assert p["audio_setting"]["bitrate"] == 64000


def test_build_payload_emotion_and_stream(client):
    p = client._build_payload("hi", "v", emotion="happy", stream=True)
    assert p["voice_setting"]["emotion"] == "happy"
    assert p["stream"] is True
    assert p["audio_setting"]["sample_rate"] == 32000


def test_build_payload_optional_blocks(client):
    p = client._build_payload(
        "hi", "v",
        pronunciation_dict={"tone": [{"原音": "你好", "替换": "nihao"}]},
        timbre_weights=[{"voice_id": "a", "weight": 80}],
        voice_modify={"pitch": -30, "sound_effects": "robotic"},
        subtitle_enable=True, subtitle_type="word",
        text_normalization_mode="quality",
        latex_read=True,
    )
    assert p["pronunciation_dict"]["tone"][0]["替换"] == "nihao"
    assert p["timbre_weights"][0]["weight"] == 80
    assert p["voice_modify"]["sound_effects"] == "robotic"
    assert p["subtitle_enable"] is True and p["subtitle_type"] == "word"
    assert p["text_normalization_mode"] == "quality"
    assert p["voice_setting"]["latex_read"] is True


def test_synth_rejects_empty_text(client):
    with pytest.raises(MiniMaxError):
        client.synth("   ", "v")


def test_synth_rejects_too_long_text(client):
    with pytest.raises(MiniMaxError) as ei:
        client.synth("啊" * 10000, "v")
    assert "10000" in str(ei.value)


def test_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr(minimax_tts, "_load_key_from_config", lambda: "")
    with pytest.raises(MiniMaxError):
        MiniMaxTTS()


# ---------------------------------------------------------------- 错误码

def test_check_raises_on_error_code():
    obj = {"base_resp": {"status_code": 1002, "status_msg": "rate limited"},
           "trace_id": "abc"}
    with pytest.raises(MiniMaxError) as ei:
        MiniMaxTTS._check(obj)
    assert ei.value.code == 1002
    assert ei.value.trace_id == "abc"
    assert "rate limited" in str(ei.value)


def test_check_passes_on_zero_code():
    obj = {"base_resp": {"status_code": 0, "status_msg": "success"}}
    assert MiniMaxTTS._check(obj) is obj


def test_error_code_table_known():
    """官方错误码表里的值应该能被识别出来。"""
    known = {0, 1000, 1001, 1002, 1004, 1008, 1013, 1027, 1039, 1042, 1043,
             2013, 2038, 2049, 2201, 2202, 2203, 2204, 2205, 2206}
    for code in known:
        obj = {"base_resp": {"status_code": code, "status_msg": "x"}}
        if code == 0:
            MiniMaxTTS._check(obj)
        else:
            with pytest.raises(MiniMaxError):
                MiniMaxTTS._check(obj)


def test_synth_raises_business_error(client):
    RECORD["responses"]["/v1/t2a_v2"] = (
        200, {"base_resp": {"status_code": 2013, "status_msg": "invalid params"},
              "trace_id": "tt"})
    with pytest.raises(MiniMaxError) as ei:
        client.synth("hi", "v")
    assert ei.value.code == 2013


# ---------------------------------------------------------------- 音频提取

def test_extract_audio_hex():
    obj = {"data": {"audio": "00ff10"}}
    assert MiniMaxTTS._extract_audio(obj) == b"\x00\xff\x10"


def test_extract_audio_missing():
    with pytest.raises(MiniMaxError):
        MiniMaxTTS._extract_audio({"data": {}})


# ---------------------------------------------------------------- 断句

def test_split_sentences_keeps_punctuation():
    out = minimax_tts._split_sentences("你好。今天天气不错！还有呢？")
    assert out == ["你好。", "今天天气不错！", "还有呢？"]


def test_split_sentences_trailing_fragment():
    out = minimax_tts._split_sentences("第一句。第二句尾巴")
    assert out == ["第一句。", "第二句尾巴"]


def test_split_sentences_empty():
    assert minimax_tts._split_sentences("") == []


# ---------------------------------------------------------------- HTTP 交互

def test_synth_posts_correct_payload(client):
    client.synth("你好世界", "voice-1")
    m, p, h, b = _last("/v1/t2a_v2")
    assert m == "POST"
    assert h["Authorization"] == "Bearer test-key-12345678"
    assert json.loads(b)["text"] == "你好世界"


def test_synth_to_file_writes_bytes(client, tmp_path):
    out = tmp_path / "a.wav"
    client.synth_to_file("hi", "v", str(out))
    assert out.read_bytes() == b"\x00\x11\x22\x33"


def test_synth_to_file_creates_parent_dir(client, tmp_path):
    out = tmp_path / "sub" / "dir" / "a.wav"
    client.synth_to_file("hi", "v", str(out))
    assert out.exists()


def test_list_voices(client):
    RECORD["responses"]["/v1/get_voice"] = (
        200, {"base_resp": {"status_code": 0, "status_msg": "success"},
              "system_voice": [{"voice_id": "a"}, {"voice_id": "b"}],
              "voice_cloning": [], "voice_generation": []})
    obj = client.list_voices()
    assert len(obj["system_voice"]) == 2
    m, p, h, b = _last("/v1/get_voice")
    assert json.loads(b)["voice_type"] == "all"


def test_delete_voice(client):
    client.delete_voice("my-voice-id", "voice_cloning")
    m, p, h, b = _last("/v1/delete_voice")
    assert json.loads(b) == {"voice_type": "voice_cloning", "voice_id": "my-voice-id"}


def test_design_voice(client):
    RECORD["responses"]["/v1/voice_design"] = (
        200, {"base_resp": {"status_code": 0, "status_msg": "success"},
              "voice_id": "ttv-voice-1", "trial_audio": "aabb"})
    obj = client.design_voice("温柔女声", "你好")
    assert obj["voice_id"] == "ttv-voice-1"
    m, p, h, b = _last("/v1/voice_design")
    assert json.loads(b)["prompt"] == "温柔女声"


# ---------------------------------------------------------------- multipart

def test_multipart_upload_format(client, tmp_path):
    RECORD["responses"]["/v1/files/upload"] = (
        200, {"base_resp": {"status_code": 0, "status_msg": "success"},
              "file_id": "file-123"})
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    file_id = client.upload_file(str(wav), purpose="voice_clone")
    assert file_id == "file-123"
    m, p, h, b = _last("/v1/files/upload")
    ctype = h["Content-Type"]
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=", 1)[1]
    assert f"--{boundary}".encode() in b
    assert b'Content-Disposition: form-data; name="purpose"' in b
    assert b"voice_clone" in b
    assert b'filename="ref.wav"' in b
    assert b"RIFF" in b
    # 文件内容完整送到了
    assert b"RIFF\x00\x00\x00\x00WAVEfmt " in b


def test_clone_voice_validates_id(client):
    for bad in ("short", "x" * 257, "1starts_with_digit", "ends_with_dash-"):
        with pytest.raises(MiniMaxError):
            client.clone_voice("a.wav", bad)


def test_clone_voice_flow(client, tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFFfake")
    RECORD["responses"]["/v1/files/upload"] = (
        200, {"base_resp": {"status_code": 0, "status_msg": "success"},
              "file_id": "file-123"})
    RECORD["responses"]["/v1/voice_clone"] = (
        200, {"base_resp": {"status_code": 0, "status_msg": "success"},
              "voice_id": "my-clone-voice", "demo_audio": "aabb"})
    obj = client.clone_voice(str(wav), "my-clone-voice", preview_text="试听")
    assert obj["voice_id"] == "my-clone-voice"
    m, p, h, b = _last("/v1/voice_clone")
    payload = json.loads(b)
    assert payload["file_id"] == "file-123"
    assert payload["voice_id"] == "my-clone-voice"
    assert payload["text"] == "试听"


# ---------------------------------------------------------------- 异步

def test_async_create_uses_audio_sample_rate(client):
    obj = client.async_create(text="长文本", voice_id="v")
    assert obj is not None
    m, p, h, b = _last("/v1/t2a_async_v2")
    payload = json.loads(b)
    au = payload["audio_setting"]
    assert "audio_sample_rate" in au
    assert "sample_rate" not in au


def test_async_create_rejects_both_empty(client):
    with pytest.raises(MiniMaxError):
        client.async_create(text=None, text_file_id=None, voice_id="v")


def test_async_create_rejects_too_long(client):
    with pytest.raises(MiniMaxError):
        client.async_create(text="啊" * 50001, voice_id="v")


def test_async_query(client):
    RECORD["responses"]["/v1/query/t2a_async_query_v2"] = (
        200, {"base_resp": {"status_code": 0, "status_msg": "success"},
              "status": "success", "file_id": "f-1"})
    obj = client.async_query("task-1")
    assert obj["status"] == "success"
    m, p, h, b = _last("/v1/query/t2a_async_query_v2")
    assert "task_id=task-1" in p


# ---------------------------------------------------------------- 预设合成

def test_synth_preset_uses_preset_params(client):
    preset = minimax_tts.get_preset("萝莉")
    client.synth_preset("你好", preset, fmt="wav")
    m, p, h, b = _last("/v1/t2a_v2")
    payload = json.loads(b)
    vs = payload["voice_setting"]
    assert vs["voice_id"] == preset["voice_id"]
    assert vs["speed"] == preset["speed"]
    assert vs["pitch"] == preset["pitch"]
    assert payload["audio_setting"]["format"] == "wav"


# ---------------------------------------------------------------- 配置存取

def test_save_and_load_key(tmp_path, monkeypatch):
    monkeypatch.setattr(minimax_tts, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(minimax_tts, "CONFIG_PATH", str(tmp_path / "minimax.json"))
    minimax_tts.save_key_to_config("sk-test-123")
    assert minimax_tts._load_key_from_config() == "sk-test-123"


def test_load_key_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(minimax_tts, "CONFIG_PATH", str(tmp_path / "nope.json"))
    assert minimax_tts._load_key_from_config() == ""


# ---------------------------------------------------------------- 域名回退

#: 永远返回 2049（invalid api key）的 mock server，模拟「Key 与域名不匹配」
_BAD_RESP = {"base_resp": {"status_code": 2049, "status_msg": "invalid api key"}}


class _BadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _respond(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        raw = json.dumps(_BAD_RESP).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = _respond
    do_POST = _respond


@pytest.fixture(scope="module")
def bad_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _BadHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()


def test_default_base_url_is_auto():
    assert minimax_tts.DEFAULT_BASE_URL == "auto"


def test_base_urls_cover_minimaxi_and_io():
    hosts = minimax_tts.BASE_URLS
    assert "https://api.minimaxi.com" in hosts
    assert "https://api.minimax.io" in hosts


def test_auto_host_falls_back_to_next_host_on_2049(server, bad_server, monkeypatch):
    """第一个域名返回 2049 时，自动换下一个域名并重试。"""
    monkeypatch.setattr(minimax_tts, "BASE_URLS", (bad_server, server))
    c = MiniMaxTTS(api_key="k", base_url="auto")
    obj = c.list_voices()
    assert obj["base_resp"]["status_code"] == 0
    assert c.base_url == server
    assert c.requested_base_url == "auto"


def test_explicit_base_url_does_not_fall_back(bad_server):
    """显式指定域名时不回退，直接把 2049 抛给调用方。"""
    c = MiniMaxTTS(api_key="k", base_url=bad_server)
    with pytest.raises(MiniMaxError) as ei:
        c.list_voices()
    assert ei.value.code == 2049
    assert c.base_url == bad_server


def test_all_hosts_bad_raises_2049(bad_server, monkeypatch):
    monkeypatch.setattr(minimax_tts, "BASE_URLS", (bad_server, bad_server))
    c = MiniMaxTTS(api_key="k", base_url="auto")
    with pytest.raises(MiniMaxError) as ei:
        c.list_voices()
    assert ei.value.code == 2049


def test_synth_with_meta_returns_response(client):
    data, obj = client.synth("你好", "v1", fmt="wav", with_meta=True)
    assert isinstance(data, bytes)
    assert obj["base_resp"]["status_code"] == 0
    assert obj["trace_id"] == "t-1"


def test_synth_without_meta_returns_bytes(client):
    assert isinstance(client.synth("你好", "v1", fmt="wav"), bytes)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
