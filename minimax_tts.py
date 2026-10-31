# -*- coding: utf-8 -*-
"""MiniMax 语音合成（T2A v2）客户端 —— **纯标准库实现，零第三方依赖**。

只用 `urllib.request` + `json`，所以在任何 Python 3.8+ 环境都能直接跑，
不需要 pip install requests / websocket-client。

覆盖官方全部 T2A 能力：
    - 同步 HTTP 合成          synth() / synth_to_file()
    - HTTP 流式              synth_stream()
    - WebSocket 同步流式      synth_ws()        （可选，需要 `websockets`）
    - WebSocket 双向流式      synth_bidi_ws()   （可选，需要 `websockets`）
    - 异步长文本             async_create() / async_query()
    - 声音克隆               clone_voice()
    - 声音设计               design_voice()
    - 音色管理               list_voices() / delete_voice()

用法::

    from minimax_tts import MiniMaxTTS

    tts = MiniMaxTTS(api_key="sk-xxx")
    tts.synth_to_file("你好", "Chinese (Mandarin)_Sweet_Lady", "hi.wav")

API Key 读取顺序：显式参数 > 环境变量 MINIMAX_API_KEY > %APPDATA%/voice-changer/minimax.json
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

__all__ = [
    "MiniMaxTTS",
    "MiniMaxError",
    "MINIMAX_PRESETS",
    "PRESET_GROUPS",
    "SYSTEM_VOICES_ZH",
    "SYSTEM_VOICES_CANTONESE",
    "SYSTEM_VOICES_JA",
    "MODELS",
    "EMOTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "CONFIG_DIR",
    "CONFIG_PATH",
]

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.minimax.io"
DEFAULT_MODEL = "speech-2.8-turbo"

#: 官方全部 speech 模型
MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
)

#: voice_setting.emotion 全部取值
EMOTIONS = (
    "happy", "sad", "angry", "fearful", "disgusted",
    "surprised", "calm", "fluent", "whisper",
)

#: audio_setting.format 全部取值
AUDIO_FORMATS = (
    "mp3", "pcm", "flac", "wav", "pcmu_raw", "pcmu_wav", "opus",
)

#: voice_modify.sound_effects 全部取值（互斥，只能选一个）
SOUND_EFFECTS = (
    "spacious_echo", "auditorium_echo", "lofi_telephone", "robotic",
)

CONFIG_DIR = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), "voice-changer"
)
CONFIG_PATH = os.path.join(CONFIG_DIR, "minimax.json")


# --------------------------------------------------------------------------
# 错误
# --------------------------------------------------------------------------

class MiniMaxError(RuntimeError):
    """MiniMax 接口返回的业务错误。

    Attributes
    ----------
    code : int | None
        ``base_resp.status_code``，0 表示成功。
    trace_id : str | None
        官方返回的请求追踪 id，排查问题时报给客服。
    raw : dict | None
        完整响应体。
    """

    def __init__(self, message, code=None, trace_id=None, raw=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.trace_id = trace_id
        self.raw = raw

    def __str__(self):
        bits = [self.message]
        if self.code is not None:
            bits.append(f"(code={self.code})")
        if self.trace_id:
            bits.append(f"(trace_id={self.trace_id})")
        return " ".join(bits)


# --------------------------------------------------------------------------
# 系统音色清单（官方 System Voice ID List，2026-10 快照）
# --------------------------------------------------------------------------

#: 普通话系统音色 34 条
SYSTEM_VOICES_ZH = (
    ("Chinese (Mandarin)_Reliable_Executive", "可靠高管"),
    ("Chinese (Mandarin)_News_Anchor", "新闻主播"),
    ("Chinese (Mandarin)_Unrestrained_Young_Man", "不羁青年"),
    ("Chinese (Mandarin)_Mature_Woman", "成熟女性"),
    ("Arrogant_Miss", "高傲小姐"),
    ("Robot_Armor", "机器人铠甲"),
    ("Chinese (Mandarin)_Kind-hearted_Antie", "慈心阿姨"),
    ("Chinese (Mandarin)_HK_Flight_Attendant", "港务空姐"),
    ("Chinese (Mandarin)_Humorous_Elder", "幽默长者"),
    ("Chinese (Mandarin)_Gentleman", "绅士"),
    ("Chinese (Mandarin)_Warm_Bestie", "温暖闺蜜"),
    ("Chinese (Mandarin)_Stubborn_Friend", "倔强朋友"),
    ("Chinese (Mandarin)_Sweet_Lady", "甜心女士"),
    ("Chinese (Mandarin)_Southern_Young_Man", "南方青年"),
    ("Chinese (Mandarin)_Wise_Women", "知性女士"),
    ("Chinese (Mandarin)_Gentle_Youth", "温润青年"),
    ("Chinese (Mandarin)_Warm_Girl", "温暖女孩"),
    ("Chinese (Mandarin)_Male_Announcer", "男播音员"),
    ("Chinese (Mandarin)_Kind-hearted_Elder", "慈祥长者"),
    ("Chinese (Mandarin)_Cute_Spirit", "灵动精灵"),
    ("Chinese (Mandarin)_Radio_Host", "电台主持"),
    ("Chinese (Mandarin)_Lyrical_Voice", "抒情歌者"),
    ("Chinese (Mandarin)_Straightforward_Boy", "直爽男孩"),
    ("Chinese (Mandarin)_Sincere_Adult", "真诚成人"),
    ("Chinese (Mandarin)_Gentle_Senior", "温润学长"),
    ("Chinese (Mandarin)_Crisp_Girl", "清脆少女"),
    ("Chinese (Mandarin)_Pure-hearted_Boy", "纯真男孩"),
    ("Chinese (Mandarin)_Soft_Girl", "轻柔气声女孩"),
    ("Chinese (Mandarin)_IntellectualGirl", "知性女孩"),
    ("Chinese (Mandarin)_Warm_HeartedGirl", "暖心女孩"),
    ("Chinese (Mandarin)_Laid_BackGirl", "随性女孩"),
    ("Chinese (Mandarin)_ExplorativeGirl", "好奇女孩"),
    ("Chinese (Mandarin)_Warm-HeartedAunt", "暖心阿姨"),
    ("Chinese (Mandarin)_BashfulGirl", "害羞女孩"),
)

#: 粤语系统音色 6 条（需 language_boost="Chinese,Yue"）
SYSTEM_VOICES_CANTONESE = (
    ("Cantonese_ProfessionalHost (F)", "专业女主播"),
    ("Cantonese_GentleLady", "温柔女士"),
    ("Cantonese_ProfessionalHost (M)", "专业男主播"),
    ("Cantonese_PlayfulMan", "顽皮男"),
    ("Cantonese_CuteGirl", "可爱女孩"),
    ("Cantonese_KindWoman", "善良女子"),
)

#: 日文系统音色 15 条
SYSTEM_VOICES_JA = (
    ("Japanese_IntellectualSenior", "知性学长"),
    ("Japanese_DecisivePrincess", "果断公主"),
    ("Japanese_LoyalKnight", "忠诚骑士"),
    ("Japanese_DominantMan", "强势男"),
    ("Japanese_SeriousCommander", "严肃指挥官"),
    ("Japanese_ColdQueen", "冷艳女王"),
    ("Japanese_DependableWoman", "可靠女性"),
    ("Japanese_GentleButler", "温柔管家"),
    ("Japanese_KindLady", "善良女士"),
    ("Japanese_CalmLady", "沉静女士"),
    ("Japanese_OptimisticYouth", "乐观青年"),
    ("Japanese_GenerousIzakayaOwner", "居酒屋老板"),
    ("Japanese_SportyStudent", "运动学生"),
    ("Japanese_InnocentBoy", "天真少年"),
    ("Japanese_GracefulMaiden", "优雅少女"),
)


# --------------------------------------------------------------------------
# 角色预设
# --------------------------------------------------------------------------

def _preset(name, group, voice_id, speed=1.0, vol=1.0, pitch=0,
            emotion=None, modify=None, note=""):
    """构造一个角色预设。

    Parameters
    ----------
    name : str
        角色显示名。
    group : str
        分组（显示用）。
    voice_id : str
        MiniMax 音色 id。
    speed, vol, pitch : float
        voice_setting 里的语速 / 音量 / 音高。
    emotion : str | None
        voice_setting.emotion，None 表示交给模型自动选。
    modify : dict | None
        voice_modify 附加项，如 ``{"pitch": -30}`` 或 ``{"sound_effects": "robotic"}``。
    note : str
        备注，显示在 GUI 上。
    """
    return {
        "name": name,
        "group": group,
        "voice_id": voice_id,
        "speed": speed,
        "vol": vol,
        "pitch": pitch,
        "emotion": emotion,
        "modify": dict(modify) if modify else None,
        "note": note,
    }


#: 本项目 30 个角色预设 → MiniMax 系统音色
#: 本项目 30 个角色预设 -> MiniMax 系统音色
#: 角色名与分组严格对齐原项目 voice_core.VOICE_PRESETS (v0.4.0 起)
MINIMAX_PRESETS = [
    # ---- 原声 / 基础 (1) ----
    _preset("原声", "原", "Chinese (Mandarin)_Sincere_Adult", 1.00, 1.0, 0,
            note="真诚质朴的成年男声"),

    # ---- 女声 (8) ----
    _preset("真女声", "女", "Chinese (Mandarin)_Sweet_Lady", 1.05, 1.0, 0,
            note="甜美柔和女声，语速略快"),
    _preset("御姐", "女", "Chinese (Mandarin)_Mature_Woman", 0.96, 1.0, -1,
            note="成熟御姐范，语速偏慢"),
    _preset("客服女", "女", "Chinese (Mandarin)_Radio_Host", 1.00, 1.0, 0, "calm",
            note="电台主持腔，客服风格"),
    _preset("播音女", "女", "Chinese (Mandarin)_News_Anchor", 1.00, 1.0, 0,
            note="标准新闻播报腔"),
    _preset("少妇", "女", "Chinese (Mandarin)_Gentle_Senior", 0.98, 1.0, 0,
            note="温柔知性女声"),
    _preset("老奶奶", "女", "Chinese (Mandarin)_Kind-hearted_Antie", 0.88, 1.0, 2,
            note="亲切阿姨，语速慢、音高偏高"),
    _preset("萌妹", "女", "Chinese (Mandarin)_Cute_Spirit", 1.12, 1.0, 2, "happy",
            note="俏皮少女"),
    _preset("嗲嗲", "女", "Chinese (Mandarin)_Soft_Girl", 1.08, 1.0, 1, None,
            modify={"intensity": 30}, note="轻柔气声 + 强度偏柔"),

    # ---- 男声 (7) ----
    _preset("真男声", "男", "Chinese (Mandarin)_Gentleman", 0.97, 1.0, 0,
            note="温文尔雅的绅士"),
    _preset("大叔", "男", "Chinese (Mandarin)_Humorous_Elder", 0.92, 1.0, -3,
            note="幽默长辈，语速慢音高低"),
    _preset("磁性男", "男", "Chinese (Mandarin)_Gentleman", 0.92, 1.0, 0, None,
            modify={"pitch": -30}, note="低沉磁性"),
    _preset("恶魔", "男", "Chinese (Mandarin)_Unrestrained_Young_Man", 0.90, 1.0, -2, None,
            modify={"pitch": -25, "timbre": -20}, note="暗哑带压迫感"),
    _preset("老人", "男", "Chinese (Mandarin)_Gentle_Senior", 0.90, 1.0, -1,
            note="温润长者"),
    _preset("客服男", "男", "Chinese (Mandarin)_Male_Announcer", 1.00, 1.0, 0, "calm",
            note="男播音员腔"),
    _preset("播音男", "男", "Chinese (Mandarin)_News_Anchor", 0.95, 1.0, -1,
            note="新闻播报男声"),

    # ---- 童声 (7) ----
    _preset("萝莉", "童", "Chinese (Mandarin)_Crisp_Girl", 1.15, 1.0, 3, "happy",
            note="清脆爽朗，语速快音高偏高"),
    _preset("正太", "童", "Chinese (Mandarin)_Straightforward_Boy", 1.10, 1.0, 2,
            note="率真少年"),
    _preset("小孩", "童", "Chinese (Mandarin)_Pure-hearted_Boy", 1.15, 1.0, 3,
            note="纯净男孩"),
    _preset("娃娃音", "童", "Chinese (Mandarin)_BashfulGirl", 1.20, 1.0, 4,
            note="语速很快音高很高"),
    _preset("小猫", "童", "Chinese (Mandarin)_BashfulGirl", 1.15, 1.0, 3, None,
            modify={"timbre": 25}, note="清亮带鼻音"),
    _preset("花栗鼠", "童", "Chinese (Mandarin)_Cute_Spirit", 1.25, 1.0, 4, "happy",
            note="极快语速的活泼音"),
    _preset("汤姆猫", "童", "Chinese (Mandarin)_Cute_Spirit", 1.20, 1.0, 3, "happy",
            modify={"timbre": 15}, note="清脆活泼"),

    # ---- 特效 (7) ----
    _preset("机器人", "特效", "Robot_Armor", 1.00, 1.0, 0, None,
            modify={"sound_effects": "robotic"}, note="机甲机械音"),
    _preset("电音女王", "特效", "Chinese (Mandarin)_Sweet_Lady", 1.10, 1.0, 1, None,
            modify={"sound_effects": "robotic"}, note="甜美女声叠机械音效"),
    _preset("外星人", "特效", "Chinese (Mandarin)_Warm_Girl", 1.05, 1.0, 2, "surprised",
            modify={"pitch": -15, "timbre": -40}, note="偏暗偏怪"),
    _preset("兽人", "特效", "Chinese (Mandarin)_Stubborn_Friend", 0.85, 1.0, -4, "angry",
            modify={"pitch": -30}, note="低沉粗犷"),
    _preset("幽灵", "特效", "Chinese (Mandarin)_Soft_Girl", 0.90, 1.0, 0, None,
            modify={"sound_effects": "spacious_echo"}, note="空旷回响"),
    _preset("回声", "特效", "Chinese (Mandarin)_Lyrical_Voice", 1.00, 1.0, 0, "calm",
            modify={"sound_effects": "auditorium_echo"}, note="礼堂混响"),
    _preset("电话音", "特效", "Chinese (Mandarin)_Radio_Host", 1.00, 1.0, 0, None,
            modify={"sound_effects": "lofi_telephone"}, note="电话听筒效果"),
]

#: 预设分组顺序
PRESET_GROUPS = ("原", "女", "男", "童", "特效")


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------

class MiniMaxTTS:
    """MiniMax T2A v2 客户端。

    Parameters
    ----------
    api_key : str | None
        接口密钥。为 None 时依次尝试环境变量 ``MINIMAX_API_KEY`` 和
        ``%APPDATA%/voice-changer/minimax.json`` 里的 ``api_key``。
    base_url : str
        API 根地址。国内直连慢可以换 ``https://api-uw.minimax.io``。
    model : str
        默认模型，见 :data:`MODELS`。
    timeout : float
        单次请求超时秒数（流式合成时按块读取，不受此限制）。
    """

    def __init__(self, api_key=None, base_url=DEFAULT_BASE_URL,
                 model=DEFAULT_MODEL, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "") or _load_key_from_config()
        if not self.api_key:
            raise MiniMaxError(
                "缺少 API Key。请设置环境变量 MINIMAX_API_KEY，"
                "或在 %APPDATA%/voice-changer/minimax.json 写入 {\"api_key\": \"...\"}"
            )

    # ---------------- 内部工具 ----------------

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _check(obj, what="接口"):
        """校验 base_resp，失败抛 :class:`MiniMaxError`。"""
        if not isinstance(obj, dict):
            raise MiniMaxError(f"{what}返回了非 JSON 内容")
        br = obj.get("base_resp") or {}
        code = br.get("status_code", 0)
        if code != 0:
            msg = br.get("status_msg") or f"{what}失败"
            raise MiniMaxError(msg, code=code, trace_id=obj.get("trace_id"), raw=obj)
        return obj

    def _request(self, req, timeout=None, what="接口"):
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                obj = json.loads(body.decode("utf-8", "replace"))
            except Exception:
                raise MiniMaxError(
                    f"{what} HTTP {e.code}: {body[:300]!r}", code=e.code
                ) from e
            # HTTP 层错误也可能带合法的 base_resp
            raise MiniMaxError(
                (obj.get("base_resp") or {}).get("status_msg")
                or f"{what} HTTP {e.code}",
                code=(obj.get("base_resp") or {}).get("status_code", e.code),
                trace_id=obj.get("trace_id"),
                raw=obj,
            ) from e
        except urllib.error.URLError as e:
            raise MiniMaxError(f"{what}网络错误: {e.reason}") from e
        except OSError as e:
            raise MiniMaxError(f"{what}网络错误: {e}") from e
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise MiniMaxError(f"{what}返回的不是合法 JSON: {raw[:200]!r}") from e
        return self._check(obj, what)

    def _post_json(self, path, payload, timeout=None, what="接口"):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, headers=self._headers(), method="POST"
        )
        return self._request(req, timeout=timeout, what=what)

    def _get_json(self, path, params=None, timeout=None, what="接口"):
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        return self._request(req, timeout=timeout, what=what)

    def _post_multipart(self, path, file_field, filename, file_bytes,
                        content_type, fields=None, timeout=None, what="接口"):
        """手工拼 multipart/form-data（标准库没有 multipart 编码器）。"""
        boundary = "----MiniMaxBoundary" + uuid.uuid4().hex
        body = bytearray()
        for k, v in (fields or {}).items():
            body += f"--{boundary}\r\n".encode("utf-8")
            body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8")
            body += f"{v}\r\n".encode("utf-8")
        body += f"--{boundary}\r\n".encode("utf-8")
        body += (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8")
        body += f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        body += file_bytes
        body += f"\r\n--{boundary}--\r\n".encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        }
        req = urllib.request.Request(
            self.base_url + path, data=bytes(body), headers=headers, method="POST"
        )
        return self._request(req, timeout=timeout, what=what)

    def _build_payload(self, text, voice_id, *, model=None, speed=1.0, vol=1.0,
                       pitch=0, emotion=None, sample_rate=32000, fmt="mp3",
                       channel=1, bitrate=128000, language_boost="Chinese",
                       pronunciation_dict=None, timbre_weights=None,
                       voice_modify=None, subtitle_enable=False,
                       subtitle_type="sentence", output_format="hex",
                       text_normalization_mode=None, latex_read=False,
                       stream=False, stream_options=None):
        """组装 T2A v2 请求体（不含 text / model 的公共部分）。"""
        voice_setting = {
            "voice_id": voice_id,
            "speed": float(speed),
            "vol": float(vol),
            "pitch": int(pitch),
        }
        if emotion:
            voice_setting["emotion"] = emotion
        if latex_read:
            voice_setting["latex_read"] = True

        audio_setting = {
            "sample_rate": int(sample_rate),
            "format": fmt,
            "channel": int(channel),
        }
        if fmt == "mp3":
            audio_setting["bitrate"] = int(bitrate)

        payload = {
            "model": model or self.model,
            "text": text,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            "output_format": output_format,
        }
        if stream:
            payload["stream"] = True
            if stream_options:
                payload["stream_options"] = stream_options
        if pronunciation_dict:
            payload["pronunciation_dict"] = pronunciation_dict
        if timbre_weights:
            payload["timbre_weights"] = timbre_weights
        if language_boost:
            payload["language_boost"] = language_boost
        if text_normalization_mode:
            payload["text_normalization_mode"] = text_normalization_mode
        if voice_modify:
            payload["voice_modify"] = voice_modify
        if subtitle_enable:
            payload["subtitle_enable"] = True
            payload["subtitle_type"] = subtitle_type
        return payload

    @staticmethod
    def _extract_audio(obj, fmt="mp3"):
        """从响应里取出音频字节（hex 或 url）。"""
        data = obj.get("data") or {}
        audio = data.get("audio")
        if not audio:
            raise MiniMaxError("响应里没有音频数据", raw=obj)
        if isinstance(audio, str) and audio.startswith("http"):
            url = audio
        else:
            try:
                return bytes.fromhex(audio)
            except ValueError as e:
                raise MiniMaxError("响应里的音频不是合法 hex", raw=obj) from e
        # url 形态
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return resp.read()
        except Exception as e:
            raise MiniMaxError(f"下载音频失败: {e}") from e

    # ---------------- 合成 ----------------

    def synth(self, text, voice_id, **kw):
        """同步合成，返回音频字节。

        Parameters
        ----------
        text : str
            要合成的文本，<10000 字符。
        voice_id : str
            音色 id（系统音色 / 克隆音色 / 声音设计音色都可用）。

        Other Parameters
        ----------------
        model, speed, vol, pitch, emotion, sample_rate, fmt, channel, bitrate,
        language_boost, pronunciation_dict, timbre_weights, voice_modify,
        subtitle_enable, subtitle_type, output_format,
        text_normalization_mode, latex_read

        Returns
        -------
        bytes
            音频数据，格式由 ``fmt`` 决定。
        """
        if not text or not text.strip():
            raise MiniMaxError("文本为空")
        if len(text) >= 10000:
            raise MiniMaxError(
                f"文本过长（{len(text)} 字符），同步接口上限 10000 字符；"
                "请改用异步接口 async_create()"
            )
        payload = self._build_payload(text, voice_id, **kw)
        obj = self._post_json("/v1/t2a_v2", payload, what="语音合成")
        return self._extract_audio(obj, kw.get("fmt", "mp3"))

    def synth_to_file(self, text, voice_id, path, **kw):
        """同步合成并写文件，返回文件路径。"""
        data = self.synth(text, voice_id, **kw)
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def synth_stream(self, text, voice_id, **kw):
        """HTTP 流式合成，逐块 yield 音频字节。

        ``stream=True`` 时官方只支持 ``output_format="hex"``。
        """
        if not text or not text.strip():
            raise MiniMaxError("文本为空")
        payload = self._build_payload(text, voice_id, stream=True, **kw)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/v1/t2a_v2", data=data,
            headers=self._headers(), method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=kw.get("timeout", self.timeout))
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                obj = json.loads(body.decode("utf-8", "replace"))
                br = obj.get("base_resp") or {}
                raise MiniMaxError(
                    br.get("status_msg") or f"语音合成 HTTP {e.code}",
                    code=br.get("status_code", e.code), trace_id=obj.get("trace_id"),
                ) from e
            except ValueError:
                raise MiniMaxError(f"语音合成 HTTP {e.code}") from e
        with resp:
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                br = obj.get("base_resp") or {}
                if br.get("status_code", 0) != 0:
                    raise MiniMaxError(
                        br.get("status_msg") or "流式合成失败",
                        code=br.get("status_code"), trace_id=obj.get("trace_id"),
                    )
                chunk = (obj.get("data") or {}).get("audio")
                if chunk:
                    yield bytes.fromhex(chunk)

    def synth_ws(self, text, voice_id, **kw):
        """WebSocket 同步流式合成，逐块 yield 音频字节。

        需要 ``pip install websockets``。断句由本方法按标点自动完成
        （官方要求客户端自己断句）。
        """
        try:
            import websockets  # noqa: PLC0415
        except ImportError as e:
            raise MiniMaxError(
                "WebSocket 合成需要 websockets 包：pip install websockets"
            ) from e
        import asyncio  # noqa: PLC0415

        payload = self._build_payload(text, voice_id, **kw)
        # task_start 只接受部分字段
        start = {
            "event": "task_start",
            "model": payload["model"],
            "voice_setting": payload["voice_setting"],
            "audio_setting": payload["audio_setting"],
        }
        for k in ("pronunciation_dict", "timbre_weights", "language_boost",
                  "text_normalization_mode", "voice_modify"):
            if k in payload:
                start[k] = payload[k]

        url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        url += "/ws/v1/t2a_v2"

        async def _run():
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with websockets.connect(url, additional_headers=headers) as ws:
                first = json.loads(await ws.recv())
                if first.get("event") != "connected_success":
                    raise MiniMaxError(f"WebSocket 连接失败: {first}", raw=first)
                await ws.send(json.dumps(start, ensure_ascii=False))
                started = json.loads(await ws.recv())
                if started.get("event") != "task_started":
                    br = started.get("base_resp") or {}
                    raise MiniMaxError(
                        br.get("status_msg") or f"task_start 失败: {started}",
                        code=br.get("status_code"), raw=started,
                    )
                for sentence in _split_sentences(text):
                    await ws.send(json.dumps(
                        {"event": "task_continue", "text": sentence},
                        ensure_ascii=False,
                    ))
                    while True:
                        msg = json.loads(await ws.recv())
                        if msg.get("event") != "task_continued":
                            continue
                        br = msg.get("base_resp") or {}
                        if br.get("status_code", 0) != 0:
                            raise MiniMaxError(
                                br.get("status_msg") or "流式合成失败",
                                code=br.get("status_code"),
                            )
                        chunk = (msg.get("data") or {}).get("audio")
                        if chunk:
                            yield bytes.fromhex(chunk)
                        if msg.get("is_final"):
                            break
                await ws.send(json.dumps({"event": "task_finish"}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("event") == "task_finished":
                        break

        loop = asyncio.new_event_loop()
        try:
            agen = _run()
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def synth_bidi_ws(self, chunks, voice_id, **kw):
        """WebSocket 双向流式合成。

        Parameters
        ----------
        chunks : Iterable[str]
            文本片段（可以逐字），服务端会自己缓冲成句子再合成。
            末尾若没有句末标点，需要自己调 ``flush()``，否则服务端会一直等。

        Yields
        ------
        bytes
            音频块。
        """
        try:
            import websockets  # noqa: PLC0415
        except ImportError as e:
            raise MiniMaxError(
                "WebSocket 合成需要 websockets 包：pip install websockets"
            ) from e
        import asyncio  # noqa: PLC0415

        payload = self._build_payload("", voice_id, **kw)
        start = {
            "event": "task_start",
            "model": payload["model"],
            "voice_setting": payload["voice_setting"],
            "audio_setting": payload["audio_setting"],
        }
        for k in ("pronunciation_dict", "timbre_weights", "language_boost",
                  "text_normalization_mode", "voice_modify"):
            if k in payload:
                start[k] = payload[k]

        url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        url += "/ws/v1/t2a_v2_bidi"

        async def _run():
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with websockets.connect(url, additional_headers=headers) as ws:
                first = json.loads(await ws.recv())
                if first.get("event") != "connected_success":
                    raise MiniMaxError(f"WebSocket 连接失败: {first}", raw=first)
                await ws.send(json.dumps(start, ensure_ascii=False))
                started = json.loads(await ws.recv())
                if started.get("event") != "task_started":
                    raise MiniMaxError(f"task_start 失败: {started}", raw=started)
                for piece in chunks:
                    await ws.send(json.dumps(
                        {"event": "task_continue", "text": piece}, ensure_ascii=False))
                await ws.send(json.dumps({"event": "task_finish"}))
                while True:
                    msg = json.loads(await ws.recv())
                    ev = msg.get("event")
                    if ev == "task_continued":
                        chunk = (msg.get("data") or {}).get("audio")
                        if chunk:
                            yield bytes.fromhex(chunk)
                    elif ev == "task_finished":
                        break
                    elif ev in ("task_continued", "task_finished"):
                        break
                    br = msg.get("base_resp") or {}
                    if br.get("status_code", 0) not in (0, None):
                        raise MiniMaxError(
                            br.get("status_msg") or "流式合成失败",
                            code=br.get("status_code"),
                        )

        loop = asyncio.new_event_loop()
        try:
            agen = _run()
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    # ---------------- 异步长文本 ----------------

    def async_create(self, text=None, text_file_id=None, voice_id=None, **kw):
        """创建异步合成任务。

        ``text`` ≤ 50000 字符，或 ``text_file_id`` ≤ 1000000 字符，二选一。
        返回 ``{"task_id":..., "file_id":..., "usage_characters":...}``。
        """
        if not text and not text_file_id:
            raise MiniMaxError("text 与 text_file_id 必须给一个")
        if text and len(text) > 50000:
            raise MiniMaxError(
                f"文本过长（{len(text)} 字符），异步接口上限 50000 字符；"
                "更长的内容请用 text_file_id"
            )
        payload = self._build_payload(text or "", voice_id, **kw)
        # 异步接口的采样率字段名不同
        asr = payload["audio_setting"].pop("sample_rate", None)
        if asr is not None:
            payload["audio_setting"]["audio_sample_rate"] = asr
        if text_file_id:
            payload.pop("text", None)
            payload["text_file_id"] = text_file_id
        obj = self._post_json("/v1/t2a_async_v2", payload, what="异步合成")
        return obj

    def async_query(self, task_id):
        """查询异步任务状态。最多 10 次/秒。"""
        obj = self._get_json("/v1/query/t2a_async_query_v2",
                             {"task_id": task_id}, what="异步查询")
        return obj

    # ---------------- 声音克隆 ----------------

    def upload_file(self, audio_path, purpose="voice_clone"):
        """上传音频，返回 ``file_id``。

        克隆用音频要求：mp3/m4a/wav，时长 10 秒 ~ 5 分钟，≤20MB。
        """
        ext = os.path.splitext(audio_path)[1].lower().lstrip(".") or "wav"
        ctype = {
            "mp3": "audio/mpeg", "m4a": "audio/mp4",
            "wav": "audio/wav", "flac": "audio/flac",
        }.get(ext, "application/octet-stream")
        with open(audio_path, "rb") as f:
            data = f.read()
        obj = self._post_multipart(
            "/v1/files/upload", "file", os.path.basename(audio_path),
            data, ctype, fields={"purpose": purpose}, what="文件上传",
        )
        file_id = obj.get("file_id")
        if file_id is None:
            raise MiniMaxError("上传成功但没拿到 file_id", raw=obj)
        return file_id

    def clone_voice(self, audio_path, voice_id, *, prompt_audio_path=None,
                    prompt_text=None, preview_text=None, model=None,
                    text_validation=None, accuracy=0.7,
                    need_noise_reduction=False, need_volume_normalization=False,
                    aigc_watermark=False):
        """克隆一个音色。

        Parameters
        ----------
        audio_path : str
            主参考音频（10 秒 ~ 5 分钟，≤20MB，mp3/m4a/wav）。
        voice_id : str
            新音色 id，长度 [8,256]，字母开头，可含字母数字 ``-`` ``_``。
        prompt_audio_path : str | None
            克隆提示音频（<8 秒），给了对相似度和稳定性提升明显。
        prompt_text : str | None
            提示音频对应的文本，需与音频一致且以标点结尾。
        preview_text : str | None
            生成试听样音用的文本（≤1000 字符），试听按字符计费。
        text_validation : str | None
            校验用文本（≤200 字符），不达标返回 1043。

        Returns
        -------
        dict
            含 ``voice_id`` / ``demo_audio`` / ``input_sensitive``。
        """
        if not (8 <= len(voice_id) <= 256):
            raise MiniMaxError(f"voice_id 长度必须在 8~256 之间（当前 {len(voice_id)}）")
        if not voice_id[0].isalpha():
            raise MiniMaxError("voice_id 必须以英文字母开头")
        if voice_id[-1] in "-_":
            raise MiniMaxError("voice_id 不能以 - 或 _ 结尾")

        file_id = self.upload_file(audio_path, purpose="voice_clone")

        payload = {"file_id": file_id, "voice_id": voice_id}
        if prompt_audio_path or prompt_text:
            cp = {}
            if prompt_audio_path:
                cp["prompt_audio"] = self.upload_file(
                    prompt_audio_path, purpose="voice_clone")
            if prompt_text:
                cp["prompt_text"] = prompt_text
            payload["clone_prompt"] = cp
        if preview_text:
            payload["text"] = preview_text
            payload["model"] = model or self.model
        if text_validation:
            payload["text_validation"] = text_validation
            payload["accuracy"] = accuracy
        if need_noise_reduction:
            payload["need_noise_reduction"] = True
        if need_volume_normalization:
            payload["need_volume_normalization"] = True
        if aigc_watermark:
            payload["aigc_watermark"] = True

        obj = self._post_json("/v1/voice_clone", payload, what="声音克隆")
        return obj

    # ---------------- 声音设计 ----------------

    def design_voice(self, prompt, preview_text, voice_id=None):
        """用自然语言描述生成音色。

        Parameters
        ----------
        prompt : str
            音色描述，例如「年长女性，温柔磁性，语速偏慢」。
        preview_text : str
            试听文本（≤500 字符）。
        voice_id : str | None
            指定音色 id；不给则自动生成 ``ttv-voice-<ts>-<rand>``。

        Returns
        -------
        dict
            含 ``voice_id`` 与 ``trial_audio``。
        """
        payload = {"prompt": prompt, "preview_text": preview_text}
        if voice_id:
            payload["voice_id"] = voice_id
        obj = self._post_json("/v1/voice_design", payload, what="声音设计")
        return obj

    # ---------------- 音色管理 ----------------

    def list_voices(self, voice_type="all"):
        """查询音色。

        Parameters
        ----------
        voice_type : str
            ``system`` / ``voice_cloning`` / ``voice_generation`` / ``all``。
        """
        obj = self._post_json(
            "/v1/get_voice", {"voice_type": voice_type}, what="音色查询"
        )
        return obj

    def delete_voice(self, voice_id, voice_type="voice_cloning"):
        """删除音色。"""
        obj = self._post_json(
            "/v1/delete_voice",
            {"voice_type": voice_type, "voice_id": voice_id},
            what="音色删除",
        )
        return obj

    # ---------------- 预设辅助 ----------------

    def synth_preset(self, text, preset, *, model=None, fmt="wav",
                     sample_rate=32000, channel=1, output_format="hex", **extra):
        """按角色预设合成。

        Parameters
        ----------
        text : str
            待合成文本。
        preset : dict
            :data:`MINIMAX_PRESETS` 里的一项。
        """
        kw = {
            "model": model or self.model,
            "speed": preset.get("speed", 1.0),
            "vol": preset.get("vol", 1.0),
            "pitch": preset.get("pitch", 0),
            "emotion": preset.get("emotion"),
            "fmt": fmt,
            "sample_rate": sample_rate,
            "channel": channel,
            "output_format": output_format,
        }
        if preset.get("modify"):
            kw["voice_modify"] = dict(preset["modify"])
        kw.update(extra)
        return self.synth(text, preset["voice_id"], **kw)


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------

#: 句末标点（用于 WebSocket 客户端自动断句）
_SENTENCE_END = "。！？…!?.；;\n"


def _split_sentences(text):
    """按句末标点切句，标点跟着前一句走。

    官方要求客户端自己断句；这里保证每段都以标点结尾，
    避免服务端把半句话合成在一起。
    """
    out = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in _SENTENCE_END:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [s for s in out if s.strip()]


def _load_key_from_config():
    """从 %APPDATA%/voice-changer/minimax.json 读 api_key。"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("api_key", "") or ""
    except Exception:
        return ""


def save_key_to_config(api_key, **extra):
    """把 API Key 存到 %APPDATA%/voice-changer/minimax.json。"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = {"api_key": api_key}
    data.update(extra)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        # 存不了就退回环境变量
        os.environ["MINIMAX_API_KEY"] = api_key
    return CONFIG_PATH


def get_preset(name):
    """按名字取角色预设，找不到返回 None。"""
    for p in MINIMAX_PRESETS:
        if p["name"] == name:
            return p
    return None


def grouped_presets():
    """返回 ``[(分组名, [预设, ...]), ...]``，便于 GUI 做分组下拉。"""
    out = []
    for g in PRESET_GROUPS:
        items = [p for p in MINIMAX_PRESETS if p["group"] == g]
        if items:
            out.append((g, items))
    leftover = [p for p in MINIMAX_PRESETS if p["group"] not in PRESET_GROUPS]
    if leftover:
        out.append(("其它", leftover))
    return out


if __name__ == "__main__":  # 简单自检
    import sys

    print(f"预设 {len(MINIMAX_PRESETS)} 个，分组 {PRESET_GROUPS}")
    print(f"中文系统音色 {len(SYSTEM_VOICES_ZH)} 个")
    try:
        client = MiniMaxTTS()
    except MiniMaxError as e:
        print(f"未配置 API Key：{e}")
        sys.exit(0)
    voices = client.list_voices()
    print("音色查询成功：",
          {k: len(v) for k, v in voices.items()
           if isinstance(v, list) and not k.startswith("base_")})
