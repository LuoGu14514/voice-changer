# MiniMax 语音合成（T2A）API 完整整理

> 整理自 MiniMax 官方文档（platform.minimax.io），抓取时间 2026-10-31。
> 全部接口路径、字段、错误码、价格均以官方文档为准，本文只做中文整理 + 可运行示例。

---

## 目录

1. [接入准备](#1-接入准备)
2. [模型选型](#2-模型选型)
3. [接口总览](#3-接口总览)
4. [同步 TTS（HTTP）](#4-同步-ttshttp)
5. [请求参数详解](#5-请求参数详解)
6. [文本内联控制](#6-文本内联控制)
7. [响应结构与计费字段](#7-响应结构与计费字段)
8. [流式合成](#8-流式合成)
9. [异步 TTS](#9-异步-tts)
10. [声音克隆](#10-声音克隆)
11. [声音设计](#11-声音设计)
12. [音色管理](#12-音色管理)
13. [错误码表](#13-错误码表)
14. [限流](#14-限流)
15. [定价](#15-定价)
16. [语言与系统音色](#16-语言与系统音色)
17. [集成到本项目](#17-集成到本项目)

---

## 1. 接入准备

### 1.1 申请 API Key

1. 登录 <https://platform.minimax.io>
2. 进入「用户中心 → 基本信息 → 接口密钥」
3. 复制 `API_key`

### 1.2 鉴权

所有接口统一用 Bearer Token：

```http
Authorization: Bearer <API_key>
Content-Type: application/json
```

### 1.3 域名选择

| 域名 | 说明 |
|---|---|
| `api.minimax.io` | 全球主站 |
| `api-uw.minimax.io` | **美西节点**，官方推荐，TTFA（首包延迟）更低 |

国内网络访问 `api.minimax.io` 通常正常；若延迟高，换 `api-uw` 或反过来试。

### 1.4 最小可用示例

```python
import requests

API_KEY = "你的_API_KEY"
BASE = "https://api.minimax.io"

resp = requests.post(
    f"{BASE}/v1/t2a_v2",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={
        "model": "speech-2.8-hd",
        "text": "你好，这是一段测试语音。",
        "stream": False,
        "voice_setting": {
            "voice_id": "Chinese (Mandarin)_Warm_Girl",
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
            "emotion": "happy",
        },
        "audio_setting": {
            "sample_rate": 32000,
            "format": "mp3",
            "channel": 1,
        },
        "output_format": "hex",
    },
    timeout=60,
)
resp.raise_for_status()
data = resp.json()

if data.get("base_resp", {}).get("status_code") == 0:
    audio_bytes = bytes.fromhex(data["data"]["audio"])
    with open("out.mp3", "wb") as f:
        f.write(audio_bytes)
    print("长度(ms):", data["extra_info"]["audio_length"])
    print("计费字符:", data["extra_info"]["usage_characters"])
else:
    print("失败:", data["base_resp"])
```

---

## 2. 模型选型

| 模型 | 定位 | 价格 ($/M 字符) | 备注 |
|---|---|---|---|
| `speech-2.8-hd` | 最高音质 | **100** | 超写实，**支持叹词/音效标签** |
| `speech-2.8-turbo` | 高音质 + 快 | **60** | 同上，功能一致只降本 |
| `speech-2.6-hd` | 高音质 | 100 | 超低延迟、智能解析 |
| `speech-2.6-turbo` | 均衡 | 60 | **额外支持 `fluent` / `whisper` 情感** |
| `speech-02-hd` | 复刻相似度高 | 100 | 节奏稳定 |
| `speech-02-turbo` | 均衡 | 60 | |
| `speech-01-hd` | 老一代 | 100 | |
| `speech-01-turbo` | 老一代 | 60 | |

**选择建议**

- 日常对话 / 角色配音 → `speech-2.8-turbo`（性价比最优，功能与 hd 完全一致）
- 追求极致音质（做视频、有偿交付）→ `speech-2.8-hd`
- 需要「耳语」效果 → 只能用 `speech-2.6-hd` / `speech-2.6-turbo`
- 预算敏感且不需要叹词标签 → `speech-02-turbo`

**官方对 hd / turbo 的定位描述**

- `hd` — "Ultra-realistic quality featuring **sound tags**"（超写实，音效标签）
- `turbo` — "Seamless speed meets natural flow / Faster, more affordable, and **ideal for your agent**"（更快更省，适合做 Agent 语音）

---

## 3. 接口总览

| 用途 | 方法 | 路径 |
|---|---|---|
| 同步 TTS | POST | `/v1/t2a_v2` |
| 异步 TTS（创建） | POST | `/v1/t2a_async_v2` |
| 异步 TTS（查询） | GET | `/v1/query/t2a_async_query_v2?task_id=<id>` |
| WebSocket 同步流式 | — | `wss://api.minimax.io/ws/v1/t2a_v2` |
| WebSocket 双向流式 | — | `wss://api.minimax.io/ws/v1/t2a_v2_bidi` |
| 文件上传（克隆用） | POST | `/v1/files/upload` |
| 声音克隆 | POST | `/v1/voice_clone` |
| 声音设计 | POST | `/v1/voice_design` |
| 音色列表 | POST | `/v1/get_voice` |
| 音色删除 | DELETE | `/v1/delete_voice` |

---

## 4. 同步 TTS（HTTP）

`POST https://api.minimax.io/v1/t2a_v2`

适合一次性合成整段文本，最简单。

### 4.1 完整请求体

```json
{
  "model": "speech-2.8-hd",
  "text": "要合成的文本，<10000 字符",
  "stream": false,
  "stream_options": { "exclude_aggregated_audio": false },
  "voice_setting": {
    "voice_id": "Chinese (Mandarin)_Warm_Girl",
    "speed": 1.0,
    "vol": 1.0,
    "pitch": 0,
    "emotion": "happy",
    "latex_read": false
  },
  "audio_setting": {
    "sample_rate": 32000,
    "bitrate": 128000,
    "format": "mp3",
    "channel": 1
  },
  "pronunciation_dict": { "tone": ["omg/oh my god"] },
  "timbre_weights": [
    { "voice_id": "female-chengshu", "weight": 30 },
    { "voice_id": "female-tianmei", "weight": 70 }
  ],
  "language_boost": "Chinese",
  "text_normalization_mode": "basic",
  "voice_modify": {
    "pitch": 0,
    "intensity": 0,
    "timbre": 0,
    "sound_effects": null
  },
  "subtitle_enable": false,
  "subtitle_type": "sentence",
  "output_format": "hex"
}
```

### 4.2 关键约束

- `text` 长度 < **10000 字符**；**超过 3000 字符建议用流式**
- `stream: true` 时只支持 `output_format: "hex"`（流式不支持 url）
- 音色混合时 `voice_setting.voice_id` **留空**，改用 `timbre_weights`
- `voice_modify.sound_effects` 与其它三个滑杆**互斥**

---

## 5. 请求参数详解

### 5.1 顶层

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | **必填**，见模型选型 |
| `text` | string | **必填**，<10000 字符 |
| `stream` | bool | 是否流式返回，默认 false |
| `stream_options.exclude_aggregated_audio` | bool | 流式时为 true 则不返回完整音频汇总 |
| `output_format` | string | `url`（默认 hex；url 有效期 24h）或 `hex` |

### 5.2 voice_setting（音色与韵律）

| 字段 | 范围 | 默认 | 说明 |
|---|---|---|---|
| `voice_id` | string | — | **必填**（用 timbre_weights 时可空） |
| `speed` | [0.5, 2] | 1.0 | 语速 |
| `vol` | (0, 10] | 1.0 | 音量 |
| `pitch` | [-12, 12] int | 0 | 音高 |
| `emotion` | enum | 自动 | 见下 |
| `latex_read` | bool | false | LaTeX 公式朗读，**仅中文**，公式包 `$$` |
| `text_normalization` | bool | — | **已废弃**，改用顶层 `text_normalization_mode` |

**emotion 枚举**：`happy` / `sad` / `angry` / `fearful` / `disgusted` / `surprised` / `calm` / `fluent` / `whisper`

- 默认由模型根据文本自动选最自然的情感，手动指定只在必要时用
- `fluent`、`whisper` **仅 2.6-hd / 2.6-turbo 支持**；2.8 系列不支持 `whisper`

### 5.3 audio_setting（音频输出）

| 字段 | 取值 | 默认 |
|---|---|---|
| `sample_rate` | 8000 / 16000 / 22050 / 24000 / 32000 / 44100 | 32000 |
| `bitrate` | 32000 / 64000 / 128000 / 256000（**仅 mp3 生效**） | 128000 |
| `format` | `mp3` / `pcm` / `flac` / `wav` / `pcmu_raw` / `pcmu_wav` / `opus` | mp3 |
| `channel` | 1（单声道）/ 2（立体声） | 1 |
| `force_cbr` | bool，仅流式 mp3 | — |

**格式说明**

- `pcm` = 裸 PCM，无文件头
- `pcmu_raw` = G.711 μ-law，**固定 8kHz**，无头；`pcmu_wav` = 同编码但带 WAV 容器
- `opus` = Ogg/Opus；**流式下音频块必须按到达顺序重组后才能解码**

### 5.4 pronunciation_dict（读音纠正）

```json
"pronunciation_dict": { "tone": ["omg/oh my god", "行/xing2"] }
```

格式为 `原音/替换`。支持：

- 普通话拼音带声调数字（`1`一声 … `5`轻声）
- IPA（如 `(lɪv)`）
- 日语假名
- 粤拼（如 `(sung3)`）

### 5.5 timbre_weights（音色混合）

最多混合 **4 个**音色：

```json
"timbre_weights": [
  { "voice_id": "voice_a", "weight": 30 },
  { "voice_id": "voice_b", "weight": 70 }
]
```

- `weight` 范围 [1, 100]
- 权重越高越接近该音色
- 使用时 `voice_setting.voice_id` 留空

### 5.6 language_boost（语言/方言增强）

40+ 选项，常用：`Chinese`、`Chinese,Yue`（粤语）、`English`、`Japanese`、`auto`（自动检测）。

完整列表：`Chinese` / `Chinese,Yue` / `English` / `Arabic` / `Russian` / `Spanish` / `French` / `Portuguese` / `German` / `Turkish` / `Dutch` / `Ukrainian` / `Vietnamese` / `Indonesian` / `Japanese` / `Italian` / `Korean` / `Thai` / `Polish` / `Romanian` / `Greek` / `Czech` / `Finnish` / `Hindi` / `Bulgarian` / `Danish` / `Hebrew` / `Malay` / `Persian` / `Slovak` / `Swedish` / `Croatian` / `Filipino` / `Hungarian` / `Norwegian` / `Slovenian` / `Catalan` / `Nynorsk` / `Tamil` / `Afrikaans` / `auto`

> `speech-01` / `speech-02` 系列不支持 `Persian`、`Filipino`、`Tamil`。

### 5.7 text_normalization_mode（文本正则化）

| 值 | 说明 | 额外延迟 |
|---|---|---|
| `basic` | 规则化，快 | ~30 ms |
| `quality` | LLM + 规则混合，正则化质量更高 | ~100 ms |

**限制**：`quality` **仅在非流式 HTTP 请求 + 2.8/2.6 系列生效**；流式或 WS 请求会自动降级为 `basic`（**不报错**）。

设置该字段后，`voice_setting.text_normalization` 与 `english_normalization` 被忽略；不设时这两个（默认关闭）生效。

### 5.8 voice_modify（音色后处理）

对应官网三个滑杆：

| 字段 | 范围 | 含义 |
|---|---|---|
| `pitch` | [-100, 100] | 低沉 ↔ 明亮 |
| `intensity` | [-100, 100] | 强 ↔ 柔 |
| `timbre` | [-100, 100] | 鼻音 ↔ 清亮 |
| `sound_effects` | enum | `spacious_echo` / `auditorium_echo` / `lofi_telephone` / `robotic`，**互斥，只能选一个** |

支持格式：非流式 mp3/wav/flac；流式仅 mp3。

### 5.9 字幕

- `subtitle_enable`（bool，默认 false）
- `subtitle_type`：`sentence`（句级时间戳）/ `word`（词级）/ `word_streaming`
- 非流式返回 `data.subtitle_file`（文件 id）；流式在 `task_continued` 里带 `data.subtitle`

### 5.10 异步接口的字段名差异（易踩坑）

异步 `POST /v1/t2a_async_v2` 里，`audio_setting` 的采样率字段叫 **`audio_sample_rate`**（不是 `sample_rate`），且 `channel` 默认 **2**。

---

## 6. 文本内联控制

在 `text` 里直接写标记，无需额外参数。

### 6.1 停顿

```
<#0.5#>  停顿 0.5 秒
```

- 格式 `<#x#>`，x 为秒，范围 **[0.01, 99.99]**，两位小数
- **不可连续使用**（两个标记不能挨着）

示例：`第一句。<#1.5#>第二句。`

### 6.2 注音（纠正读音）

| 类型 | 写法 |
|---|---|
| 普通话拼音带声调 | `(he2)平` → 读「héng」 |
| IPA | `(lɪv)` |
| 粤拼 | `(sung3)` |

示例：`我(wo3)们(men)一起(qi3yi4)唱(chang4)`。

### 6.3 叹词 / 音效标签（**仅 2.8 系列**）

2.8 系列支持在文本里插入音效标记，共 22 个，包括：

`(laughs)` `(sighs)` `(breath)` `(gasps)` `(whistles)` `(crying)` `(applause)` …

（clone / async 文档中列出的完整 22 个）

> 2.6 / 02 / 01 系列不支持音效标签。

### 6.4 段落

用 `\n` 分段。

---

## 7. 响应结构与计费字段

### 7.1 成功响应

```json
{
  "data": {
    "audio": "hex 字符串或 url",
    "status": 2,
    "subtitle_file": 123456
  },
  "extra_info": {
    "audio_length": 9914,
    "audio_sample_rate": 32000,
    "audio_size": 157869,
    "bitrate": 128000,
    "word_count": 158,
    "usage_characters": 158,
    "usage_voice_count": 0,
    "invisible_character_ratio": 0,
    "audio_format": "mp3",
    "audio_channel": 1,
    "norm_text_char_count": 150
  },
  "trace_id": "04ee3794e2c9e4a6d5f99e77742f06fd",
  "base_resp": { "status_code": 0, "status_msg": "success" }
}
```

### 7.2 字段说明

| 字段 | 说明 |
|---|---|
| `data.audio` | hex 编码音频；或 url（**有效 24 小时**） |
| `data.status` | 1 = 合成中，2 = 完成 |
| `data.subtitle_file` | 字幕文件 id（仅非流式） |
| `extra_info.audio_length` | 音频时长 **毫秒** |
| `extra_info.usage_characters` | **计费字符数** |
| `extra_info.word_count` | 文本字符数（含标点空格），仅供参考，不用于计费 |
| `extra_info.usage_voice_count` | 本次计费的音色数（音色首次使用时才收费） |
| `extra_info.invisible_character_ratio` | 无效字符占比，**>10% 直接报错 1042** |
| `trace_id` | 单次请求追踪 id，排查问题用 |

### 7.3 计费口径

**按 `usage_characters` 计**，不是 `word_count`。文本里的标点、空格、注音标记都会计入。

---

## 8. 流式合成

### 8.1 HTTP 流式

`"stream": true`，服务器按 SSE 风格逐块返回，**每个 chunk 的 `data.audio` 都是完整 hex 块，需要拼接**。

```python
import requests, json

def tts_stream(text, voice_id, model="speech-2.8-turbo"):
    with requests.post(
        "https://api.minimax.io/v1/t2a_v2",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model,
            "text": text,
            "stream": True,
            "voice_setting": {"voice_id": voice_id},
            "audio_setting": {"sample_rate": 32000, "format": "mp3", "channel": 1},
        },
        stream=True,
        timeout=300,
    ) as r:
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            obj = json.loads(line)
            chunk = obj.get("data", {}).get("audio")
            if chunk:
                yield bytes.fromhex(chunk)
```

> `stream: true` 时 `output_format` 只能是 `hex`。

### 8.2 WebSocket 同步流式

`wss://api.minimax.io/ws/v1/t2a_v2`

**事件流**

```
客户端                                   服务端
  │── connected_success ◄───────────────── │
  │── task_start      ─────────────────► │   带 model/voice_setting/audio_setting/...
  │◄─ task_started    ────────────────── │
  │── task_continue{text} ──────────────► │   一次送一句
  │◄─ task_continued{data.audio} ─────── │   is_final 标记是否本次最后一块
  │── task_continue{text} ──────────────► │
  │◄─ task_continued ─────────────────── │
  │── task_finish     ─────────────────► │
  │◄─ task_finished   ────────────────── │   关连接
```

**要点**

- **断句由客户端负责**：每次 `task_continue` 送一句，服务端不自动切句
- 连接空闲约 **120 秒**会被服务端以 `2201` 关闭
- **服务端不发 ping 帧**，客户端要自己发 ping 保活（服务端回 pong 并刷新计时器）

```python
import json, websocket

ws = websocket.create_connection(
    "wss://api.minimax.io/ws/v1/t2a_v2",
    header=[f"Authorization: Bearer {API_KEY}"],
)
print(ws.recv())                       # connected_success

ws.send(json.dumps({
    "event": "task_start",
    "model": "speech-2.8-turbo",
    "voice_setting": {"voice_id": "Chinese (Mandarin)_Warm_Girl"},
    "audio_setting": {"sample_rate": 32000, "format": "mp3", "channel": 1},
}))
print(ws.recv())                       # task_started

for sentence in ["第一句话。", "第二句话。"]:
    ws.send(json.dumps({"event": "task_continue", "text": sentence}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("event") == "task_continued":
            audio = msg.get("data", {}).get("audio")
            if audio:
                print(len(bytes.fromhex(audio)), "bytes")
            if msg.get("is_final"):
                break

ws.send(json.dumps({"event": "task_finish"}))
print(ws.recv())                       # task_finished
```

### 8.3 WebSocket 双向流式（LLM 输出直灌语音）

`wss://api.minimax.io/ws/v1/t2a_v2_bidi`

**与同步流式的区别**

| | `/ws/v1/t2a_v2` | `/ws/v1/t2a_v2_bidi` |
|---|---|---|
| 断句 | **客户端**负责 | **服务端自动缓冲** |
| 逐字发送 | 每个字都合成一次，音频碎 | 先缓冲成整句再合成 |
| 打断 | 不支持，须断连 | `task_cancel`，之后可继续 |
| 句边界事件 | 无 | `sentence_start` / `sentence_end` |
| `task_finish` | 立即关连接 | 先 flush 缓冲再关 |
| 不结束会话就 flush | 无 | `task_flush` |

**事件流**

```
connected_success → task_start → task_started
  → task_continue(任意粒度，甚至单字)
      → sentence_start → task_continued(音频块) → sentence_end
  → [task_cancel → task_canceled]
  → [task_flush  → task_flushed]
  → task_finish  → task_finished
```

**三层完成语义，别混淆**

- `is_final` = **单次请求**音频结束
- `sentence_end` = **当前句**结束
- `task_finished` = **整个会话**结束

**断句规则**（服务端按标点判断）

| 文本情况 | 服务端行为 |
|---|---|
| 以 `。！？…!?.` 或换行结尾 | **立即合成**，无额外延迟 |
| 含 `，、；：,;:` | 累积够了才切，避免碎片 |
| 长文本完全无标点 | 按长度强制切（**边界与语义无关，听感突兀**） |
| 结尾无标点但文本够长 | 短暂静音窗口后 flush |
| 结尾无标点且文本短 | **继续等待**（服务端不会为降延迟而合成半个词） |

**实操要点**

- 多轮对话场景，若一轮结尾又短又没标点，会卡住 → 解决办法：**末尾加句末标点**，或**主动发 `task_flush`**
- `task_finish` 不能替代 flush —— 它会关连接，多轮复用同一连接时不能用
- **保留原始标点**：清洗 LLM 输出时把标点删掉会导致断句位置错乱、听感明显不自然
- 单次 `task_continue` 超 10000 字符 → 返回 `2204`，该段跳过，连接不断

**其它约束**

- 一个连接只能承载一个合成会话；再次发 `task_start` → `2206` 并断连
- 上游 LLM 卡顿时音频里会有对应的空档，这是不可避免的（服务端不会为填空档而合成半句）

---

## 9. 异步 TTS

适合超长文本 / 批量生成。

### 9.1 创建任务

`POST https://api.minimax.io/v1/t2a_async_v2`

```json
{
  "model": "speech-2.8-hd",
  "text": "或者用文件",
  "text_file_id": 123456789,
  "voice_setting": { "voice_id": "..." },
  "audio_setting": { "audio_sample_rate": 32000, "format": "mp3", "channel": 2 }
}
```

- `text` ≤ **50,000 字符**，或 `text_file_id`（≤ **1,000,000 字符**，txt/zip）—— 二选一
- 响应：`task_id` / `task_token` / `file_id` / `usage_characters`

### 9.2 查询任务

`GET https://api.minimax.io/v1/query/t2a_async_query_v2?task_id=<id>`

`status` ∈ `processing` / `success` / `failed` / `expired`

**最多 10 次查询/秒**，超过会报 1002。

### 9.3 下载

用返回的 `file_id` 调文件 Retrieve API 下载。**下载 URL 有效期 9 小时（32400 秒）**。

### 9.4 产物

- txt 输入 → 音频文件 + 字幕文件 + 额外 JSON
- json 输入 → 支持 `title` / `content` / `extra` 三字段，**每个字段产出一个独立文件夹**

---

## 10. 声音克隆

两步：先上传音频，再创建克隆音色。

### 10.1 上传音频

`POST https://api.minimax.io/v1/files/upload`

```
multipart/form-data
  purpose = voice_clone
  file    = <音频文件>
```

响应 `file_id`。

**音频要求**

| 项 | 要求 |
|---|---|
| 格式 | mp3 / m4a / wav |
| 时长 | **10 秒 ~ 5 分钟** |
| 大小 | ≤ 20 MB |

### 10.2 创建克隆音色

`POST https://api.minimax.io/v1/voice_clone`

```json
{
  "file_id": 123456789,
  "voice_id": "my_character_01",
  "clone_prompt": {
    "prompt_audio": 987654321,
    "prompt_text": "克隆提示音频对应的文本，需要以标点结尾"
  },
  "text": "用于生成试听样音的文本，<=1000 字符",
  "model": "speech-2.8-hd",
  "text_validation": "校验用文本，<=200 字符",
  "accuracy": 0.7,
  "need_noise_reduction": false,
  "need_volume_normalization": false,
  "aigc_watermark": false
}
```

**字段说明**

- `file_id` + `voice_id` **必填**
- `voice_id` 规则：长度 [8, 256]，**英文字母开头**，可含字母数字、`-`、`_`；不能以 `-`/`_` 结尾；不能重复
- `clone_prompt`：可选但**强烈建议给**，能显著提升相似度与稳定性
  - `prompt_audio` 是另一个 file_id，**时长 < 8 秒**
  - `prompt_text` 必须与音频内容一致，**且以标点结尾**
- `text` + `model`：可选，返回 `demo_audio` 试听 URL；**试听按 T2A 字符正常计费**
- `text_validation` + `accuracy`：可选，ASR 相似度校验，不达标返回 `1043`；`accuracy` 默认 0.7

**响应**

```json
{
  "voice_id": "my_character_01",
  "demo_audio": "https://...",
  "input_sensitive": { "type": 0 },
  "extra_info": { ... }
}
```

`input_sensitive.type`：`0` 正常 / `1` 严重 / `2` 色情 / `3` 广告 / `4` 违禁 / `5` 辱骂 / `6` 恐怖暴力 / `7` 其他

**重要限制**

- **克隆音色 7 天内未被使用会被系统删除**
- 克隆音色在 `get_voice` 里初始为 **inactive**，需要**成功合成过一次**才会变为可查状态

### 10.3 上传克隆提示音频

`POST https://api.minimax.io/v1/files/upload`，`purpose` 用 `voice_clone`，拿到 file_id 后填进 `clone_prompt.prompt_audio`。

---

## 11. 声音设计

用自然语言描述生成音色，不需要任何样本。

`POST https://api.minimax.io/v1/voice_design`

```json
{
  "prompt": "一位年长女性的声音，温柔磁性，语速偏慢，带一点沙哑",
  "preview_text": "你好，很高兴认识你。",
  "voice_id": "my_designed_voice_01"
}
```

- `prompt` + `preview_text` **必填**，`preview_text` ≤ 500 字符
- `voice_id` 可选，不给则自动生成 `ttv-voice-<时间戳>-<随机串>`
- 响应：`voice_id` + `trial_audio`（hex）

**计费**：Voice Design 本身 **$3 / 个音色**，**首次用于合成时才收费**（不是生成时收）；API 内的预览音频按 **$60 / M 字符** 计费。

---

## 12. 音色管理

### 12.1 查询音色

`POST https://api.minimax.io/v1/get_voice`

```json
{ "voice_type": "all" }
```

`voice_type` ∈ `system` / `voice_cloning` / `voice_generation` / `all`

响应：

```json
{
  "system_voice": [
    { "voice_id": "...", "voice_name": "...", "description": ["..."] }
  ],
  "voice_cloning": [
    { "voice_id": "...", "description": ["..."], "created_time": 1234567890 }
  ],
  "voice_generation": [ ... ]
}
```

### 12.2 删除音色

`DELETE https://api.minimax.io/v1/delete_voice`

```json
{ "voice_type": "voice_cloning", "voice_id": "my_character_01" }
```

> 官方文档对该接口细节描述较少，实际以返回为准。

---

## 13. 错误码表

错误信息在 `base_resp` 里：

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1000 | 未知错误 |
| 1001 | 超时 |
| 1002 | 限流 |
| 1004 | 鉴权失败 |
| 1008 | 余额不足（upload / design） |
| 1013 | 内部错误（clone） |
| 1027 | 输出内容错误（design） |
| 1039 | TPM 限流 |
| 1042 | 无效字符占比 > 10% |
| 1043 | ASR 相似度校验失败（clone） |
| 2013 | 请求参数非法 |
| 2038 | 无克隆权限（需实名） |
| 2049 | API Key 无效（upload） |
| 2201 | 连接空闲超时被关闭 |
| 2202 | 非法事件 |
| 2203 | 空文本被跳过 |
| 2204 | 单次 task_continue 超 10000 字符，该段跳过（连接不断） |
| 2205 | 队列文本过多 —— **软失败**，稍后重发该 task_continue 即可，**不要重连、不要再发 task_start** |
| 2206 | 非法事件顺序（例如重复 task_start） |

> 双向流式接口会**静默丢弃纯空白文本**，不会返回 2203。

---

## 14. 限流

| API | 模型 | RPM | CONN | TPM |
|---|---|---|---|---|
| T2A | speech-2.8 / 2.6 / 02 全系列 | **60** | — | — |
| Voice Cloning | — | 60 | — | — |
| Voice Design | — | **20** | — | — |
| Speech to Text | asr-1.0 | — | 2 | 30,000（按音频秒数） |

> ASR 仅付费账户可用，免费额度不可用。
> 实际额度可能随订阅套餐提升，见 Audio 订阅包说明（Starter RPM 10 → Business RPM 800）。

---

## 15. 定价

### 15.1 按量

| 项目 | 价格 |
|---|---|
| T2A 同步 speech-2.8-hd | **$100 / M 字符** |
| T2A 同步 speech-2.8-turbo | **$60 / M 字符** |
| T2A 异步 speech-2.8-hd | $100 / M 字符 |
| T2A 异步 speech-2.8-turbo | $60 / M 字符 |
| speech-2.6-hd / speech-02-hd | $100 / M 字符 |
| speech-2.6-turbo / speech-02-turbo | $60 / M 字符 |
| Voice Design（声音设计） | **$3 / 个音色**（首次合成时收费） |
| Rapid Voice Cloning（快速克隆） | **$1.5 / 个音色**（首次合成时收费） |
| ASR Speech-to-Text | $0.38 / 小时 |

### 15.2 Audio 订阅包

| 套餐 | 月费 | 点数 | 音色槽位 | RPM |
|---|---|---|---|---|
| Starter | $5 | 10 万 | 10 | 10 |
| Standard | $30 | 30 万 | 100 | 50 |
| Pro | $99 | 110 万 | 250 | 200 |
| Scale | $249 | 330 万 | 500 | 500 |
| Business | $999 | 2000 万 | 800 | 800 |
| 定制 | 按需 | 按需 | 更多 | 无限 RPM/TPM |

**折扣后的实际月费**

| 套餐 | 月付 | 连续 3 个月订阅（9 折） | 年付（8 折） |
|---|---|---|---|
| Starter | $5 | $13.5（3 个月） | $48 |
| Standard | $30 | $81 | $288 |
| Pro | $99 | $267 | $950 |
| Scale | $249 | $672 | $2,390 |
| Business | $999 | $2,697 | $9,590 |

- Audio 包覆盖所有 speech 模型，支持 T2A v2 与 T2A large v2
- ASR / 音视频不在 Audio 包覆盖范围

### 15.3 成本估算

| 场景 | 用量 | turbo ($60/M) | hd ($100/M) |
|---|---|---|---|
| 1 句日常回复（约 30 字） | 30 字 | $0.0018 | $0.003 |
| 1 分钟语音（约 250 字） | 250 字 | $0.015 | $0.025 |
| 1 小时播客（约 9000 字） | 9000 字 | $0.54 | $0.90 |
| 1 万字长文 | 10000 字 | $0.60 | $1.00 |

---

## 16. 语言与系统音色

### 16.1 支持语言（40 种）

中文、粤语、英语、西班牙语、法语、俄语、德语、葡萄牙语、阿拉伯语、意大利语、日语、韩语、印尼语、越南语、土耳其语、荷兰语、乌克兰语、泰语、波兰语、罗马尼亚语、希腊语、捷克语、芬兰语、印地语、保加利亚语、丹麦语、希伯来语、马来语、波斯语、斯洛伐克语、瑞典语、克罗地亚语、菲律宾语、匈牙利语、挪威语、斯洛文尼亚语、加泰罗尼亚语、新挪威语、泰米尔语、南非荷兰语

> `speech-01` / `speech-02` 不支持波斯语、菲律宾语、泰米尔语。

### 16.2 中文系统音色（共 34 个，编号 46–79）

| voice_id | 名称 |
|---|---|
| `Chinese (Mandarin)_Reliable_Executive` | 可靠高管 |
| `Chinese (Mandarin)_News_Anchor` | 新闻主播 |
| `Chinese (Mandarin)_Unrestrained_Young_Man` | 不羁青年 |
| `Chinese (Mandarin)_Mature_Woman` | 成熟女性 |
| `Arrogant_Miss` | 高傲小姐 |
| `Robot_Armor` | 机器人铠甲 |
| `Chinese (Mandarin)_Kind-hearted_Antie` | 慈心阿姨 |
| `Chinese (Mandarin)_HK_Flight_Attendant` | 港务空姐 |
| `Chinese (Mandarin)_Humorous_Elder` | 幽默长者 |
| `Chinese (Mandarin)_Gentleman` | 绅士 |
| `Chinese (Mandarin)_Warm_Bestie` | 温暖闺蜜 |
| `Chinese (Mandarin)_Stubborn_Friend` | 倔强朋友 |
| `Chinese (Mandarin)_Sweet_Lady` | 甜心女士 |
| `Chinese (Mandarin)_Southern_Young_Man` | 南方青年 |
| `Chinese (Mandarin)_Wise_Women` | 知性女士 |
| `Chinese (Mandarin)_Gentle_Youth` | 温柔青年 |
| `Chinese (Mandarin)_Warm_Girl` | 温暖女孩 |
| `Chinese (Mandarin)_Male_Announcer` | 男播音员 |
| `Chinese (Mandarin)_Kind-hearted_Elder` | 慈祥长者 |
| `Chinese (Mandarin)_Cute_Spirit` | 灵动精灵 |
| `Chinese (Mandarin)_Radio_Host` | 电台主持 |
| `Chinese (Mandarin)_Lyrical_Voice` | 抒情歌者 |
| `Chinese (Mandarin)_Straightforward_Boy` | 直爽男孩 |
| `Chinese (Mandarin)_Sincere_Adult` | 真诚成人 |
| `Chinese (Mandarin)_Gentle_Senior` | 温润学长 |
| `Chinese (Mandarin)_Crisp_Girl` | 清脆少女 |
| `Chinese (Mandarin)_Pure-hearted_Boy` | 纯真男孩 |
| `Chinese (Mandarin)_Soft_Girl` | 温柔女孩 |
| `Chinese (Mandarin)_IntellectualGirl` | 知性女孩 |
| `Chinese (Mandarin)_Warm_HeartedGirl` | 暖心女孩 |
| `Chinese (Mandarin)_Laid_BackGirl` | 随性女孩 |
| `Chinese (Mandarin)_ExplorativeGirl` | 好奇女孩 |
| `Chinese (Mandarin)_Warm-HeartedAunt` | 暖心阿姨 |
| `Chinese (Mandarin)_BashfulGirl` | 害羞女孩 |

另有 `moss_audio_*` 系列内部音色 id（如 `moss_audio_ce44fc67-7ce3-11f0-8de5-96e35d26fb85`），官方在接口说明里直接给出，可用但非语义化命名。

### 16.3 粤语（需 `language_boost: "Chinese,Yue"`）

| voice_id | 名称 |
|---|---|
| `Cantonese_ProfessionalHost (F)` | 专业女主播 |
| `Cantonese_GentleLady` | 温柔女士 |
| `Cantonese_ProfessionalHost (M)` | 专业男主播 |
| `Cantonese_PlayfulMan` | 顽皮男 |
| `Cantonese_CuteGirl` | 可爱女孩 |
| `Cantonese_KindWoman` | 善良女子 |

### 16.4 日文（15 个，编号 80–94）

`Japanese_IntellectualSenior`、`Japanese_DecisivePrincess`、`Japanese_LoyalKnight`、`Japanese_DominantMan`、`Japanese_SeriousCommander`、`Japanese_ColdQueen`、`Japanese_DependableWoman`、`Japanese_GentleButler`、`Japanese_KindLady`、`Japanese_CalmLady`、`Japanese_OptimisticYouth`、`Japanese_GenerousIzakayaOwner`、`Japanese_SportyStudent`、`Japanese_InnocentBoy`、`Japanese_GracefulMaiden`

### 16.5 英文（45 个）

`English_expressive_narrator`、`English_radiant_girl`、`English_magnetic_voiced_man`、`English_compelling_lady1`、`English_Aussie_Bloke`、`English_captivating_female1`、`English_Upbeat_Woman`、`English_Trustworth_Man`、`English_CalmWoman`、`English_UpsetGirl`、`English_Gentle-voiced_man`、`English_Whispering_girl`、`English_Diligent_Man`、`English_Graceful_Lady`、`English_ReservedYoungMan`、`English_PlayfulGirl`、`English_ManWithDeepVoice`、`English_MaturePartner`、`English_FriendlyPerson`、`English_MatureBoss`、`English_Debator`、`English_LovelyGirl`、`English_Steadymentor`、`English_Deep-VoicedGentleman`、`English_Wiselady`、`English_CaptivatingStoryteller`、`English_DecentYoungMan`、`English_SentimentalLady`、`English_ImposingManner`、`English_SadTeen`、`English_PassionateWarrior`、`English_WiseScholar`、`English_Soft-spokenGirl`、`English_SereneWoman`、`English_ConfidentWoman`、`English_PatientMan`、`English_Comedian`、`English_BossyLeader`、`English_Strong-WilledBoy`、`English_StressedLady`、`English_AssertiveQueen`、`English_AnimeCharacter`、`English_Jovialman`、`English_WhimsicalGirl`、`English_Kind-heartedGirl`

### 16.6 其它语言

韩语 49 个（101–149）、西语 47 个（150–196）、葡语 73 个（197–269）、法语 6 个（270–275）、印尼语 9 个（276–284）、德语 3 个（285–287）、俄语 8 个（288–295）、意大利语 4 个（296–299）、荷兰语 2 个（300–301）、越南语 1 个（302）、阿拉伯语 2 个（303–304）、土耳其语 2 个（305–306）、乌克兰语 2 个（307–308）、泰语 4 个（309–312）、波兰语 4 个（313–316）、罗马尼亚语 4 个（317–320）、希腊语 3 个（321–323）、捷克语 3 个（324–326）、芬兰语 3 个（327–329）、印地语 3 个（330–332）。

**完整清单以 `get_voice` 接口返回为准**，或见 <https://platform.minimax.io/docs/faq/system-voice-id>。

### 16.7 全部 332 条的语言分布

| 语言 | 条数 | 语言 | 条数 |
|---|---|---|---|
| 葡萄牙语 | 73 | 韩语 | 49 |
| 英语 | 45 | 俄语 | 8 |
| 西班牙语 | 47 | 印尼语 | 9 |
| 普通话 | 34 | 德语 | 3 |
| 日语 | 15 | 意大利语 | 4 |
| 粤语 | 6 | 荷兰语 | 2 |
| 法语 | 6 | 越南语 | 1 |
| 阿拉伯语 | 2 | 土耳其语 | 2 |
| 乌克兰语 | 2 | 泰语 | 4 |
| 波兰语 | 4 | 罗马尼亚语 | 4 |
| 希腊语 | 3 | 捷克语 | 3 |
| 芬兰语 | 3 | 印地语 | 3 |

---

## 17. 集成到本项目

本项目 `voice-changer` 原本是「实时麦克风变声」，现在已转向「文本 → 角色音色语音」。MiniMax API 提供了两条可行路线：

### 路线 A：MiniMax 云端 TTS（推荐）

**优点**：零本地算力、无内存压力、音质高、支持声音克隆与 332 个系统音色、34 个中文音色可直接映射到现有 30 个角色预设。

**实现位置**：`D:\代码\voice_changer\minimax_tts.py`（待建）+ `tts_app.py` / `voice_changer.py` 的 TTS 面板

**建议结构**

```python
class MiniMaxTTS:
    def __init__(self, api_key, model="speech-2.8-turbo"):
        ...
    def list_voices(self) -> list[dict]: ...      # get_voice
    def synth(self, text, voice_id, speed=1.0, vol=1.0, pitch=0,
              emotion=None, fmt="mp3", sample_rate=32000) -> bytes: ...
    def synth_stream(self, text, voice_id, **kw) -> Iterator[bytes]: ...
    def clone_voice(self, audio_path, voice_id, prompt_text=None) -> str: ...
    def design_voice(self, prompt, preview_text) -> str: ...
```

**角色预设映射建议**（30 个角色 → MiniMax voice_id）

| 原角色 | MiniMax voice_id | 附加参数 |
|---|---|---|
| 原声 | `Chinese (Mandarin)_Sincere_Adult` | — |
| 真女声 | `Chinese (Mandarin)_Sweet_Lady` | speed 1.05 |
| 真男声 | `Chinese (Mandarin)_Gentleman` | speed 0.95 |
| 萝莉 | `Chinese (Mandarin)_Crisp_Girl` | pitch +3, speed 1.15 |
| 大叔 | `Chinese (Mandarin)_Humorous_Elder` | pitch -3, speed 0.92 |
| 客服女 | `Chinese (Mandarin)_Radio_Host` | — |
| 播音女 | `Chinese (Mandarin)_News_Anchor` | — |
| 少妇 | `Chinese (Mandarin)_Mature_Woman` | — |
| 老奶奶 | `Chinese (Mandarin)_Kind-hearted_Antie` | pitch +2, speed 0.88 |
| 磁性男 | `English_magnetic_voiced_man`（或中文男声+voice_modify.pitch=-30） | — |
| 机器人 | `Robot_Armor` | voice_modify.sound_effects=robotic |
| 电话音 | — | voice_modify.sound_effects=lofi_telephone |
| 幽灵 | — | voice_modify.sound_effects=spacious_echo |
| 猫/花栗鼠等 | `Chinese (Mandarin)_BashfulGirl` + pitch +6 | — |

**注意**：`voice_modify` 的 `pitch`（-100~100）比 `voice_setting.pitch`（-12~12）范围大得多，做角色化更适合用前者。

### 路线 B：本地 IndexTTS（继续修内存）

仍在推进，瓶颈是系统提交量（见 `EXPERIENCE.md`）。两条路线**不冲突**：MiniMax 作为云端备胎，IndexTTS 作为离线/隐私方案。

---

## 附：官方文档索引

<https://platform.minimax.io/docs/llms.txt>

| 页面 | 路径 |
|---|---|
| 同步 HTTP TTS | `/docs/api-reference/speech-t2a-http` |
| 同步 WS TTS | `/docs/api-reference/speech-t2a-websocket` |
| 双向流式 WS TTS | `/docs/api-reference/speech-t2a-websocket-bidi` |
| 异步创建 | `/docs/api-reference/speech-t2a-async-create` |
| 异步查询 | `/docs/api-reference/speech-t2a-async-query` |
| 声音克隆 | `/docs/api-reference/voice-cloning-clone` |
| 上传克隆音频 | `/docs/api-reference/voice-cloning-uploadcloneaudio` |
| 上传克隆提示 | `/docs/api-reference/voice-cloning-uploadprompt` |
| 声音设计 | `/docs/api-reference/voice-design-design` |
| 音色管理 | `/docs/api-reference/voice-management-get` |
| 音色删除 | `/docs/api-reference/voice-management-delete` |
| 系统音色表 | `/docs/faq/system-voice-id` |
| 限流 | `/docs/guides/rate-limits` |
| 定价 | `/docs/pricing/overview` |
| 错误码 | `/docs/api-reference/errorcode` |

> 全部页面加 `.md` 后缀可直接取到纯文本版（如 `https://platform.minimax.io/docs/api-reference/speech-t2a-http.md`）。
