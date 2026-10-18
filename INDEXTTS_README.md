# IndexTTS-2.5 本地 TTS（CPU）

零样本声音克隆：给一段 3-15 秒参考音频 + 任意文本，输出该音色的语音。

## 环境

| 项 | 值 |
|---|---|
| Python | 3.11.15（uv 管理的独立 venv，与主 `.venv` 隔离）|
| venv | `D:\代码\voice_changer\_indextts_venv` |
| 源码 | `D:\代码\voice_changer\_indextts_src`（index-tts/index-tts，bilibili 模型许可）|
| 模型 | `_indextts_src\checkpoints`（8.5 GB）|
| 推理设备 | CPU（AMD Ryzen 7 7840HS，8 核 16 线程；无 CUDA）|

**必须通过 ASCII 路径运行。** 目录 `D:\代码\voice_changer` 含中文，而 `kaldifst` 的 C++ 代码用
`std::ifstream` 打不开中文路径，会导致 `wetext` 加载 FST 失败：

```
RuntimeError: kaldi-io.cc:Input:744 [E] Error opening input stream
  '...\_indextts_venv\Lib\site-packages\wetext\fsts\traditional_to_simple.fst'
```

已创建 junction 绕过：

```powershell
cmd /c mklink /J D:\vc D:\代码\voice_changer
```

**所有命令都在 `D:\vc` 下执行**，不要用中文路径。

## 模型组成

```
_indextts_src\checkpoints\
  gpt.pth                     3.26 GB  GPT 自回归主干
  s2mel.pth                   415 MB   s2mel 扩散（mel 谱）
  codec.pth                   607 MB   EnhancedCodec 语义编解码
  qwen0.6bemo4-merge\         1.19 GB  Qwen 0.6B 情感分类（可选）
  feat1.pt / feat2.pt                 说话人/情感矩阵
  bpe.model / *.tiktoken              分词器
  wav2vec2bert_stats.pt               特征归一化统计
  hf_cache\
    w2v-bert-2.0\model.safetensors  2.32 GB  Wav2Vec2-BERT 特征提取
    bigvgan\bigvgan_generator.pt    449 MB   BigVGAN 声码器
    campplus_cn_common.bin           28 MB   CAMPPlus 说话人嵌入
```

`hf_cache` 里的辅助模型默认会由 `indextts.utils.model_download.ensure_models_available()`
从 HF/ModelScope 自动下载。本机已用 `_dl_indextts_aux.py` 预先精确下载（跳过了
`conformer_shaw.pt` 2.33 GB 和 BigVGAN discriminator/optimizer 2.7 GB 等用不到的文件）。

## 用法

### 基本

```powershell
cd D:\vc
$env:PYTHONIOENCODING='utf-8'

.\_indextts_venv\Scripts\python.exe ind_extts_cli.py `
  --prompt-wav "D:\vc\models\gsv\v2ProPlus\三月七\reference_audios\中文\emotions\【开心】你看你看，小家伙有反应了！看来它也很喜欢这个名字～.wav" `
  --text "原来你也在这里呀。今天的星穹列车，也要一起出发吗？" `
  --lang ZH `
  --output "D:\vc\tts_outputs\sanyueqi_demo.wav"
```

参数含中文全角标点时容易在 PowerShell 里被转义，推荐用 Python 包一层
（见 `_smoke_indextts.py`）。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--prompt-wav` | 必填 | 参考音频，3-15 秒，时长范围外会警告/报错 |
| `--text` | 必填 | 要合成的文本 |
| `--lang` | `ZH` | `ZH` / `EN` / `JA` / `ES` / `AR` |
| `--output` | `tts_outputs/indextts_out.wav` | 输出路径，22050 Hz 单声道 |
| `--duration-factor` | `1.0` | 语速，0.5-2.0；>1 变慢 |
| `--diffusion-steps` | `20` | s2mel 扩散步数（上游 25）。见下方扫描表 |
| `--num-beams` | `1` | GPT 采样 beam 数（上游 3）。1 最快 |
| `--cfg-rate` | `0.7` | s2mel CFG |
| `--temperature` / `--top-k` / `--top-p` | `0.8` / `30` / `0.8` | 采样参数 |
| `--threads` | `0`（= CPU 逻辑核数） | torch CPU 线程数 |
| `--no-qwen` | 关 | 不加载 Qwen 情感模型，省 1.19 GB 内存和加载时间 |
| `--peak-limit` | `0.95` | 写盘前峰值上限。超过则整体缩放（不改频谱）。设 `1.0` 关闭 |

**峰值保护**：不同参考音频响度差异会让输出电平漂移。实测青雀的【中立】take 输出峰值顶到
1.000（有轻微削波），三月七【开心】只有 0.870。CLI 会在写盘前检查峰值，超过 `--peak-limit`
就整体乘系数拉回来——只动电平，不动音色。

## 性能（实测，CPU）

模型加载 **28-41 秒**（一次性，同一进程内可反复合成）。

单句 3.9 秒音频的分解：

| 阶段 | 耗时 |
|---|---|
| GPT 自回归生成 | 18.8 s |
| s2mel 扩散 | 16.0 s |
| BigVGAN 声码 | 3.4 s |
| **合计** | **44.6 s（RTF 11.5）** |

`--diffusion-steps` 扫描（`_sweep_indextts.py`，4.4-4.6 秒音频，beam=1，不含 Qwen）：

| steps | 耗时 | 音频时长 | peak | 削波 | RTF |
|---|---|---|---|---|---|
| 25 | 47.6 s | 4.42 s | 0.588 | 0.00% | 10.76 |
| **20** | **36.2 s** | 4.31 s | **0.759** | **0.00%** | **8.41** |
| 15 | 33.2 s | 4.42 s | 0.996 | 0.00% | 7.50 |
| 10 | 27.1 s | 4.55 s | 0.978 | 0.00% | 5.96 |
| 8 | 25.8 s | 4.50 s | 1.000 | 0.05% | 5.74 |

**默认取 20**：比上游 25 快 24%，峰值 0.759 留足余量。15 步以下 peak 逼近 1.0，
波形顶到满幅，实际听感会有失真。

## 已做的源码改动

`_indextts_src\indextts\infer_v2_5.py`：

1. 新增模块级 `DIFFUSION_STEPS = 25` / `CFG_RATE = 0.7`（第 39-42 行附近）；
2. `infer_generator()` 里原本硬编码的 `diffusion_steps = 25` / `inference_cfg_rate = 0.7`
   改为读这两个模块变量，让 CLI 能覆盖而不必改源码。

只改了这两处，其余保持上游一致。

## 踩过的坑

1. **中文路径** → `kaldifst` 的 `std::ifstream` 失败。用 `D:\vc` junction 解决。
2. **`C:\Users\<user>\AppData\Local` 写入受限** → ModelScope 的 `.lock`、matplotlib 缓存目录
   都会报 `WinError 5 拒绝访问`。已在脚本里把 `MODELSCOPE_CACHE` / `MPLCONFIGDIR` /
   `HF_HOME` / `XDG_CACHE_HOME` 全部指向项目内目录。
3. **modelscope CLI 不存在 `python -m modelscope`** → 必须直接调用
   `_indextts_venv\Scripts\modelscope.exe`，或走 `snapshot_download` Python API。
4. **torch 2.8.1+cpu 在 aliyun/清华镜像上没有** → 只有 `2.8.0`。用
   `--index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match`。
5. **pip 解析大依赖会卡死** → 换 `uv pip install`，8 分钟的 torch 下载 + 秒级解析。
6. **`wetext` 的 FST 数据是运行必需**，不是可选依赖，务必装 `wetext`。

## 图形界面

```powershell
cd D:\代码\voice_changer
.\.venv\Scripts\pythonw.exe tts_app.py
```

界面分四块：

| 区域 | 内容 |
|---|---|
| 角色 | 角色下拉（三月七 / 青雀）+ 情绪参考下拉（从参考音频目录自动列出）+ 参考文本提示 |
| 文本 | 多行输入框，右下角实时字数 |
| 参数 | 语速系数 / 扩散步数 / beam |
| 按钮 | 合成 / 播放 / 停止播放 / 另存为 / 打开目录 |

合成结果落在 `tts_outputs\<角色>_<时间戳>.wav`。

### 架构

GUI 和合成跑在两个进程里：

```
tts_app.py (主 venv, Python 3.12, tkinter)
        │  JSON-Lines over stdin/stdout
        ▼
indextts_worker.py (_indextts_venv, Python 3.11, IndexTTS)
```

模型加载要 30-60 秒，所以 worker 是**常驻**的：程序启动时就在后台拉起来，
之后每次合成都复用同一个已加载的模型，不用重复等。

协议：

| 方向 | 报文 |
|---|---|
| 启动完成 → | `{"ready": true, "load_s": 42.3, "sr": 22050}` |
| 请求 → | `{"id": 1, "text": "...", "ref": "...", "lang": "ZH", "output": "...", "duration_factor": 1.0, "diffusion_steps": 20, "num_beams": 1}` |
| 成功 ← | `{"id": 1, "ok": true, "output": "...", "dur": 3.11, "rtf": 19.02, "peak": 0.647, "infer_s": 59.2}` |
| 失败 ← | `{"id": 1, "ok": false, "error": "..."}` |
| 退出 → | `{"cmd": "quit"}` |

worker 里把 `sys.stdout` 重定向到了 stderr（模型内部有大量 `print`），
协议走单独保存的 `PROTO` 句柄，保证 stdout 上只有 JSON。

worker 关闭了 `use_qwen_emo`，省 1.19 GB 内存；GUI 不做情感文本推断，
情感靠选不同的参考音频来体现。

## 实测结果

| 角色 | 参考音频 | 文本 | 时长 | peak | 削波 | RTF |
|---|---|---|---|---|---|---|
| 三月七 | 【开心】…喜欢这个名字～ | 原来你也在这里呀。今天的星穹列车，也要一起出发吗？ | 3.87 s | 0.870 | 0.00% | 11.53 |
| 青雀 | 【中立】我甚至找不出一张完整的参赛选手名单… | 太卜大人说，今日卦象大吉。要不要来一局？ | 4.23 s | 0.950 | 0.00% | 17.57 |
| 三月七（GUI） | 【开心】…喜欢这个名字～ | 测试一下图形界面的合成流程。 | 3.11 s | 0.647 | 0.00% | 19.02 |

两个角色都能正常合成，音色由参考音频决定。

GUI 的 RTF 比 CLI 高（19 vs 11.5），因为 tkinter 事件循环和 worker 进程在同一台
机器上抢 CPU，加上模型常驻占内存。可接受范围内。

## 磁盘占用

| 目录 | 大小 | 说明 |
|---|---|---|
| `_indextts_src` | 7.9 GB | 源码 + 模型（必需）|
| `_indextts_venv` | 3.9 GB | Python 3.11 环境（必需）|
| `_gsv_venv` | 1.4 GB | **已废弃**（GPT-SoVITS 路线）|
| `_gsv_src` | 1.8 GB | **已废弃** |
| `models/gsv` | 1.6 GB | 参考音频来源，**仍在用，别删** |
| `_wheels` | 0.2 GB | 已废弃的 wheel 缓存 |

`_gsv_venv` / `_gsv_src` / `_wheels` 合计约 3.4 GB 可以清理。
注意 `models/gsv/v2ProPlus/*/reference_audios/` 是当前唯一的参考音频来源，
删 `_gsv_*` 目录不影响它。

## 接入 QQ 机器人（nahida-bot-2）

机器人本体在 `D:\nahida-bot-v2\nahida-bot-2`（Python 3.12），而 IndexTTS 要求
Python >=3.10,<3.12，**两者不能同进程**。因此把引擎包成 HTTP 服务，机器人侧只加一个
`TtsProvider` 适配器（与仓库里已有的 gpt-sovits-v2 / minimax-t2a-v2 并列）。

### 两条进程

| 进程 | 目录 | 端口 | 启动 |
|---|---|---|---|
| IndexTTS 引擎 | `D:\vc`（ASCII junction，中文路径会让 wetext 报 kaldifst 错误） | 9881 | `start_indextts_server.bat` |
| QQ 机器人 | `D:\nahida-bot-v2\nahida-bot-2` | 6186 (webapi) / 6185 | `restart_bot.ps1` 或 `uv run nahida-bot start --config-yaml config.yaml` |

启动顺序无所谓：引擎模型在后台线程加载（约 50 秒），期间 `/health` 返回
`loaded:false`、`/tts` 返回 503，机器人侧会把 503 当作**可重试**错误处理。

### 引擎侧接口（`indextts_server.py`，纯标准库 ThreadingHTTPServer）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health`、`/api/health` | `{ok, loaded, version, sr, load_s, voices, threads, error}` |
| GET | `/voices` | 12 条音色（三月七 7 情绪 + 青雀 5 情绪） |
| POST | `/tts`、`/api/tts`、`/synthesize` | body `{text, ref_audio|voice, lang, duration_factor, diffusion_steps, num_beams, peak_limit}` → `audio/wav` |

成功响应带 `X-Audio-Duration-Ms` / `X-Audio-Sample-Rate` / `X-Audio-Peak` /
`X-Infer-Ms` / `X-Rtf` / `X-Diffusion-Steps` 头；失败统一
`{"error":{"code","message"}}`，code 取 `bad_request` / `empty_text` /
`voice_not_found` / `ref_audio_not_found` / `model_not_ready` / `synthesis_failed`。

`voice` 支持 `角色/情绪`、`角色:情绪`、纯角色名（优先中立）、纯情绪名。

### 机器人侧

- `nahida_bot/speech/providers/indextts.py`：`IndexTTSProvider`，`type = "indextts-v2"`，
  已注册进 `service.py` 的 `_BUILTIN_PROVIDERS`。
- `speed` 语义换算：`duration_factor = 1.0 / speed`（IndexTTS 控制的是时长倍数）。
- 语言优先级沿用仓库既有约定：voice.lang 覆盖 request.text_lang，全空才用 backend.lang；
  最终统一成两位大写码（`zh`/`zh-CN`/`cmn` → `ZH`）。
- 配置写在 `config.yaml` 两处：`webapi.speech`（共享 SpeechService，供 WebAPI 用）与顶层
  `tts:`（speak 插件）。两处用 YAML 锚点 `&tts` / `<<: *tts` 合并，避免重复。

### 三种调用方式

1. QQ 里 `说话 <文本>` 或 `/speak <文本>`（插件命令）
2. 让模型自己调 `speak` 工具（参数 `text` / `emotion` / `text_lang` / `send` / `caption`）
3. WebAPI：`POST /api/speech/jobs` → 拿 `download_url` → `GET /api/media/speech/{id}`

### 实测

- 引擎加载 51.6 秒，12 个音色。
- 走机器人实时服务：青雀 2.95 秒音频耗时 41.4 秒（RTF ≈ 14），130 KB，peak 0.95、无削波。
- `tests/test_indextts_provider.py` 27 passed；speech 相关既有 8 个测试文件 79 passed。
- speak 插件端到端（真配置 + 真服务）产出 213 KB wav 并成功挂 voice 附件。

## 下一步

- [x] CLI 跑通
- [x] GUI 集成
- [x] 接入 QQ 机器人（nahida-bot-2）
- [ ] 实时流式合成（当前是整句离线合成，一句 30-60 秒）
- [ ] 长文本自动切句，避免单次推理过长
