# RVC 模型目录

把下载好的 RVC 模型放这里。

## 推荐步骤

1. 浏览器打开 https://weights.gg （国内访问可能需要 VPN）
2. 搜索 "Chinese female" / "Mandarin" / "Girl voice" 等
3. 找一个干净的（用户多、口碑好的）模型，点进去下载页：
   - 必选: `.onnx` 或 `.pth` 主模型文件（40MB ~ 100MB）
   - 推荐: `.index` 文件（Faiss 检索增强音色还原，~1MB）
5. 把模型文件名重命名为不含特殊字符的（中文名容易出问题），例如:
   ```
   chinese-female-generic.onnx
   chinese-female-generic.index
   ```
6. 复制到本目录 `D:\代码\voice_changer\models\`

## 国内镜像（不需要 VPN）

- 哔哩哔哩搜索 "RVC 中文 整合包"，输出拉满
- QQ 群 / 频道 搜 "RVC 模型"
- Discord: RVC 社群

## 验证模型安装成功

```
cd D:\代码\voice_changer
.\.venv\Scripts\python.exe _verify_rvc.py
```

如果出现：
```
ONNX models: 1
  - chinese-female-generic.onnx (xx.x MB)
[OK] Model loaded, ready to infer!
```
说明模型装好了。

## 使用

```
.\.venv\Scripts\python.exe rvc_infer.py -i test.wav -o out.wav -m models/chinese-female-generic.onnx
```

默认采样率 40000（RVC 训练采样率）；带 --index 可让音质更还原。

## 注意

- **我没法替你下模型**：这些仓库全部靠 VPN / 国内群 走下载，DSH 会话在的 network 无法访问 weights.gg / huggingface.co 直连
- **别用太花哨的模型**：例如 *浓音气音 / 表情赛高* 这种，会过拟合。上个 *干干净净 5-20 分钟录音 * 的都行
- **模型 + index 一起拿**：index 是训练的音频特征索引，理论上可以提升还原度