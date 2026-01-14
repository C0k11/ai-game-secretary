# AI Game Secretary

一个用于 **Steam/Windows 版《碧蓝档案》(Blue Archive)** 的本地自动化助手。

你将得到：
- 截图 -> 识别（物体检测 + OCR）
- 基于屏幕状态的下一步决策
- 自动执行点击/滑动
- 训练阶段的数据采集与数据集准备脚本

## 缓存与模型目录
- HuggingFace/Transformers 缓存：`D:\Project\ml_cache\huggingface`
- 本地模型目录：`D:\Project\ml_cache\models`

## 运行（桌面 App）
- 可执行文件：`dist\GameSecretaryApp\GameSecretaryApp.exe`
- 启动后会打开本地 Dashboard。

## 运行（开发方式）
```powershell
py -m pip install -r requirements.txt
py scripts\run_backend.py
```
然后打开：
- `http://127.0.0.1:8000/dashboard.html`

## 训练阶段（数据采集：Windows 游戏窗口）
采集指定窗口标题的截图序列（用于标注/训练）：
```powershell
py scripts\collect_window_dataset.py --title "Blue Archive" --out data\captures
```

准备标注文件（你标完后放到 `labels.jsonl`）：
- 每行一个样本，字段：`image`、`annotations`（bbox=[x1,y1,x2,y2] + label）

把标注转换成训练输入（骨架入口）：
```powershell
py scripts\prepare_florence_dataset.py --labels data\captures\labels.jsonl --out data\datasets\florence
```
