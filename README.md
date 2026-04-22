# torch-xpu-patch

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/torch-2.9.0%2Bxpu-orange)](pyproject.toml)
[![Lightning](https://img.shields.io/badge/lightning-2.5%20%7C%202.6-purple)](pyproject.toml)

Intel XPU (Arc GPU) 通用 Torch Monkey Patch — 讓任何使用 CUDA 的 torch 專案**無需修改任何原始碼**即可使用 Intel XPU 加速。

---

## 功能

套用此補丁後，以下 CUDA 寫法**全部自動重導向到 XPU**：

```python
model.cuda()                        # → model.xpu()
tensor.to('cuda')                   # → tensor.to('xpu')
tensor.to('cuda:0')                 # → tensor.to('xpu:0')
torch.cuda.is_available()           # → torch.xpu.is_available()
torch.cuda.device_count()           # → torch.xpu.device_count()
with torch.autocast('cuda'): ...    # → autocast('xpu')

# PyTorch Lightning
pl.Trainer(accelerator='auto')      # 自動選 XPU
pl.Trainer(accelerator='gpu')       # 也選 XPU
pl.Trainer(accelerator='xpu')       # 明確指定 XPU
pl.Trainer(precision='16-mixed')    # 混合精度 XPU 支援
```

## 安裝

### 前提：需已安裝 XPU 版本的 torch

```toml
# pyproject.toml
[tool.poetry.group.windows.dependencies]
torch = { url = "https://download.pytorch.org/whl/xpu/torch-2.9.0%2Bxpu-cp311-cp311-win_amd64.whl#sha256=d56c44ab4818aba57e5c7b628f422d014e0d507427170a771c5be85e308b0bc6" }
torchvision = { url = "https://download.pytorch.org/whl/xpu/torchvision-0.24.0%2Bxpu-cp311-cp311-win_amd64.whl#sha256=9bb0d1421c544ac8e2eca5b47daacaf54706dc9139c003aa5e77ee5f355c5931" }
torchaudio = { url = "https://download.pytorch.org/whl/xpu/torchaudio-2.9.0%2Bxpu-cp311-cp311-win_amd64.whl#sha256=431334d35c70608a83582f6397135bcfd49d69d1764644a273fbcdabd22a346f" }
pytorch-triton-xpu = { url = "https://download.pytorch.org/whl/pytorch_triton_xpu-3.5.0-cp311-cp311-win_amd64.whl#sha256=debf75348da8e8c7166b4d4a9b91d1508bb8d6581e339f79f7604b2e6746bacd" }
```

### 從 GitHub 安裝（推薦）

```bash
# pip
pip install git+https://github.com/HANK572718/torch-xpu-patch.git

# Poetry
poetry add git+https://github.com/HANK572718/torch-xpu-patch.git
```

### 從本地路徑安裝

```bash
pip install D:/project/torch-xpu-patch
# 或
poetry add D:/project/torch-xpu-patch --group dev
```

## 使用方式

### 方式 A：一次性安裝到 venv（最推薦）

```bash
# 安裝補丁後執行一次即可，之後 Python 每次啟動自動套用
poetry run xpu-patch-install

# 驗證
poetry run xpu-patch-status
poetry run xpu-patch-verify
```

### 方式 B：在主程式加一行

```python
# 必須在所有 import 之前
import torch_xpu_patch; torch_xpu_patch.apply()

import torch
model = MyModel().cuda()  # 自動走 XPU
```

### 方式 C：PyTorch Lightning

```python
import torch_xpu_patch
torch_xpu_patch.apply()

from torch_xpu_patch.lightning_patch import apply_lightning_patch
apply_lightning_patch()

import lightning.pytorch as pl
trainer = pl.Trainer(accelerator='auto')  # 自動選 XPU
```

## 補丁範圍

| 補丁 | 原始 API | 重導向 |
|------|---------|--------|
| torch.cuda 模組 | `torch.cuda.*` | `torch.xpu.*` |
| Tensor 方法 | `tensor.cuda()` | `tensor.xpu()` |
| Tensor to() | `tensor.to('cuda')` | `tensor.to('xpu')` |
| nn.Module | `model.cuda()` | `model.xpu()` |
| autocast | `autocast('cuda')` | `autocast('xpu')` |
| GradScaler | `cuda.amp.GradScaler` | XPU 相容版本 |
| Lightning Accelerator | `accelerator='auto'/'gpu'` | 自動選 XPU |
| Lightning 混合精度 | `precision='16-mixed'` | XPU 支援 |

## 相容性

| 套件 | 版本 |
|------|------|
| Python | 3.11, 3.12 |
| torch (XPU) | 2.9.0+xpu |
| PyTorch Lightning | 2.5.0, 2.5.1, 2.6.0, 2.6.1 |
| 硬體 | Intel Arc A-Series, Intel Data Center GPU Max |
| 平台 | Windows (主要測試), Linux |

## 測試

```bash
# 執行完整測試套件（需要 XPU 硬體）
poetry run xpu-patch-test

# 只跑 torch 層
poetry run python -m torch_xpu_patch.tests.test_core

# 只跑 Lightning 層
poetry run python -m torch_xpu_patch.tests.test_lightning
```

測試結果（Intel Arc A770, Lightning 2.6.1）：
- torch 層：**18/18 PASS**
- Lightning 層：**23/23 PASS**（含端到端訓練迴圈）

## CLI 工具

```bash
xpu-patch-install    # 安裝自動啟動（usercustomize.py）
xpu-patch-uninstall  # 移除自動啟動
xpu-patch-status     # 顯示補丁狀態與設備資訊
xpu-patch-verify     # 功能驗證測試
xpu-patch-test       # 完整測試套件
```

## 注意事項

此補丁**無法**處理：
- CUDA kernel / PTX 原生程式碼（`.cu` 檔）
- `torch.backends.cudnn.*` cuDNN 專屬設定
- 已編譯的 CUDA C++ extension
- `onnxruntime-gpu`（CUDA 版本）

## License

Apache License 2.0 — 詳見 [LICENSE](LICENSE)

## 詳細文件

參見 [docs/USAGE.md](docs/USAGE.md)
