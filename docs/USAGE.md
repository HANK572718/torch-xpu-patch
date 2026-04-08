# torch-xpu-patch 使用指南

Intel Arc GPU (XPU) 通用補丁，讓任何 torch CUDA 專案**無需修改任何原始碼**即可使用 Intel XPU 加速。

---

## 目錄

1. [補丁能解決什麼問題](#補丁能解決什麼問題)
2. [補丁無法解決什麼](#補丁無法解決什麼)
3. [安裝](#安裝)
4. [使用方式](#使用方式)
   - [方式 A：一次性安裝到 venv（最推薦）](#方式-a一次性安裝到-venv最推薦)
   - [方式 B：在主程式最前面加一行](#方式-b在主程式最前面加一行)
   - [方式 C：Lightning 專用補丁](#方式-c-lightning-專用補丁)
5. [補丁範圍說明](#補丁範圍說明)
6. [驗證與狀態查詢](#驗證與狀態查詢)
7. [加入其他專案的步驟](#加入其他專案的步驟)
8. [常見問題](#常見問題)

---

## 補丁能解決什麼問題

原本寫給 NVIDIA GPU 的程式碼，通常使用以下 CUDA 專用 API：

```python
# 這些 CUDA 寫法在 Intel XPU 會失敗
model.cuda()
tensor.to('cuda')
tensor.to('cuda:0')
torch.device('cuda')
torch.cuda.is_available()
torch.cuda.device_count()
with torch.autocast('cuda'): ...
trainer = pl.Trainer(accelerator='auto')   # Lightning 不認識 XPU
trainer = pl.Trainer(accelerator='gpu')
```

套用 `torch-xpu-patch` 後，**以上全部自動重導向到對應的 XPU API**，無需修改任何程式碼。

---

## 補丁無法解決什麼

| 情況 | 說明 |
|------|------|
| CUDA kernel / PTX 程式碼 | `.cu` 或 `torch.cuda.CUDAGraph` 等 NVIDIA 專屬低層 API |
| `torch.backends.cudnn.*` | cuDNN 專屬設定 |
| 已編譯的 CUDA extension | 需要重新針對 XPU 編譯 |
| `onnxruntime-gpu`（CUDA 版本） | 需改用 `onnxruntime` + OpenVINO |

---

## 安裝

### 前提：安裝 XPU 版本的 torch

```toml
# pyproject.toml（Poetry）
[tool.poetry.group.windows.dependencies]
torch = { url = "https://download.pytorch.org/whl/xpu/torch-2.9.0%2Bxpu-cp311-cp311-win_amd64.whl#sha256=d56c44ab4818aba57e5c7b628f422d014e0d507427170a771c5be85e308b0bc6" }
torchaudio = { url = "https://download.pytorch.org/whl/xpu/torchaudio-2.9.0%2Bxpu-cp311-cp311-win_amd64.whl#sha256=431334d35c70608a83582f6397135bcfd49d69d1764644a273fbcdabd22a346f" }
torchvision = { url = "https://download.pytorch.org/whl/xpu/torchvision-0.24.0%2Bxpu-cp311-cp311-win_amd64.whl#sha256=9bb0d1421c544ac8e2eca5b47daacaf54706dc9139c003aa5e77ee5f355c5931" }
pytorch-triton-xpu = { url = "https://download.pytorch.org/whl/pytorch_triton_xpu-3.5.0-cp311-cp311-win_amd64.whl#sha256=debf75348da8e8c7166b4d4a9b91d1508bb8d6581e339f79f7604b2e6746bacd" }
```

### 安裝 torch-xpu-patch 到你的專案

**方法 1：直接 pip 安裝本地路徑**

```bash
# 在你的目標專案 venv 中
pip install D:/project/torch-xpu-patch
# 或 Poetry
poetry add D:/project/torch-xpu-patch --group dev
```

**方法 2：Poetry path 依賴**

```toml
# 在你的專案 pyproject.toml 中
[tool.poetry.dependencies]
torch-xpu-patch = { path = "D:/project/torch-xpu-patch", develop = true }
```

---

## 使用方式

### 方式 A：一次性安裝到 venv（最推薦）

在目標專案的 venv 安裝好 `torch-xpu-patch` 後，執行一次：

```bash
poetry run xpu-patch-install
```

**效果**：在 `.venv/Lib/site-packages/usercustomize.py` 安裝自動啟動腳本。
Python 每次啟動都會自動執行這個檔案，補丁自動套用，**不需要改任何程式碼**。

```bash
# 移除自動啟動
poetry run xpu-patch-uninstall

# 查看狀態
poetry run xpu-patch-status
```

---

### 方式 B：在主程式最前面加一行

如果你不想用 usercustomize 方式，可以在主程式（或任何入口點）的**最頂端**加入：

```python
# main.py 或 train.py 的第一行
import torch_xpu_patch; torch_xpu_patch.apply()

# 之後的程式碼完全不需要改動
import torch
model = MyModel().cuda()           # 自動走 XPU
tensor = torch.zeros(3).to('cuda') # 自動走 XPU
```

**重要**：必須在所有其他 `import` **之前**執行，特別是 `import torch` 之前。

---

### 方式 C：Lightning 專用補丁

如果你的專案使用 PyTorch Lightning，還需要額外套用 Lightning 層補丁：

```python
# 在主程式最開頭
import torch_xpu_patch
torch_xpu_patch.apply()                                          # torch 層

from torch_xpu_patch.lightning_patch import apply_lightning_patch
apply_lightning_patch()                                          # Lightning 層

# 之後正常使用
import lightning.pytorch as pl
trainer = pl.Trainer(accelerator='auto')   # 自動選 XPU
trainer = pl.Trainer(accelerator='gpu')    # 也會選 XPU
trainer = pl.Trainer(accelerator='xpu')    # 明確指定 XPU
```

或是用 usercustomize 方式（已內建 Lightning 補丁）：

```bash
poetry run xpu-patch-install  # 同時安裝 torch + Lightning 兩層補丁
```

---

## 補丁範圍說明

| 補丁目標 | 原始 API | 重導向後 | 說明 |
|---------|---------|---------|------|
| `torch.cuda` 模組 | `torch.cuda.*` | `torch.xpu.*` | 代理模組，優先查 xpu |
| Tensor 方法 | `tensor.cuda()` | `tensor.xpu()` | 包含 device 參數重寫 |
| Tensor `to()` | `tensor.to('cuda')` | `tensor.to('xpu')` | 字串 device 重寫 |
| nn.Module 方法 | `model.cuda()` | `model.xpu()` | |
| nn.Module `to()` | `model.to('cuda')` | `model.to('xpu')` | |
| `torch.device()` | `torch.device('cuda')` | `torch.device('xpu')` | 工廠包裝 |
| `torch.autocast` | `autocast('cuda')` | `autocast('xpu')` | |
| `GradScaler` | `cuda.amp.GradScaler` | XPU 相容版本 | |
| Lightning 加速器 | `accelerator='auto'/'gpu'` | 自動選 XPU | 需 Lightning 補丁 |
| Lightning Registry | - | 新增 `'xpu'` 加速器 | |

---

## 驗證與狀態查詢

```bash
# 查看補丁狀態與設備資訊
poetry run xpu-patch-status

# 執行功能驗證測試
poetry run xpu-patch-verify
```

輸出範例（XPU 可用）：

```
============================================================
  torch-xpu-patch 狀態
============================================================
  補丁已套用: 是
  XPU 可用:   是
  XPU 設備 (1 個):
    - Intel(R) Arc(TM) A770 Graphics

  已覆蓋的 API：
    torch.cuda
    torch.Tensor.cuda
    torch.Tensor.to
    torch.nn.Module.cuda
    torch.nn.Module.to
    torch.device
    torch.autocast
============================================================
```

---

## 加入其他專案的步驟

1. **確認 XPU torch 已安裝**（參見上方安裝章節的 WHL 連結）
2. **在目標專案的 venv 安裝 torch-xpu-patch**
3. **執行一次 `xpu-patch-install`**，之後所有程式自動生效
4. **執行 `xpu-patch-verify` 確認功能正常**
5. 若有使用 PyTorch Lightning，需確認 `apply_lightning_patch()` 也被呼叫（usercustomize 方式已內建）

---

## 常見問題

### Q：安裝後訓練速度沒有提升？

確認模型確實在 XPU 上：

```python
import torch
print(next(model.parameters()).device)  # 應該顯示 xpu:0
```

### Q：出現 `RuntimeError: XPU is not available`？

- 確認安裝的是 XPU 版本的 torch（WHL 連結中要有 `+xpu`）
- 確認 Intel GPU 驅動已安裝（`intel_extension_for_pytorch` 或 Windows 內建 Arc 驅動）
- 執行 `poetry run xpu-patch-status` 查看詳細狀態

### Q：`usercustomize.py` 是什麼？會有副作用嗎？

Python 啟動時會自動 import 名為 `usercustomize` 的模組（若存在於 site-packages）。
補丁只修改 in-memory 的 Python 物件，不修改任何磁碟上的安裝檔案，可以安全移除。

### Q：可以和 CUDA 版本的 torch 並存嗎？

不建議。若同時安裝了 CUDA torch 和 XPU torch，行為未定義。
請在獨立的 venv 環境中使用各自版本。

### Q：如何在同一個機器上切換 CPU/XPU/CUDA？

使用環境變數控制補丁：

```python
import os
import torch_xpu_patch

# 只在有設定此環境變數時才套用補丁
if os.getenv("USE_XPU", "1") == "1":
    torch_xpu_patch.apply()
```

---

## 技術背景

### 為什麼現有的 Lightning XPU 補丁不夠？

PyTorch Lightning 的 `accelerator='auto'` 選擇邏輯不包含 XPU，
但即使加入了 Lightning 層補丁，直接呼叫 `tensor.cuda()` 或 `model.cuda()` 的程式碼仍會失敗，
因為這是在 torch 的 C++ extension 層發生的，需要在 Python 層進行攔截。

本套件同時解決這兩個層次的問題。

### Monkey Patch 的安全性

本補丁不修改任何磁碟上的安裝檔案（`.py` 或 `.so`），
所有修改都是在 Python process 啟動時，對 in-memory 的物件進行替換。
Process 結束後自動還原。可以呼叫 `torch_xpu_patch.unapply()` 手動還原。
