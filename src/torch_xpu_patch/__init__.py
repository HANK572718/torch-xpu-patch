"""torch-xpu-patch

Intel XPU 通用 Monkey Patch 套件。

讓任何原本針對 NVIDIA CUDA 設計的 torch 專案，
無需修改任何一行原始碼，即可使用 Intel Arc / Data Center GPU (XPU) 加速。

快速使用：
    # 在主程式最開頭（任何 import torch 之前）加入一行：
    import torch_xpu_patch; torch_xpu_patch.apply()

    # 或是一次性把補丁安裝進 venv（之後不需要再加那一行）：
    poetry run xpu-patch-install
"""

from .core import apply, unapply, is_applied, get_status

__version__ = "1.0.0"

__all__ = ["apply", "unapply", "is_applied", "get_status"]
