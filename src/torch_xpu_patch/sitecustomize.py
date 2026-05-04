"""usercustomize.py — 自動注入版本

當這個檔案被放到 .venv/Lib/site-packages/ 目錄時，
Python 每次啟動都會自動執行它（不需要修改任何程式碼）。

安裝方式：
    poetry run xpu-patch-install

手動安裝方式：
    copy 這個檔案到 .venv/Lib/site-packages/usercustomize.py
"""

# 自動套用 XPU 補丁（無任何輸出，避免影響 stdout）
try:
    import torch_xpu_patch as _xpu_patch
    _xpu_patch.apply(verbose=False)

    # 同時套用 Lightning 補丁（如果有安裝 Lightning）
    try:
        from torch_xpu_patch.lightning_patch import apply_lightning_patch as _lp
        _lp(verbose=False)
    except (ImportError, Exception):
        pass

except (ImportError, Exception):
    # 若 torch_xpu_patch 未安裝或有任何問題，靜默跳過
    pass
