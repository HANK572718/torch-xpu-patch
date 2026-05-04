"""torch_xpu_patch.cli

CLI 工具，提供四個命令：

    xpu-patch-install    將 usercustomize.py 安裝到目前 venv，
                         讓後續每次 python 啟動都自動套用 XPU 補丁。

    xpu-patch-uninstall  移除 usercustomize.py。

    xpu-patch-status     顯示目前補丁狀態與 XPU 設備資訊。

    xpu-patch-verify     執行完整的 XPU 功能驗證測試。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _get_site_packages() -> Path:
    """取得目前 Python 環境的 site-packages 路徑。"""
    import site
    try:
        paths = site.getsitepackages()
        # getsitepackages()[0] 在某些 venv 實作中回傳 prefix 根目錄而非 site-packages
        # 找第一個路徑名稱為 site-packages 的項目
        for p in paths:
            if Path(p).name == "site-packages":
                return Path(p)
        if paths:
            return Path(paths[-1])  # 最後一項通常是 site-packages
    except (RuntimeError, AttributeError):
        pass
    # fallback: Windows venv 用 Lib/，Linux 用 lib/pythonX.Y/
    if sys.platform == "win32":
        return Path(sys.prefix) / "Lib" / "site-packages"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return Path(sys.prefix) / "lib" / version / "site-packages"


def _get_usercustomize_template() -> Path:
    """取得 usercustomize.py 模板的路徑。"""
    return Path(__file__).parent / "usercustomize.py"


def install_usercustomize() -> None:
    """將 usercustomize.py 安裝到目前 venv 的 site-packages。"""
    template = _get_usercustomize_template()
    target_dir = _get_site_packages()
    target = target_dir / "usercustomize.py"

    print("=" * 60)
    print("  torch-xpu-patch：安裝自動啟動補丁")
    print("=" * 60)
    print(f"  目標位置: {target}")

    if target.exists():
        # 檢查是否已有其他 usercustomize.py
        content = target.read_text(encoding="utf-8")
        if "torch_xpu_patch" in content:
            print("  [OK] usercustomize.py 已包含 torch_xpu_patch，無需重新安裝。")
            return
        else:
            print("  [WARN] 目標已存在 usercustomize.py（不含 torch_xpu_patch）")
            print("         將在檔案末尾附加 XPU patch 程式碼...")
            append_content = "\n\n# === torch_xpu_patch auto-inject (appended) ===\n"
            append_content += template.read_text(encoding="utf-8")
            with open(target, "a", encoding="utf-8") as f:
                f.write(append_content)
            print("  [OK] 已附加完成。")
            return

    shutil.copy2(template, target)
    print("  [OK] 安裝完成！")
    print()
    print("  效果：此 venv 的所有 python 程式啟動時，")
    print("        會自動套用 torch.cuda → torch.xpu 重導向。")
    print("        無需在任何程式碼中加入 import。")
    print("=" * 60)


def uninstall_usercustomize() -> None:
    """移除 usercustomize.py（或移除其中的 torch_xpu_patch 區段）。"""
    target_dir = _get_site_packages()
    target = target_dir / "usercustomize.py"

    print("=" * 60)
    print("  torch-xpu-patch：移除自動啟動補丁")
    print("=" * 60)

    if not target.exists():
        print("  [INFO] usercustomize.py 不存在，無需移除。")
        return

    content = target.read_text(encoding="utf-8")
    if "torch_xpu_patch" not in content:
        print("  [INFO] usercustomize.py 不含 torch_xpu_patch，無需移除。")
        return

    # 如果整個檔案都是我們的（以 usercustomize.py 模板開頭）
    template = _get_usercustomize_template()
    template_content = template.read_text(encoding="utf-8")
    if content.strip() == template_content.strip():
        target.unlink()
        print("  [OK] usercustomize.py 已刪除。")
        return

    # 否則只移除附加的部分
    marker = "# === torch_xpu_patch auto-inject (appended) ==="
    if marker in content:
        new_content = content.split(marker)[0].rstrip()
        target.write_text(new_content, encoding="utf-8")
        print("  [OK] 已從 usercustomize.py 移除 torch_xpu_patch 區段。")
        return

    print("  [WARN] 無法精確移除，請手動編輯 usercustomize.py。")
    print(f"  檔案位置: {target}")


def show_status() -> None:
    """顯示目前 XPU 補丁狀態。"""
    print("=" * 60)
    print("  torch-xpu-patch 狀態")
    print("=" * 60)

    # 套用補丁（在查詢前）
    try:
        import torch_xpu_patch
        torch_xpu_patch.apply(verbose=False)
        status = torch_xpu_patch.get_status()
    except Exception as e:
        print(f"  [ERROR] 無法載入 torch_xpu_patch: {e}")
        return

    print(f"  補丁已套用: {'是' if status['patched'] else '否'}")
    print(f"  XPU 可用:   {'是' if status['xpu_available'] else '否'}")

    if status["xpu_devices"]:
        print(f"  XPU 設備 ({len(status['xpu_devices'])} 個):")
        for d in status["xpu_devices"]:
            print(f"    - {d}")
    else:
        print("  XPU 設備: 無")

    print()
    print("  已覆蓋的 API：")
    for key in status["overridden_keys"]:
        print(f"    {key}")

    # 檢查 usercustomize.py
    target = _get_site_packages() / "usercustomize.py"
    if target.exists() and "torch_xpu_patch" in target.read_text(encoding="utf-8"):
        print()
        print(f"  [OK] 自動啟動已安裝: {target}")
    else:
        print()
        print("  [INFO] 尚未安裝自動啟動 (執行 xpu-patch-install 可啟用)")

    print("=" * 60)


def verify() -> None:
    """執行 XPU 功能驗證測試。"""
    print("=" * 60)
    print("  torch-xpu-patch 驗證測試")
    print("=" * 60)

    results: dict[str, bool] = {}

    # 測試 1: 基本 import
    try:
        import torch_xpu_patch
        torch_xpu_patch.apply(verbose=False)
        results["套件 import & apply()"] = True
    except Exception as e:
        results["套件 import & apply()"] = False
        print(f"  [FAIL] {e}")

    # 測試 2: torch.cuda.is_available() 回傳 True
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        xpu_avail = torch.xpu.is_available()
        results["torch.cuda.is_available() == XPU 狀態"] = (cuda_avail == xpu_avail)
    except Exception as e:
        results["torch.cuda.is_available() == XPU 狀態"] = False

    # 測試 3: tensor.cuda() → XPU
    try:
        import torch
        if torch.xpu.is_available():
            t = torch.zeros(3)
            t_gpu = t.cuda()
            results["tensor.cuda() → XPU device"] = t_gpu.device.type == "xpu"
        else:
            results["tensor.cuda() → XPU device"] = None  # 跳過
    except Exception as e:
        results["tensor.cuda() → XPU device"] = False

    # 測試 4: torch.device('cuda') → xpu
    try:
        import torch
        d = torch.device("cuda")
        results["torch.device('cuda') → xpu type"] = d.type == "xpu"
    except Exception as e:
        results["torch.device('cuda') → xpu type"] = False

    # 測試 5: nn.Module.cuda() → XPU
    try:
        import torch
        if torch.xpu.is_available():
            m = torch.nn.Linear(4, 4)
            m_gpu = m.cuda()
            param_device = next(m_gpu.parameters()).device.type
            results["nn.Module.cuda() → XPU device"] = param_device == "xpu"
        else:
            results["nn.Module.cuda() → XPU device"] = None
    except Exception as e:
        results["nn.Module.cuda() → XPU device"] = False

    # 印出結果
    print()
    passed = 0
    skipped = 0
    failed = 0
    for name, result in results.items():
        if result is True:
            print(f"  [PASS] {name}")
            passed += 1
        elif result is None:
            print(f"  [SKIP] {name} (XPU 不可用)")
            skipped += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    print()
    print(f"  結果: {passed} 通過, {skipped} 跳過, {failed} 失敗")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
