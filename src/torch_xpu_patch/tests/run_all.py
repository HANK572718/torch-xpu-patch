"""run_all.py — 執行所有補丁驗證測試

執行：
    poetry run python -m torch_xpu_patch.tests.run_all

選擇性只跑 torch 層：
    poetry run python -m torch_xpu_patch.tests.test_core

選擇性只跑 Lightning 層：
    poetry run python -m torch_xpu_patch.tests.test_lightning
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_module(module: str) -> tuple[int, str]:
    """執行子模組，回傳 (exit_code, output)。"""
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout + result.stderr


def main():
    print("=" * 70)
    print("  torch-xpu-patch 完整測試套件")
    print("=" * 70)

    suites = [
        ("torch 層補丁", "torch_xpu_patch.tests.test_core"),
        ("Lightning 層補丁", "torch_xpu_patch.tests.test_lightning"),
    ]

    total_failed = 0
    for suite_name, module in suites:
        print(f"\n>>> 執行: {suite_name} ({module})")
        code, output = _run_module(module)
        # 過濾掉無法顯示的字元（主機 console 編碼問題）
        try:
            print(output, end="")
        except UnicodeEncodeError:
            safe = output.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
            print(safe, end="")
        if code != 0:
            total_failed += 1

    print("\n" + "=" * 70)
    if total_failed == 0:
        print("  [OK] 所有測試套件通過！")
    else:
        print(f"  [FAIL] {total_failed} 個測試套件有失敗項目，請查看上方輸出。")
    print("=" * 70)

    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
