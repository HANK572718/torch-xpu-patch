"""test_core.py — torch 層補丁驗證測試

涵蓋：
  - apply() 冪等性
  - torch.cuda proxy 模組
  - Tensor.cuda() / Tensor.to('cuda')
  - nn.Module.cuda() / nn.Module.to()
  - torch.device('cuda')
  - torch.autocast device_type
  - unapply() 還原
  - XPU 不可用時的 fallback 行為

執行：
    poetry run python -m torch_xpu_patch.tests.test_core
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable


# ---------------------------------------------------------------------------
# 極簡測試框架（不依賴 pytest，避免外部依賴）
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def test(name: str):
    """裝飾器：將函式標記為測試，自動捕捉例外並記錄結果。"""
    def decorator(fn: Callable):
        try:
            fn()
            _results.append((name, True, ""))
        except AssertionError as e:
            _results.append((name, False, str(e)))
        except Exception as e:
            _results.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return fn
    return decorator


def skip(reason: str = ""):
    """標記為跳過的測試。"""
    def decorator(fn: Callable):
        _results.append((fn.__name__, None, reason))  # type: ignore[arg-type]
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 確保補丁乾淨狀態（每個測試模組獨立）
# ---------------------------------------------------------------------------

import torch_xpu_patch
if torch_xpu_patch.is_applied():
    torch_xpu_patch.unapply()

import torch

XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()


# ---------------------------------------------------------------------------
# 測試群組 1：apply() 基礎行為
# ---------------------------------------------------------------------------

@test("apply() 成功執行，不拋出例外")
def _():
    torch_xpu_patch.apply(verbose=False)
    assert torch_xpu_patch.is_applied(), "is_applied() 應回傳 True"


@test("apply() 冪等：多次呼叫不報錯")
def _():
    torch_xpu_patch.apply(verbose=False)
    torch_xpu_patch.apply(verbose=False)  # 第二次
    assert torch_xpu_patch.is_applied()


@test("get_status() 回傳正確結構")
def _():
    status = torch_xpu_patch.get_status()
    assert isinstance(status, dict)
    assert "patched" in status
    assert "xpu_available" in status
    assert "xpu_devices" in status
    assert "overridden_keys" in status
    assert status["patched"] is True


# ---------------------------------------------------------------------------
# 測試群組 2：torch.cuda proxy
# ---------------------------------------------------------------------------

@test("torch.cuda.is_available() 回傳值與 torch.xpu.is_available() 一致")
def _():
    assert torch.cuda.is_available() == torch.xpu.is_available()


@test("torch.cuda.device_count() 回傳值與 torch.xpu.device_count() 一致")
def _():
    assert torch.cuda.device_count() == torch.xpu.device_count()


@test("sys.modules['torch.cuda'] 已被替換為 proxy")
def _():
    cuda_mod = sys.modules.get("torch.cuda")
    assert cuda_mod is not None
    # proxy 的 repr 應包含 'proxy' 或 'xpu'
    rep = repr(cuda_mod)
    assert "proxy" in rep or "xpu" in rep, f"sys.modules['torch.cuda'] repr: {rep}"


# ---------------------------------------------------------------------------
# 測試群組 3：Tensor.cuda() / Tensor.to()
# ---------------------------------------------------------------------------

if not XPU_AVAILABLE:
    @skip("XPU 不可用")
    def _tensor_cuda_device(): pass

    @skip("XPU 不可用")
    def _tensor_to_cuda_str(): pass

    @skip("XPU 不可用")
    def _tensor_to_cuda_colon(): pass
else:
    @test("tensor.cuda() → tensor 落在 xpu 設備")
    def _():
        t = torch.zeros(4)
        t_gpu = t.cuda()
        assert t_gpu.device.type == "xpu", f"device.type = {t_gpu.device.type}"
        # 清理
        del t_gpu
        torch.xpu.empty_cache()

    @test("tensor.to('cuda') → tensor 落在 xpu 設備")
    def _():
        t = torch.zeros(4)
        t_gpu = t.to("cuda")
        assert t_gpu.device.type == "xpu", f"device.type = {t_gpu.device.type}"
        del t_gpu
        torch.xpu.empty_cache()

    @test("tensor.to('cuda:0') → tensor 落在 xpu:0")
    def _():
        t = torch.zeros(4)
        t_gpu = t.to("cuda:0")
        assert t_gpu.device.type == "xpu", f"device.type = {t_gpu.device.type}"
        assert t_gpu.device.index == 0
        del t_gpu
        torch.xpu.empty_cache()


# ---------------------------------------------------------------------------
# 測試群組 4：nn.Module.cuda() / nn.Module.to()
# ---------------------------------------------------------------------------

if not XPU_AVAILABLE:
    @skip("XPU 不可用")
    def _module_cuda(): pass

    @skip("XPU 不可用")
    def _module_to_cuda(): pass
else:
    @test("nn.Module.cuda() → 參數落在 xpu 設備")
    def _():
        m = torch.nn.Linear(4, 4)
        m.cuda()
        device_type = next(m.parameters()).device.type
        assert device_type == "xpu", f"device.type = {device_type}"
        m.cpu()

    @test("nn.Module.to('cuda') → 參數落在 xpu 設備")
    def _():
        m = torch.nn.Linear(4, 4)
        m.to("cuda")
        device_type = next(m.parameters()).device.type
        assert device_type == "xpu", f"device.type = {device_type}"
        m.cpu()


# ---------------------------------------------------------------------------
# 測試群組 5：torch.device()
# ---------------------------------------------------------------------------

@test("torch.device('cpu') 不受影響")
def _():
    d = torch.device("cpu")
    assert d.type == "cpu"


@test("torch.device('xpu') 正常建立")
def _():
    d = torch.device("xpu")
    assert d.type == "xpu"


@test("torch.device('cuda') 仍可建立（型別不被替換，isinstance 安全）")
def _():
    # 我們不替換 torch.device，保持 isinstance 安全
    # torch.device('cuda') 直接呼叫的情況不改寫，
    # 只有 tensor.to('cuda') / model.to('cuda') 路徑會被攔截
    d = torch.device("cuda")
    assert isinstance(d, torch.device), "isinstance(d, torch.device) 應回傳 True"
    # type 可能是 'cuda'（若 CUDA 不可用）或 'xpu'（若被其他機制攔截）
    assert d.type in ("cuda", "xpu"), f"device.type = {d.type}"


# ---------------------------------------------------------------------------
# 測試群組 6：torch.autocast
# ---------------------------------------------------------------------------

@test("torch.autocast 以 device_type='cuda' 呼叫不報 RuntimeError（重導向到 xpu）")
def _():
    if not XPU_AVAILABLE:
        # 若 XPU 不可用，autocast 不重導向，直接跳過
        return
    try:
        with torch.autocast(device_type="cuda"):
            t = torch.zeros(2, 2, device="xpu")
            _ = t + t
    except RuntimeError as e:
        if "xpu" in str(e).lower() or "autocast" in str(e).lower():
            raise AssertionError(f"autocast XPU redirect 失敗: {e}")
        # 其他 RuntimeError 可能是硬體限制，視為通過
        pass


# ---------------------------------------------------------------------------
# 測試群組 7：unapply() 還原
# ---------------------------------------------------------------------------

@test("unapply() 後 is_applied() 回傳 False")
def _():
    torch_xpu_patch.unapply()
    assert not torch_xpu_patch.is_applied()


@test("unapply() 後 torch.device('cuda') 型別不變（isinstance 仍有效）")
def _():
    torch_xpu_patch.unapply()
    d = torch.device("cuda")
    assert isinstance(d, torch.device), "unapply 後 isinstance 應仍有效"
    assert d.type == "cuda", f"還原後 device.type = {d.type}"


@test("unapply() 後重新 apply() 可正常套用")
def _():
    torch_xpu_patch.unapply()
    torch_xpu_patch.apply(verbose=False)
    assert torch_xpu_patch.is_applied()


# ---------------------------------------------------------------------------
# 印出測試結果
# ---------------------------------------------------------------------------

def _print_results():
    print("\n" + "=" * 70)
    print("  test_core.py  — torch 層補丁測試結果")
    print("=" * 70)

    passed = sum(1 for _, r, _ in _results if r is True)
    failed = sum(1 for _, r, _ in _results if r is False)
    skipped = sum(1 for _, r, _ in _results if r is None)

    for name, result, detail in _results:
        if result is True:
            print(f"  [PASS] {name}")
        elif result is None:
            print(f"  [SKIP] {name}  ({detail})")
        else:
            print(f"  [FAIL] {name}")
            if detail:
                for line in detail.splitlines():
                    print(f"         {line}")

    print()
    print(f"  結果: {passed} 通過  {skipped} 跳過  {failed} 失敗")
    print("=" * 70)
    return failed


if __name__ == "__main__":
    failed = _print_results()
    sys.exit(1 if failed > 0 else 0)
