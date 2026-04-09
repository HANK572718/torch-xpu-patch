"""test_lightning.py — Lightning 層補丁驗證測試

涵蓋：
  - XPUAccelerator 類別可用性
  - AcceleratorRegistry 註冊
  - accelerator='auto' 自動選 XPU
  - accelerator='xpu' 明確指定
  - accelerator='gpu' 也選 XPU
  - accelerator='cpu' 不受影響（回退正常）
  - 設備解析器 _parse_gpu_ids 支援 include_xpu
  - Lightning 補丁與 torch 補丁可同時作用
  - apply_lightning_patch() 冪等性

執行：
    poetry run python -m torch_xpu_patch.tests.test_lightning
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable


# ---------------------------------------------------------------------------
# 極簡測試框架（與 test_core.py 相同，不依賴 pytest）
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool | None, str]] = []


def test(name: str):
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
    def decorator(fn: Callable):
        _results.append((fn.__name__, None, reason))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 確認 lightning 已安裝
# ---------------------------------------------------------------------------

try:
    import lightning  # noqa: F401
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

if not LIGHTNING_AVAILABLE:
    print("[WARN] lightning 未安裝，所有 Lightning 測試將被跳過。")
    print("       執行: poetry install --with lightning")

# ---------------------------------------------------------------------------
# 套用補丁（torch 層 + Lightning 層）
# ---------------------------------------------------------------------------

import torch_xpu_patch
torch_xpu_patch.apply(verbose=False)

if LIGHTNING_AVAILABLE:
    from torch_xpu_patch.lightning_patch import apply_lightning_patch
    apply_lightning_patch(verbose=False)
    # apply 後才能取到繼承正確基底的版本
    import torch_xpu_patch.lightning_patch as _lp_module
    XPUAccelerator = _lp_module.XPUAccelerator
else:
    XPUAccelerator = None

import torch
XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()


# ---------------------------------------------------------------------------
# 測試群組 1：XPUAccelerator 類別
# ---------------------------------------------------------------------------

if not LIGHTNING_AVAILABLE:
    @skip("lightning 未安裝")
    def _xpu_accelerator_class(): pass
else:
    @test("XPUAccelerator.is_available() 回傳布林值")
    def _():
        result = XPUAccelerator.is_available()
        assert isinstance(result, bool), f"回傳型別: {type(result)}"

    @test("XPUAccelerator.auto_device_count() 回傳非負整數")
    def _():
        count = XPUAccelerator.auto_device_count()
        assert isinstance(count, int) and count >= 0, f"device count: {count}"

    @test("XPUAccelerator 可實例化")
    def _():
        acc = XPUAccelerator()
        assert acc is not None


# ---------------------------------------------------------------------------
# 測試群組 2：AcceleratorRegistry 註冊
# ---------------------------------------------------------------------------

if not LIGHTNING_AVAILABLE:
    @skip("lightning 未安裝")
    def _registry(): pass
else:
    @test("AcceleratorRegistry 包含 'xpu' 鍵")
    def _():
        from lightning.pytorch.accelerators import AcceleratorRegistry
        assert "xpu" in AcceleratorRegistry, f"可用的加速器: {list(AcceleratorRegistry.keys())}"

    @test("AcceleratorRegistry['xpu'] 回傳 XPUAccelerator 類別")
    def _():
        from lightning.pytorch.accelerators import AcceleratorRegistry
        registered_cls = AcceleratorRegistry["xpu"]["accelerator"]
        assert registered_cls is XPUAccelerator, f"registered: {registered_cls}"

    @test("lightning.pytorch.accelerators 模組有 XPUAccelerator 屬性")
    def _():
        import lightning.pytorch.accelerators as accel_module
        assert hasattr(accel_module, "XPUAccelerator"), "XPUAccelerator 未注入到模組"


# ---------------------------------------------------------------------------
# 測試群組 3：設備解析器
# ---------------------------------------------------------------------------

if not LIGHTNING_AVAILABLE:
    @skip("lightning 未安裝")
    def _device_parser(): pass
else:
    @test("_parse_gpu_ids 支援 include_xpu 參數（不拋 TypeError）")
    def _():
        from lightning.fabric.utilities.device_parser import _parse_gpu_ids
        import inspect
        sig = inspect.signature(_parse_gpu_ids)
        assert "include_xpu" in sig.parameters, \
            f"_parse_gpu_ids 缺少 include_xpu 參數，現有: {list(sig.parameters)}"

    @test("_get_all_visible_xpu_devices 已注入到 device_parser 模組")
    def _():
        import lightning.fabric.utilities.device_parser as m
        assert hasattr(m, "_get_all_visible_xpu_devices"), \
            "_get_all_visible_xpu_devices 未注入"

    if XPU_AVAILABLE:
        @test("_parse_gpu_ids(1, include_xpu=True) 回傳 [0]")
        def _():
            from lightning.fabric.utilities.device_parser import _parse_gpu_ids
            result = _parse_gpu_ids(1, include_xpu=True)
            assert result == [0], f"回傳: {result}"
    else:
        @skip("XPU 不可用")
        def _parse_xpu_ids(): pass


# ---------------------------------------------------------------------------
# 測試群組 4：Trainer accelerator 選擇
# ---------------------------------------------------------------------------

if not LIGHTNING_AVAILABLE:
    @skip("lightning 未安裝")
    def _trainer_auto(): pass

    @skip("lightning 未安裝")
    def _trainer_xpu(): pass

    @skip("lightning 未安裝")
    def _trainer_gpu(): pass

    @skip("lightning 未安裝")
    def _trainer_cpu(): pass
else:
    import lightning.pytorch as pl

    @test("Trainer(accelerator='cpu') → CPUAccelerator（補丁不影響 CPU）")
    def _():
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, logger=False, enable_checkpointing=False)
        acc_type = type(trainer.accelerator).__name__
        assert acc_type == "CPUAccelerator", f"加速器: {acc_type}"

    if not XPU_AVAILABLE:
        @skip("XPU 不可用")
        def _trainer_auto_xpu(): pass

        @skip("XPU 不可用")
        def _trainer_explicit_xpu(): pass

        @skip("XPU 不可用")
        def _trainer_gpu_xpu(): pass
    else:
        @test("Trainer(accelerator='auto') → XPUAccelerator")
        def _():
            trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=1, logger=False, enable_checkpointing=False)
            acc_type = type(trainer.accelerator).__name__
            assert acc_type == "XPUAccelerator", f"加速器: {acc_type}"

        @test("Trainer(accelerator='xpu') → XPUAccelerator")
        def _():
            trainer = pl.Trainer(accelerator="xpu", devices=1, max_epochs=1, logger=False, enable_checkpointing=False)
            acc_type = type(trainer.accelerator).__name__
            assert acc_type == "XPUAccelerator", f"加速器: {acc_type}"

        @test("Trainer(accelerator='gpu') → XPUAccelerator（XPU 優先於 CUDA）")
        def _():
            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False, enable_checkpointing=False)
            acc_type = type(trainer.accelerator).__name__
            assert acc_type == "XPUAccelerator", f"加速器: {acc_type}"

        @test("Trainer 使用 XPU 時 root_device 是原生 torch.device 且 type == 'xpu'")
        def _():
            import torch as _torch
            trainer = pl.Trainer(accelerator="xpu", devices=1, max_epochs=1, logger=False, enable_checkpointing=False)
            root_device = trainer.strategy.root_device
            # 確認是原生 torch.device（isinstance 必須有效）
            assert isinstance(root_device, _torch.device), \
                f"root_device 型別: {type(root_device)}"
            assert root_device.type == "xpu", f"root_device: {root_device}"


# ---------------------------------------------------------------------------
# 測試群組 5：實際訓練小迴圈（端到端）
# ---------------------------------------------------------------------------

if not LIGHTNING_AVAILABLE or not XPU_AVAILABLE:
    @skip("lightning 未安裝或 XPU 不可用")
    def _e2e_training(): pass
else:
    @test("XPU 上的端到端訓練迴圈（1 epoch，驗證 training_step 中設備正確）")
    def _():
        import torch
        import lightning.pytorch as pl
        from torch.utils.data import DataLoader, TensorDataset

        _step_devices: list[str] = []

        class _TinyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(4, 2)

            def forward(self, x):
                return self.layer(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                # 記錄 training_step 執行時的設備
                _step_devices.append(x.device.type)
                _step_devices.append(next(self.parameters()).device.type)
                loss = torch.nn.functional.mse_loss(self(x), y)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=1e-3)

        dataset = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))
        loader = DataLoader(dataset, batch_size=4)

        model = _TinyModel()
        trainer = pl.Trainer(
            accelerator="xpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)

        # 驗證 training_step 中 batch 和 model 都在 xpu 上
        assert len(_step_devices) > 0, "training_step 未被執行"
        non_xpu = [d for d in _step_devices if d != "xpu"]
        assert not non_xpu, f"training_step 中發現非 XPU 設備: {non_xpu}"


# ---------------------------------------------------------------------------
# 測試群組 6：apply_lightning_patch() 冪等性
# ---------------------------------------------------------------------------

if not LIGHTNING_AVAILABLE:
    @skip("lightning 未安裝")
    def _lightning_idempotent(): pass
else:
    @test("apply_lightning_patch() 多次呼叫不拋出例外")
    def _():
        apply_lightning_patch(verbose=False)
        apply_lightning_patch(verbose=False)
        # 確認 registry 仍只有一個 xpu
        from lightning.pytorch.accelerators import AcceleratorRegistry
        assert "xpu" in AcceleratorRegistry


# ---------------------------------------------------------------------------
# 印出結果
# ---------------------------------------------------------------------------

def _print_results():
    print("\n" + "=" * 70)
    print("  test_lightning.py  — Lightning 層補丁測試結果")
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
