"""torch_xpu_patch.core

核心補丁邏輯：將 torch.cuda.* API 全部重導向到 torch.xpu.*，
並攔截 tensor/module 的 .cuda() 呼叫，讓它們落到 XPU 上。

設計原則：
- 零侵入：不修改任何已安裝套件的磁碟檔案
- 可撤銷：所有修改都保留原始引用，可以完整還原
- 安全降級：若 XPU 不可用，自動保持原始行為（不強制報錯）
- 冪等：多次呼叫 apply() 只會執行一次
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

# 補丁是否已套用
_patched: bool = False

# 保存所有原始引用，用於還原
_originals: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def apply(verbose: bool = True) -> None:
    """套用所有 XPU 相容補丁。

    必須在 ``import torch`` **之前**，或至少在任何模型/訓練器建立之前呼叫。
    多次呼叫是安全的（冪等）。

    Args:
        verbose: 是否輸出詳細啟動資訊。
    """
    global _patched
    if _patched:
        logger.debug("torch_xpu_patch 已套用，跳過重複呼叫。")
        return

    import torch  # 此時才 import，確保 patch 在 torch 物件上進行

    if verbose:
        _print_banner()

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

    if not xpu_available:
        logger.warning(
            "[torch_xpu_patch] 警告：未偵測到可用的 XPU 設備。\n"
            "  補丁仍會安裝，但所有 CUDA redirect 將 fallback 到原始行為。\n"
            "  請確認已安裝 Intel XPU 版本的 torch，且驅動正常。"
        )

    _patch_cuda_module(torch, xpu_available)
    _patch_tensor_methods(torch, xpu_available)
    _patch_nn_module(torch, xpu_available)
    _patch_device_string(torch, xpu_available)
    _patch_autocast(torch, xpu_available)
    _patch_amp_grad_scaler(torch, xpu_available)

    _patched = True

    if verbose:
        _print_status(torch, xpu_available)


def unapply() -> None:
    """還原所有補丁（用於測試或臨時切換）。"""
    global _patched
    if not _patched:
        return

    import torch

    for key, original in _originals.items():
        module_path, attr = key.rsplit(".", 1)
        try:
            obj = _resolve(module_path)
            setattr(obj, attr, original)
        except Exception as e:
            logger.warning(f"還原 {key} 失敗: {e}")

    _originals.clear()
    _patched = False
    logger.info("[torch_xpu_patch] 補丁已還原。")


def is_applied() -> bool:
    """回傳補丁是否已套用。"""
    return _patched


def get_status() -> dict[str, Any]:
    """回傳目前補丁狀態與設備資訊。"""
    try:
        import torch
        xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
        devices = []
        if xpu_ok:
            for i in range(torch.xpu.device_count()):
                try:
                    devices.append(torch.xpu.get_device_name(i))
                except Exception:
                    devices.append(f"XPU:{i}")
    except ImportError:
        xpu_ok = False
        devices = []

    return {
        "patched": _patched,
        "xpu_available": xpu_ok,
        "xpu_devices": devices,
        "overridden_keys": list(_originals.keys()),
    }


# ---------------------------------------------------------------------------
# 內部工具
# ---------------------------------------------------------------------------

def _save_and_set(obj: Any, attr: str, new_value: Any, key: str) -> None:
    """保存原始值並設定新值。"""
    if key not in _originals:
        _originals[key] = getattr(obj, attr, None)
    setattr(obj, attr, new_value)


def _resolve(dotted_path: str) -> Any:
    """解析點分隔路徑到物件。例如 'torch.cuda' -> torch.cuda 模組。"""
    parts = dotted_path.split(".")
    obj = sys.modules.get(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# 補丁 1：torch.cuda 模組 → 重導向到 torch.xpu
# ---------------------------------------------------------------------------

def _patch_cuda_module(torch: Any, xpu_available: bool) -> None:
    """讓 torch.cuda.* 的呼叫透明地落到 torch.xpu.*。

    策略：建立一個代理模組 (proxy module)，對所有屬性存取先查 torch.xpu，
    找不到再查原始的 torch.cuda。這樣既相容 XPU，也不會破壞純 CUDA 才有的 API。
    """
    if not xpu_available:
        return

    original_cuda_module = torch.cuda
    xpu_module = torch.xpu

    class _CudaXpuProxy(types.ModuleType):
        """代理模組：優先回傳 torch.xpu 的屬性，回退到 torch.cuda。"""

        def __init__(self):
            super().__init__("torch.cuda")
            self.__spec__ = original_cuda_module.__spec__
            self.__package__ = original_cuda_module.__package__

        def __getattr__(self, name: str) -> Any:
            # 優先從 xpu 取
            if hasattr(xpu_module, name):
                return getattr(xpu_module, name)
            # fallback 到原始 cuda
            return getattr(original_cuda_module, name)

        def __repr__(self) -> str:
            return f"<torch.cuda → torch.xpu proxy (torch_xpu_patch)>"

    proxy = _CudaXpuProxy()

    # 特別處理幾個高頻 API，確保行為正確
    proxy.is_available = xpu_module.is_available
    proxy.device_count = xpu_module.device_count
    proxy.current_device = xpu_module.current_device
    proxy.set_device = xpu_module.set_device
    proxy.get_device_name = xpu_module.get_device_name
    proxy.empty_cache = xpu_module.empty_cache
    proxy.synchronize = xpu_module.synchronize

    if hasattr(xpu_module, "memory_allocated"):
        proxy.memory_allocated = xpu_module.memory_allocated
    if hasattr(xpu_module, "max_memory_allocated"):
        proxy.max_memory_allocated = xpu_module.max_memory_allocated
    if hasattr(xpu_module, "memory_reserved"):
        proxy.memory_reserved = xpu_module.memory_reserved

    # 保留 Stream / Event 若 xpu 有對應實作
    if hasattr(xpu_module, "Stream"):
        proxy.Stream = xpu_module.Stream
    if hasattr(xpu_module, "Event"):
        proxy.Event = xpu_module.Event

    # 掛回 torch.cuda 並更新 sys.modules
    _save_and_set(torch, "cuda", proxy, "torch.cuda")
    _originals["sys.modules[torch.cuda]"] = sys.modules.get("torch.cuda")
    sys.modules["torch.cuda"] = proxy

    logger.debug("[patch] torch.cuda → torch.xpu proxy 已建立")


# ---------------------------------------------------------------------------
# 補丁 2：Tensor.cuda() / Tensor.to('cuda') → xpu
# ---------------------------------------------------------------------------

def _patch_tensor_methods(torch: Any, xpu_available: bool) -> None:
    """攔截 tensor.cuda() 和 tensor.to('cuda:N')，重導向到 XPU。"""
    if not xpu_available:
        return

    Tensor = torch.Tensor
    original_cuda_method = Tensor.cuda
    original_to_method = Tensor.to

    def _xpu_cuda(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        """tensor.cuda() → tensor.xpu()"""
        if device is None:
            return self.xpu(non_blocking=non_blocking, memory_format=memory_format)
        # device 可能是 int 或 'cuda:0'
        if isinstance(device, str) and device.startswith("cuda"):
            device = device.replace("cuda", "xpu")
        elif isinstance(device, int):
            pass  # xpu() 接受 int device index
        return self.xpu(device=device, non_blocking=non_blocking, memory_format=memory_format)

    def _xpu_to(self, *args, **kwargs):
        """tensor.to('cuda') / tensor.to('cuda:0') → tensor.to('xpu') / tensor.to('xpu:0')"""
        args = _rewrite_device_args(args)
        if "device" in kwargs:
            kwargs["device"] = _rewrite_device(kwargs["device"])
        return original_to_method(self, *args, **kwargs)

    _save_and_set(Tensor, "cuda", _xpu_cuda, "torch.Tensor.cuda")
    _save_and_set(Tensor, "to", _xpu_to, "torch.Tensor.to")

    logger.debug("[patch] Tensor.cuda() / Tensor.to() 已重導向至 XPU")


# ---------------------------------------------------------------------------
# 補丁 3：nn.Module.cuda() → xpu
# ---------------------------------------------------------------------------

def _patch_nn_module(torch: Any, xpu_available: bool) -> None:
    """攔截 model.cuda() 和 model.to('cuda')，重導向到 XPU。"""
    if not xpu_available:
        return

    nn = torch.nn
    Module = nn.Module
    original_cuda_method = Module.cuda
    original_to_method = Module.to

    def _module_xpu_cuda(self, device=None):
        """model.cuda() → model.xpu()"""
        if device is None:
            return self.xpu()
        if isinstance(device, str) and device.startswith("cuda"):
            device = device.replace("cuda", "xpu")
        return self.xpu(device)

    def _module_xpu_to(self, *args, **kwargs):
        """model.to('cuda') → model.to('xpu')"""
        args = _rewrite_device_args(args)
        if "device" in kwargs:
            kwargs["device"] = _rewrite_device(kwargs["device"])
        return original_to_method(self, *args, **kwargs)

    _save_and_set(Module, "cuda", _module_xpu_cuda, "torch.nn.Module.cuda")
    _save_and_set(Module, "to", _module_xpu_to, "torch.nn.Module.to")

    logger.debug("[patch] nn.Module.cuda() / Module.to() 已重導向至 XPU")


# ---------------------------------------------------------------------------
# 補丁 4：torch.device('cuda') → torch.device('xpu')
# ---------------------------------------------------------------------------

def _patch_device_string(torch: Any, xpu_available: bool) -> None:
    """讓 torch.device('cuda') 自動變成 torch.device('xpu')。

    torch.device 是 C++ binding，無法繼承。
    改用 callable 工廠函式包裝：呼叫 torch.device('cuda') 時自動改寫參數，
    回傳的仍是原生 torch.device，不影響 isinstance 判斷。
    """
    if not xpu_available:
        return

    OriginalDevice = torch.device

    def _xpu_device_factory(*args, **kwargs):
        """工廠函式：把 'cuda' / 'cuda:N' 改寫為 'xpu' / 'xpu:N' 再建立 device。
        回傳值仍是原生 torch.device 實例，isinstance(x, OriginalDevice) 不受影響。
        """
        args = _rewrite_device_args(args)
        if "device" in kwargs:
            kwargs["device"] = _rewrite_device(kwargs["device"])
        return OriginalDevice(*args, **kwargs)

    _save_and_set(torch, "device", _xpu_device_factory, "torch.device")

    logger.debug("[patch] torch.device('cuda*') → torch.device('xpu*') 已套用")


# ---------------------------------------------------------------------------
# 補丁 5：torch.autocast / torch.cuda.amp.autocast → xpu
# ---------------------------------------------------------------------------

def _patch_autocast(torch: Any, xpu_available: bool) -> None:
    """讓 autocast(device_type='cuda') 自動切換為 xpu。"""
    if not xpu_available:
        return

    original_autocast = torch.autocast

    class _XpuAutocast(original_autocast):
        def __init__(self, device_type: str = "xpu", *args, **kwargs):
            if device_type == "cuda":
                device_type = "xpu"
            super().__init__(device_type, *args, **kwargs)

    _save_and_set(torch, "autocast", _XpuAutocast, "torch.autocast")

    # 同步更新 torch.cuda.amp.autocast（若存在）
    try:
        _save_and_set(torch.cuda.amp, "autocast", _XpuAutocast, "torch.cuda.amp.autocast")
    except AttributeError:
        pass

    logger.debug("[patch] torch.autocast device_type='cuda' → 'xpu' 已套用")


# ---------------------------------------------------------------------------
# 補丁 6：GradScaler → XPU 版本
# ---------------------------------------------------------------------------

def _patch_amp_grad_scaler(torch: Any, xpu_available: bool) -> None:
    """提供 XPU 相容的 GradScaler，讓 AMP 混合精度訓練可用。"""
    if not xpu_available:
        return

    try:
        from torch.cuda.amp import GradScaler as CudaGradScaler

        class _XpuGradScaler(CudaGradScaler):
            """XPU 相容的 GradScaler。

            Intel XPU torch 2.9+ 支援原生 AMP，GradScaler 行為與 CUDA 相同，
            只需讓 device 指向 xpu 即可。
            """
            def __init__(self, *args, **kwargs):
                # 移除 CUDA 專屬 kwargs
                kwargs.pop("init_scale", None)
                super().__init__(*args, **kwargs)

        # 掛到 torch.cuda.amp.GradScaler
        try:
            _save_and_set(torch.cuda.amp, "GradScaler", _XpuGradScaler, "torch.cuda.amp.GradScaler")
        except AttributeError:
            pass

        logger.debug("[patch] GradScaler → XPU 相容版本 已套用")
    except Exception as e:
        logger.debug(f"[patch] GradScaler patch 跳過: {e}")


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _rewrite_device(device: Any) -> Any:
    """把 'cuda' / 'cuda:N' 字串改成 'xpu' / 'xpu:N'。"""
    if isinstance(device, str):
        if device == "cuda" or device.startswith("cuda:"):
            return device.replace("cuda", "xpu", 1)
    return device


def _rewrite_device_args(args: tuple) -> tuple:
    """重寫位置參數中的 device 字串。"""
    if not args:
        return args
    first = args[0]
    rewritten = _rewrite_device(first)
    if rewritten is not first:
        return (rewritten,) + args[1:]
    return args


# ---------------------------------------------------------------------------
# 輸出工具
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print("=" * 70)
    print("  torch-xpu-patch  |  Intel XPU 通用相容補丁")
    print("=" * 70)


def _print_status(torch: Any, xpu_available: bool) -> None:
    status = get_status()
    if xpu_available:
        print(f"  [OK] XPU 可用，偵測到 {len(status['xpu_devices'])} 個設備：")
        for d in status["xpu_devices"]:
            print(f"       - {d}")
        print()
        print("  已套用的重導向：")
        print("    torch.cuda.*          → torch.xpu.*")
        print("    tensor.cuda()         → tensor.xpu()")
        print("    tensor.to('cuda')     → tensor.to('xpu')")
        print("    model.cuda()          → model.xpu()")
        print("    model.to('cuda')      → model.to('xpu')")
        print("    torch.device('cuda')  → torch.device('xpu')")
        print("    torch.autocast('cuda')→ torch.autocast('xpu')")
    else:
        print("  [WARN] XPU 不可用，補丁已安裝但不會重導向。")
    print("=" * 70)
