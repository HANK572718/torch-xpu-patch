"""torch_xpu_patch.lightning_patch

PyTorch Lightning 專用補丁：讓 Lightning Trainer 的 accelerator 自動選擇 XPU。

這個模組直接從 anomalib_api 專案中提取並獨立化，
讓任何使用 Lightning 的專案都可以引入。

使用：
    import torch_xpu_patch
    torch_xpu_patch.apply()               # 先套用 torch 層補丁
    from torch_xpu_patch.lightning_patch import apply_lightning_patch
    apply_lightning_patch()               # 再套用 Lightning 層補丁
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

_lightning_patched: bool = False


# ---------------------------------------------------------------------------
# XPU Accelerator 類
# ---------------------------------------------------------------------------

def _is_xpu_available() -> bool:
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except (RuntimeError, AttributeError):
        return False


def _num_xpu_devices() -> int:
    if not _is_xpu_available():
        return 0
    return torch.xpu.device_count()


class XPUAccelerator:
    """Intel XPU 加速器，供 Lightning Trainer 使用。"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def setup_device(device: torch.device) -> None:
        if not _is_xpu_available():
            raise RuntimeError("XPU is not available")
        torch.xpu.set_device(device)

    def setup(self, trainer: Any) -> None:
        self.set_intel_flags(trainer.local_rank)

    @staticmethod
    def set_intel_flags(local_rank: int = 0) -> None:
        if "XPU_VISIBLE_DEVICES" not in os.environ and local_rank >= 0:
            os.environ["XPU_VISIBLE_DEVICES"] = str(local_rank)

    @staticmethod
    def get_device_stats(device: Union[torch.device, str, int]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if not _is_xpu_available():
            return stats
        if hasattr(torch.xpu, "memory_allocated"):
            stats["allocated_memory"] = torch.xpu.memory_allocated(device) / 1024 ** 2
        if hasattr(torch.xpu, "max_memory_allocated"):
            stats["max_allocated_memory"] = torch.xpu.max_memory_allocated(device) / 1024 ** 2
        return stats

    def teardown(self) -> None:
        if _is_xpu_available():
            torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        from lightning.fabric.utilities.device_parser import _parse_gpu_ids
        return _parse_gpu_ids(devices, include_xpu=True)

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        return _num_xpu_devices()

    @staticmethod
    def is_available() -> bool:
        return _is_xpu_available()

    @classmethod
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        accelerator_registry.register("xpu", cls, description="XPUAccelerator")


# ---------------------------------------------------------------------------
# 設備解析器補丁
# ---------------------------------------------------------------------------

def _get_all_visible_xpu_devices() -> list[int]:
    try:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return []
        return list(range(torch.xpu.device_count()))
    except (RuntimeError, AttributeError):
        return []


def _patched_get_all_available_gpus(
    include_cuda: bool = False,
    include_mps: bool = False,
    include_xpu: bool = False,
) -> list[int]:
    from lightning.fabric.accelerators.cuda import _get_all_visible_cuda_devices
    from lightning.fabric.accelerators.mps import _get_all_available_mps_gpus
    cuda = _get_all_visible_cuda_devices() if include_cuda else []
    mps = _get_all_available_mps_gpus() if include_mps else []
    xpu = _get_all_visible_xpu_devices() if include_xpu else []
    return cuda + mps + xpu


def _patched_sanitize_gpu_ids(
    gpus: list[int],
    include_cuda: bool = False,
    include_mps: bool = False,
    include_xpu: bool = False,
) -> list[int]:
    if sum((include_cuda, include_mps, include_xpu)) == 0:
        raise ValueError("At least one gpu type should be specified!")
    all_available = _patched_get_all_available_gpus(
        include_cuda=include_cuda, include_mps=include_mps, include_xpu=include_xpu
    )
    for gpu in gpus:
        if gpu not in all_available:
            from lightning.fabric.utilities.exceptions import MisconfigurationException
            raise MisconfigurationException(
                f"You requested gpu: {gpus}\n But your machine only has: {all_available}"
            )
    return gpus


def _patched_normalize_parse_gpu_input_to_list(
    gpus: Union[int, list[int], tuple[int, ...]],
    include_cuda: bool,
    include_mps: bool,
    include_xpu: bool,
) -> Optional[list[int]]:
    from collections.abc import MutableSequence
    assert gpus is not None
    if isinstance(gpus, (MutableSequence, tuple)):
        return list(gpus)
    if not gpus:
        return None
    if gpus == -1:
        return _patched_get_all_available_gpus(
            include_cuda=include_cuda, include_mps=include_mps, include_xpu=include_xpu
        )
    return list(range(gpus))


def _patched_parse_gpu_ids(
    gpus: Optional[Union[int, str, list[int]]],
    include_cuda: bool = False,
    include_mps: bool = False,
    include_xpu: bool = False,
) -> Optional[list[int]]:
    if gpus is None or gpus == "" or (isinstance(gpus, str) and gpus.strip() == ""):
        return None
    if not isinstance(gpus, (int, str, list)):
        raise TypeError("`gpus` must be an int, a string, a list of ints or None.")
    if isinstance(gpus, str):
        gpus = gpus.strip()
        if "," in gpus:
            try:
                gpus = [int(x.strip()) for x in gpus.split(",") if x.strip()]
            except ValueError:
                raise ValueError(f"Could not parse GPU indices from: {gpus}")
        else:
            try:
                gpus = int(gpus)
            except ValueError:
                raise ValueError(f"Could not parse GPU index from: {gpus}")
    gpus = _patched_normalize_parse_gpu_input_to_list(
        gpus, include_cuda=include_cuda, include_mps=include_mps, include_xpu=include_xpu
    )
    if gpus is None:
        return None
    if (
        len(gpus) == 0
        and sum((include_cuda, include_mps, include_xpu)) == 1
        and len(_patched_get_all_available_gpus(
            include_cuda=include_cuda, include_mps=include_mps, include_xpu=include_xpu
        )) == 1
    ):
        gpus = [0]
    return _patched_sanitize_gpu_ids(
        gpus, include_cuda=include_cuda, include_mps=include_mps, include_xpu=include_xpu
    )


def _patch_device_parser() -> None:
    import lightning.fabric.utilities.device_parser as m
    m._parse_gpu_ids = _patched_parse_gpu_ids
    m._sanitize_gpu_ids = _patched_sanitize_gpu_ids
    m._normalize_parse_gpu_input_to_list = _patched_normalize_parse_gpu_input_to_list
    m._get_all_available_gpus = _patched_get_all_available_gpus
    m._get_all_visible_xpu_devices = _get_all_visible_xpu_devices


# ---------------------------------------------------------------------------
# 加速器連接器補丁
# ---------------------------------------------------------------------------

def _patched_choose_auto_accelerator() -> str:
    from lightning.pytorch.accelerators.xla import XLAAccelerator
    from lightning.pytorch.accelerators.mps import MPSAccelerator
    from lightning.pytorch.accelerators.cuda import CUDAAccelerator
    from lightning.pytorch.utilities.imports import _habana_available_and_importable

    if XLAAccelerator.is_available():
        return "tpu"
    if _habana_available_and_importable():
        try:
            from lightning_habana import HPUAccelerator
            if HPUAccelerator.is_available():
                return "hpu"
        except ImportError:
            pass
    if XPUAccelerator.is_available():
        return "xpu"
    if MPSAccelerator.is_available():
        return "mps"
    if CUDAAccelerator.is_available():
        return "cuda"
    return "cpu"


def _patched_choose_gpu_accelerator_backend() -> str:
    from lightning.pytorch.accelerators.mps import MPSAccelerator
    from lightning.pytorch.accelerators.cuda import CUDAAccelerator
    from lightning.fabric.utilities.exceptions import MisconfigurationException

    if XPUAccelerator.is_available():
        return "xpu"
    if MPSAccelerator.is_available():
        return "mps"
    if CUDAAccelerator.is_available():
        return "cuda"
    raise MisconfigurationException("No supported gpu backend found!")


def _patch_accelerator_connector() -> None:
    from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector

    _AcceleratorConnector._choose_auto_accelerator = staticmethod(_patched_choose_auto_accelerator)
    _AcceleratorConnector._choose_gpu_accelerator_backend = staticmethod(_patched_choose_gpu_accelerator_backend)

    # Patch _choose_strategy for single XPU device
    original_choose_strategy = _AcceleratorConnector._choose_strategy

    def patched_choose_strategy(self) -> Any:
        from lightning.pytorch.strategies import SingleDeviceStrategy
        from lightning.pytorch.accelerators.cuda import CUDAAccelerator
        from lightning.pytorch.accelerators.mps import MPSAccelerator
        from lightning.pytorch.trainer.connectors.accelerator_connector import _determine_root_gpu_device

        if len(self._parallel_devices) <= 1:
            if isinstance(self._accelerator_flag, (CUDAAccelerator, MPSAccelerator, XPUAccelerator)) or (
                isinstance(self._accelerator_flag, str)
                and self._accelerator_flag in ("cuda", "gpu", "mps", "xpu")
            ):
                device = _determine_root_gpu_device(self._parallel_devices)
            else:
                device = "cpu"
            return SingleDeviceStrategy(device=device)
        return original_choose_strategy(self)

    _AcceleratorConnector._choose_strategy = patched_choose_strategy


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def apply_lightning_patch(verbose: bool = True) -> None:
    """套用 Lightning XPU 補丁。

    必須在建立任何 ``pl.Trainer`` 之前呼叫。
    """
    global _lightning_patched
    if _lightning_patched:
        return

    try:
        import lightning.pytorch.accelerators as accel_module
        accel_module.XPUAccelerator = XPUAccelerator

        _patch_device_parser()

        from lightning.pytorch.accelerators import AcceleratorRegistry
        if "xpu" not in AcceleratorRegistry:
            AcceleratorRegistry.register("xpu", XPUAccelerator, description="XPUAccelerator")

        _patch_accelerator_connector()

        _lightning_patched = True

        if verbose:
            logger.info("[torch_xpu_patch] Lightning XPU 補丁已套用。")
            logger.info("  accelerator='auto' / 'gpu' / 'xpu' 均可使用 Intel XPU。")

    except Exception as e:
        logger.error(f"[torch_xpu_patch] Lightning 補丁套用失敗: {e}")
        raise
