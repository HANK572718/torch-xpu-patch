"""torch_xpu_patch.lightning_patch

PyTorch Lightning 專用補丁：讓 Lightning Trainer 的 accelerator 自動選擇 XPU。

涵蓋範圍：
  - XPUAccelerator 類別（繼承 Lightning 原生 Accelerator）
  - AcceleratorRegistry 註冊
  - 設備解析器（_parse_gpu_ids 等）支援 include_xpu
  - _AcceleratorConnector 自動選擇邏輯（auto / gpu / xpu）
  - _choose_strategy 單設備策略
  - _set_devices_flag_if_auto_passed 互動環境多設備保護
  - _check_config_and_set_final_flags XPU parallel_devices 驗證
  - _check_and_init_precision 混合精度（16-mixed / bf16-mixed）XPU 支援

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

# 在 core patch 可能替換 torch.device 之前先保存原始型別，
# 確保 Lightning 內部的 isinstance(x, torch.device) 始終能正常工作
_OriginalTorchDevice = torch.device


# ---------------------------------------------------------------------------
# XPU 可用性工具
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


def _get_all_visible_xpu_devices() -> list[int]:
    try:
        if not _is_xpu_available():
            return []
        return list(range(torch.xpu.device_count()))
    except (RuntimeError, AttributeError):
        return []


# ---------------------------------------------------------------------------
# 預先載入 torch._dynamo.device_interface，繞過缺少 CUDA symbol 的問題
# ---------------------------------------------------------------------------

def _patch_missing_cuda_symbols() -> None:
    """預先載入 torch._dynamo.device_interface，繞過 torch._C CUDA symbol 問題。

    torch._dynamo.device_interface 的實作是條件式的：
        if torch.cuda._is_compiled():
            from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
        else:
            get_cuda_stream = None

    在 torch 2.9.0+xpu 中，cuda._is_compiled() 應回傳 False，
    所以這個 import 本身不該失敗。但問題出在 lightning 的 import chain 觸發
    torch._dynamo 整個模組，其中某些子模組（convert_frame 等）在 import 時
    做了額外的 CUDA 相關操作。

    最安全的解法：確保在 lightning import 之前，torch._dynamo.device_interface
    已被正確載入（這樣就不會被重新觸發）。
    """
    import sys

    if "torch._dynamo.device_interface" not in sys.modules:
        try:
            import torch._dynamo.device_interface  # noqa: F401
        except Exception:
            pass  # 若失敗也不影響主流程


# ---------------------------------------------------------------------------
# XPUAccelerator 工廠（動態繼承 Lightning 的 Accelerator 基底類）
# ---------------------------------------------------------------------------

def _get_lightning_accelerator_base():
    """取得 lightning.pytorch.accelerators.Accelerator 基底類別。

    優先從 sys.modules 快取取，避免觸發 lightning 的完整 import chain
    （lightning 2.6.x import 時會觸發 torch._dynamo，而 torch 2.9.0+xpu 缺少 CUDA symbol）。
    """
    import sys

    # 優先從快取取
    accel_mod = sys.modules.get("lightning.pytorch.accelerators")
    if accel_mod is not None and hasattr(accel_mod, "Accelerator"):
        return accel_mod.Accelerator

    # 嘗試直接從子模組取（不觸發完整 lightning import）
    try:
        import importlib
        spec = importlib.util.find_spec("lightning.pytorch.accelerators.accelerator")
        if spec:
            accel_base_mod = importlib.util.module_from_spec(spec)
            sys.modules["lightning.pytorch.accelerators.accelerator"] = accel_base_mod
            spec.loader.exec_module(accel_base_mod)
            return accel_base_mod.Accelerator
    except Exception:
        pass

    # 最後才嘗試完整 import（可能失敗）
    from lightning.pytorch.accelerators import Accelerator as _Acc
    return _Acc


def _make_xpu_accelerator_class():
    """動態建立繼承正確基底類別的 XPUAccelerator。

    在模組載入時無法繼承，因為 lightning 可能還沒 import。
    這個工廠在 apply_lightning_patch() 被呼叫時執行，確保基底類別已存在。
    """
    _Accelerator = _get_lightning_accelerator_base()

    class _XPUAccelerator(_Accelerator):
        """Intel XPU 加速器，供 Lightning Trainer 使用。

        支援 Intel Arc Graphics 系列和 Data Center GPU Max Series。
        需要 PyTorch 2.5+ 的原生 XPU 支援。
        """

        def setup_device(self, device: torch.device) -> None:
            if not _is_xpu_available():
                raise RuntimeError("XPU is not available")
            torch.xpu.set_device(device)

        def setup(self, trainer: Any) -> None:
            self.set_intel_flags(trainer.local_rank)

        @staticmethod
        def set_intel_flags(local_rank: int = 0) -> None:
            if "XPU_VISIBLE_DEVICES" not in os.environ and local_rank >= 0:
                os.environ["XPU_VISIBLE_DEVICES"] = str(local_rank)

        def get_device_stats(self, device: Union[torch.device, str, int]) -> dict[str, Any]:
            stats: dict[str, Any] = {}
            if not _is_xpu_available():
                return stats
            if hasattr(torch.xpu, "memory_allocated"):
                stats["allocated_memory"] = torch.xpu.memory_allocated(device) / 1024 ** 2
            if hasattr(torch.xpu, "max_memory_allocated"):
                stats["max_allocated_memory"] = torch.xpu.max_memory_allocated(device) / 1024 ** 2
            return stats

        @property
        def name(self) -> str:
            return "xpu"

        def teardown(self) -> None:
            if _is_xpu_available():
                torch.xpu.empty_cache()

        @staticmethod
        def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
            from lightning.fabric.utilities.device_parser import _parse_gpu_ids
            return _parse_gpu_ids(devices, include_xpu=True)

        @staticmethod
        def get_parallel_devices(devices: list[int]) -> list[torch.device]:
            return [_OriginalTorchDevice("xpu", i) for i in devices]

        @staticmethod
        def auto_device_count() -> int:
            return _num_xpu_devices()

        @staticmethod
        def is_available() -> bool:
            return _is_xpu_available()

        @classmethod
        def register_accelerators(cls, accelerator_registry: Any) -> None:
            accelerator_registry.register("xpu", cls, description="XPUAccelerator")

    _XPUAccelerator.__name__ = "XPUAccelerator"
    _XPUAccelerator.__qualname__ = "XPUAccelerator"
    return _XPUAccelerator


# 模組層級的 XPUAccelerator 引用（在 apply_lightning_patch() 前為 None）
# 外部程式碼 `from torch_xpu_patch.lightning_patch import XPUAccelerator` 在
# apply_lightning_patch() 呼叫後會取到正確的繼承版本
XPUAccelerator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 設備解析器補丁
# ---------------------------------------------------------------------------

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

    if XLAAccelerator.is_available():
        return "tpu"

    # habana 檢查：lightning 2.5.x 與 2.6.x 介面不同，做相容處理
    try:
        from lightning.pytorch.utilities.imports import _habana_available_and_importable
        if _habana_available_and_importable():
            from lightning_habana import HPUAccelerator
            if HPUAccelerator.is_available():
                return "hpu"
    except (ImportError, Exception):
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

    _patch_connector_instance_methods(_AcceleratorConnector)


def _patch_connector_instance_methods(connector_class: type) -> None:
    """補丁連接器的實例方法，涵蓋：
    - _choose_strategy：單設備 XPU 策略
    - _set_devices_flag_if_auto_passed：互動環境多設備自動限制
    - _check_config_and_set_final_flags：XPU parallel_devices 驗證
    - _check_and_init_precision：混合精度 XPU 支援
    """

    # --- _choose_strategy ---
    original_choose_strategy = connector_class._choose_strategy

    def patched_choose_strategy(self) -> Any:
        from lightning.pytorch.strategies import SingleDeviceStrategy
        from lightning.pytorch.accelerators.cuda import CUDAAccelerator
        from lightning.pytorch.accelerators.mps import MPSAccelerator
        from lightning.pytorch.trainer.connectors.accelerator_connector import _determine_root_gpu_device

        if len(self._parallel_devices) <= 1:
            import sys as _sys
            _XPU = getattr(_sys.modules[__name__], "XPUAccelerator", None)
            _xpu_types = (CUDAAccelerator, MPSAccelerator) + ((_XPU,) if _XPU else ())
            if isinstance(self._accelerator_flag, _xpu_types) or (
                isinstance(self._accelerator_flag, str)
                and self._accelerator_flag in ("cuda", "gpu", "mps", "xpu")
            ):
                device = _determine_root_gpu_device(self._parallel_devices)
                # 確保 device 是原生 torch.device 實例
                if device is not None and not isinstance(device, _OriginalTorchDevice):
                    device = _OriginalTorchDevice(str(device))
            else:
                device = "cpu"
            return SingleDeviceStrategy(device=device)
        return original_choose_strategy(self)

    connector_class._choose_strategy = patched_choose_strategy

    # --- _set_devices_flag_if_auto_passed ---
    # 在互動環境（Jupyter 等）且多 XPU 時，自動限制 devices=1，避免多進程問題
    original_set_devices = connector_class._set_devices_flag_if_auto_passed

    def patched_set_devices_flag_if_auto_passed(self) -> None:
        if self._devices_flag != "auto":
            return original_set_devices(self)

        from lightning.pytorch.accelerators.cuda import CUDAAccelerator
        try:
            from lightning.pytorch.utilities.rank_zero import rank_zero_info
            from lightning.pytorch.trainer.connectors.accelerator_connector import _IS_INTERACTIVE

            _XPU = XPUAccelerator
            if (
                _IS_INTERACTIVE
                and isinstance(self.accelerator, (CUDAAccelerator,) + ((_XPU,) if _XPU else ()))
                and self.accelerator.auto_device_count() > 1
            ):
                self._devices_flag = 1
                rank_zero_info(
                    "You are in an interactive environment and are using multiple devices. "
                    "Defaulting to `devices=1` to avoid issues with multiprocessing."
                )
            else:
                self._devices_flag = self.accelerator.auto_device_count()
        except Exception:
            # _IS_INTERACTIVE 可能不存在（版本差異），fallback 到原始邏輯
            return original_set_devices(self)

    connector_class._set_devices_flag_if_auto_passed = patched_set_devices_flag_if_auto_passed

    # --- _check_config_and_set_final_flags ---
    # 當 strategy 的 parallel_devices 是 xpu 時，確保 _accelerator_flag 設為 'xpu'
    original_check_config = connector_class._check_config_and_set_final_flags

    def patched_check_config_and_set_final_flags(
        self,
        strategy: Any,
        accelerator: Any,
        precision: Any,
        plugins: Any,
        sync_batchnorm: bool,
    ) -> None:
        original_check_config(self, strategy, accelerator, precision, plugins, sync_batchnorm)

        from lightning.fabric.utilities.exceptions import MisconfigurationException

        if hasattr(self._strategy_flag, "parallel_devices") and self._strategy_flag.parallel_devices:
            if self._strategy_flag.parallel_devices[0].type == "xpu":
                if self._accelerator_flag and self._accelerator_flag not in ("auto", "xpu", "gpu"):
                    raise MisconfigurationException(
                        f"XPU parallel_devices set through {self._strategy_flag.__class__.__name__} class,"
                        f" but accelerator set to {self._accelerator_flag}, please choose one device type"
                    )
                self._accelerator_flag = "xpu"

    connector_class._check_config_and_set_final_flags = patched_check_config_and_set_final_flags

    # --- _check_and_init_precision ---
    # 支援 16-mixed / bf16-mixed 混合精度在 XPU 上正確初始化
    original_init_precision = connector_class._check_and_init_precision

    def patched_check_and_init_precision(self) -> Any:
        from lightning.pytorch.plugins.precision import MixedPrecision
        from lightning.pytorch.utilities.rank_zero import rank_zero_info

        if self._precision_flag in ("16-mixed", "bf16-mixed"):
            rank_zero_info(
                f"Using {'16bit' if self._precision_flag == '16-mixed' else 'bfloat16'} "
                f"Automatic Mixed Precision (AMP)"
            )
            if self._accelerator_flag == "cpu":
                device = "cpu"
            elif self._accelerator_flag == "xpu" or isinstance(self._accelerator_flag, XPUAccelerator):
                device = "xpu"
            else:
                device = "cuda"
            return MixedPrecision(self._precision_flag, device)

        return original_init_precision(self)

    connector_class._check_and_init_precision = patched_check_and_init_precision


# ---------------------------------------------------------------------------
# 版本相容性檢查
# ---------------------------------------------------------------------------

def check_lightning_version() -> None:
    """檢查 Lightning 版本相容性，對未測試版本發出警告。"""
    try:
        import lightning
        version = lightning.__version__
        # 已驗證相容的版本
        compatible_versions = ["2.5.0", "2.5.1", "2.6.0", "2.6.1"]
        if version not in compatible_versions:
            logger.warning(
                f"[torch_xpu_patch] Lightning {version} 未在相容清單中（已測試：{', '.join(compatible_versions)}），"
                "部分行為可能不同。"
            )
        else:
            logger.debug(f"[torch_xpu_patch] Lightning 版本 {version} 相容。")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 補丁驗證
# ---------------------------------------------------------------------------

def verify_patches() -> dict[str, bool]:
    """驗證所有 Lightning 補丁是否正確套用。

    Returns:
        dict[str, bool]: 各項檢查結果。
    """
    checks: dict[str, bool] = {}
    try:
        import lightning.pytorch.accelerators as accel_module
        checks["XPUAccelerator 已注入到模組"] = hasattr(accel_module, "XPUAccelerator")

        from lightning.pytorch.accelerators import AcceleratorRegistry
        checks["AcceleratorRegistry 包含 'xpu'"] = "xpu" in AcceleratorRegistry

        import lightning.fabric.utilities.device_parser as parser_module
        checks["設備解析器已補丁"] = parser_module._parse_gpu_ids == _patched_parse_gpu_ids
        checks["_get_all_visible_xpu_devices 已注入"] = hasattr(parser_module, "_get_all_visible_xpu_devices")

        from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector
        checks["_choose_auto_accelerator 已替換"] = (
            _AcceleratorConnector._choose_auto_accelerator == _patched_choose_auto_accelerator
            or getattr(_AcceleratorConnector._choose_auto_accelerator, "__func__", None) == _patched_choose_auto_accelerator
        )
    except Exception as e:
        logger.error(f"[torch_xpu_patch] 驗證補丁時發生錯誤: {e}")
        checks["驗證錯誤"] = False
    return checks


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def apply_lightning_patch(verbose: bool = True) -> None:
    """套用 Lightning XPU 補丁。

    必須在建立任何 ``pl.Trainer`` 之前呼叫。
    多次呼叫是安全的（冪等）。

    Args:
        verbose: 是否輸出啟動資訊（透過 logger.info）。
    """
    global _lightning_patched, XPUAccelerator
    if _lightning_patched:
        return

    try:
        # 預先載入 device_interface，繞過缺少 CUDA symbol 的問題
        _patch_missing_cuda_symbols()

        # 版本相容性檢查
        check_lightning_version()

        # 建立繼承正確基底類別的 XPUAccelerator
        XPUAccelerator = _make_xpu_accelerator_class()

        # 同步更新模組層級引用（讓 patched 函式可透過 __name__ 取到）
        import sys
        sys.modules[__name__].XPUAccelerator = XPUAccelerator

        # 注入到 lightning.pytorch.accelerators 模組命名空間
        import lightning.pytorch.accelerators as accel_module
        accel_module.XPUAccelerator = XPUAccelerator

        # 補丁設備解析器
        _patch_device_parser()

        # 註冊到 AcceleratorRegistry
        from lightning.pytorch.accelerators import AcceleratorRegistry
        if "xpu" in AcceleratorRegistry:
            del AcceleratorRegistry["xpu"]
        AcceleratorRegistry.register("xpu", XPUAccelerator, description="XPUAccelerator")

        # 補丁加速器連接器
        _patch_accelerator_connector()

        _lightning_patched = True

        if verbose:
            logger.info("[torch_xpu_patch] Lightning XPU 補丁已套用。")
            logger.info("  accelerator='auto' / 'gpu' / 'xpu' 均可使用 Intel XPU。")
            if _is_xpu_available():
                count = _num_xpu_devices()
                logger.info(f"  偵測到 {count} 個 XPU 設備：")
                for i in range(count):
                    try:
                        logger.info(f"    - Device {i}: {torch.xpu.get_device_name(i)}")
                    except Exception:
                        logger.info(f"    - Device {i}: XPU Device")
            else:
                logger.warning("[torch_xpu_patch] 系統未偵測到可用的 XPU 設備。")

    except Exception as e:
        logger.error(f"[torch_xpu_patch] Lightning 補丁套用失敗: {e}")
        raise


def unapply_lightning_patch() -> None:
    """還原所有 Lightning 補丁（用於測試或臨時切換）。

    注意：此函式僅供測試環境使用，在生產環境移除補丁可能造成不可預期的行為。
    """
    global _lightning_patched
    if not _lightning_patched:
        return

    logger.warning("[torch_xpu_patch] 正在還原 Lightning 補丁...")

    try:
        import lightning.fabric.utilities.device_parser as parser_module
        # 無法精確還原（原始函式在補丁前未快取），重新 import 模組以取得乾淨狀態
        import importlib
        importlib.reload(parser_module)
        logger.debug("[torch_xpu_patch] 設備解析器已還原（模組重載）。")
    except Exception as e:
        logger.warning(f"[torch_xpu_patch] 還原設備解析器失敗: {e}")

    try:
        from lightning.pytorch.accelerators import AcceleratorRegistry
        if "xpu" in AcceleratorRegistry:
            del AcceleratorRegistry["xpu"]
    except Exception:
        pass

    _lightning_patched = False
    logger.info("[torch_xpu_patch] Lightning 補丁已還原。")
