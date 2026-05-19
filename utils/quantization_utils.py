import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Callable, Union, Optional

from utils.tensor_utils import group, degroup
from utils.float_quant_kernel import (
    SUPPORTED_ACTIVATION_FORMATS,
    symmetric_fake_quant,
)


def int_activation_quantization_pre_hook(module_name: str,
                                         quantization_bit: int,
                                         input_group_size: int,
                                         output_dict: Dict[str, torch.Tensor],
                                         activation_format: Optional[str] = None,
                                         ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]:
    def hook(module: nn.Module, input_args: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        quant_input = activation_quantizer(
            input_args=input_args,
            quantization_bit=quantization_bit,
            group_size=input_group_size,
            activation_format=activation_format,
        )

        return (quant_input,)

    return hook


def activation_quantization_pre_hook(module_name: str,
                                     input_group_size: int,
                                     output_dict: Dict[str, torch.Tensor],
                                     quantization_bit: int = 4,
                                     activation_format: Optional[str] = None,
                                     ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]:
    """Forward pre-hook for activation fake-quant (integer or FP8/NVFP4)."""
    return int_activation_quantization_pre_hook(
        module_name=module_name,
        quantization_bit=quantization_bit,
        input_group_size=input_group_size,
        output_dict=output_dict,
        activation_format=activation_format,
    )


def int_quantizer(tensor: torch.Tensor,
                  quantization_bit: int,
                  dim: int,
                  min_val: float,
                  max_val: float,
                  mask: torch.Tensor,
                  method: str) -> torch.Tensor:
    if method == "minmax":
        if min_val is None and max_val is None:
            min_max_tensor = tensor.clone().masked_fill(~mask, float("inf"))
            min_val, _ = min_max_tensor.min(dim=dim, keepdim=True)
            min_max_tensor = min_max_tensor.masked_fill(~mask, float("-inf"))
            max_val, _ = min_max_tensor.max(dim=dim, keepdim=True)

        scale = (max_val - min_val) / (2**quantization_bit - 1)
        scale.clamp_(min=1e-5)

        q = torch.round((tensor - min_val) / scale).clamp(0, 2**quantization_bit - 1)
        quant_tensor = q * scale + min_val
    elif method == "absmax":
        scale = tensor.clone().masked_fill(~mask, float("-inf")).abs().max(dim=dim, keepdim=True)[0]
        q_max = 2**(quantization_bit - 1) - 1
        scale.clamp_(min=1e-5).div_(q_max)
        quant_tensor = tensor.div(scale).round_().mul_(scale)

    return quant_tensor


def _resolve_activation_format(activation_format: Optional[str]) -> Optional[str]:
    if activation_format is None or activation_format in ("int", "integer"):
        return None
    if activation_format not in SUPPORTED_ACTIVATION_FORMATS:
        raise ValueError(
            f"Unknown activation_format {activation_format!r}; "
            f"choose from {SUPPORTED_ACTIVATION_FORMATS} or 'int'"
        )
    return activation_format


def group_postfix(group_size: int) -> str:
    """``group`` / ``nongroup`` from ``input_group_size`` (-1 = embedding-wise)."""
    return "nongroup" if group_size == -1 else "group"


def activation_cache_suffix(activation_format: Optional[str] = None) -> str:
    """
    Filename suffix for R1 checkpoints tied to activation fake-quant during AOS.

    Examples: ``fp8_e4m3`` -> ``_act_fp8e4m3``; integer (default) -> ``""``.
    """
    fmt = _resolve_activation_format(activation_format)
    if fmt is None:
        return ""
    return f"_act_{fmt.replace('_', '')}"


def r1_checkpoint_filename(
    model_type: str,
    input_group_size: int,
    activation_format: Optional[str] = None,
) -> str:
    postfix = group_postfix(input_group_size)
    act_suffix = activation_cache_suffix(activation_format)
    return f"{model_type}_r1_{postfix}{act_suffix}.pt"


def float_group_quantizer(tensor: torch.Tensor,
                          mask: torch.Tensor,
                          activation_format: str,
                          dim: int = -1,
                          epsilon: float = 1e-12) -> torch.Tensor:
    """Symmetric FP8/NVFP4 fake-quant per group row (same grouping as int activation quant)."""
    absmax = tensor.clone().masked_fill(~mask, float("-inf")).abs().amax(dim=dim, keepdim=True)
    absmax = absmax.clamp(min=epsilon)
    return symmetric_fake_quant(tensor, absmax, activation_format, epsilon=0.0)


def activation_quantizer(input_args: Tuple[torch.Tensor, ...],
                         group_size: int = 1024,
                         quantization_bit: int = 4,
                         activation_format: Optional[str] = None) -> torch.Tensor:
    fmt = _resolve_activation_format(activation_format)
    org_shape = input_args[0].shape

    input_tensor = input_args[0].reshape(-1, org_shape[-1])
    group_input_tensor, mask = group(input_tensor, group_size=group_size)
    if fmt is None:
        quant_input_tensor = int_quantizer(
            group_input_tensor, quantization_bit, -1, None, None, mask, method="minmax"
        )
    else:
        quant_input_tensor = float_group_quantizer(
            group_input_tensor, mask, fmt, dim=-1
        )
    quant_input_tensor = degroup(quant_input_tensor, group_size, input_tensor.shape)
    return quant_input_tensor.reshape(org_shape)


def weight_quantizer(weight: torch.Tensor,
                     group_size: int = 1024,
                     quantization_bit: int = 4) -> torch.Tensor:
    group_weight, mask = group(weight, group_size)
    quant_group_weight = int_quantizer(group_weight, quantization_bit, -1, None, None, mask, method="absmax")
    quant_weight = degroup(quant_group_weight, group_size, weight.shape)

    return quant_weight
