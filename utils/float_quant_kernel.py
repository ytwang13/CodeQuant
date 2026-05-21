"""
FP8 and NVFP4 fake-quantization kernels (torch).

Adapted from paro-paper/understand/trail/quant_kernel.py for CodeQuant activation quant.

Formats
-------
- ``fp8_e4m3`` / ``fp8_e5m2``: fake quant — explicit E(x)M(y) grid in fp16/fp32.
- ``e4m3`` / ``e5m2``: direct grid — scale then ``.to(float8_*)`` cast.
- ``nvfp4`` / ``nvfp4_plus``: fake quant only (E2M1 grid + optional FP8 scale quant).
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch

Backend = Literal["torch"]

FP8_E4M3 = (4, 3)
FP8_E5M2 = (5, 2)
NVFP4_E2M1 = (2, 1)

FP8_CAST_DTYPE = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
FP8_CAST_MAX = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}

SCALE_MIN_THRES = 1e-10

_FORMAT_SPECS = {
    "fp8_e4m3": ("exmy", FP8_E4M3, "none"),
    "fp8_e5m2": ("exmy", FP8_E5M2, "none"),
    "e4m3": ("cast", None, "none"),
    "e5m2": ("cast", None, "none"),
    "nvfp4": ("nv", NVFP4_E2M1, "e4m3"),
    "nvfp4_plus": ("nv", NVFP4_E2M1, "e4m3_plus"),
}

SUPPORTED_ACTIVATION_FORMATS = tuple(_FORMAT_SPECS.keys())


def _qp_symmetric(e_bit: int, m_bit: int) -> float:
    qp = (2 - 2 ** (-m_bit)) * (2 ** (2 ** (e_bit - 1)))
    if e_bit == 4 and m_bit == 3:
        qp = 448.0
    elif e_bit == 5 and m_bit == 2:
        qp = 57344.0
    return qp


def _parse_format(fmt: str) -> Tuple[str, str, Tuple[int, int] | None, str]:
    if fmt not in _FORMAT_SPECS:
        raise ValueError(
            f"Unknown activation format {fmt!r}; choose from {list(_FORMAT_SPECS)}"
        )
    kind, em_bits, scale_mode = _FORMAT_SPECS[fmt]
    return fmt, kind, em_bits, scale_mode


def float_exmy_quantize_torch(
    x: torch.Tensor,
    e_bit: int,
    m_bit: int,
    stochastic: bool = False,
    ceil: bool = False,
) -> torch.Tensor:
    sign, x_abs = x.sign(), x.abs()
    tiny = x_abs < SCALE_MIN_THRES
    x_abs = x_abs.clamp(min=SCALE_MIN_THRES)
    elow = -(2 ** (e_bit - 1)) + 2
    ehigh = 2 ** (e_bit - 1)
    mhigh = 2**m_bit
    expo = torch.floor(torch.log2(x_abs))
    expo = torch.clamp(expo, min=elow, max=ehigh)
    mant = x_abs / torch.exp2(expo)
    mant_int = torch.floor(mant)
    mant_frac = (mant - mant_int) * mhigh
    if stochastic:
        mant_frac = mant_frac + torch.empty_like(mant_frac).uniform_(-0.5, 0.5)
    if ceil:
        mant_frac = torch.ceil(mant_frac)
    else:
        mant_frac = torch.round(mant_frac)
    mant_q = mant_int + mant_frac / mhigh
    out = (sign * torch.exp2(expo) * mant_q).to(x.dtype)
    return torch.where(tiny, torch.zeros_like(out), out)


def fp8_grid_cast_torch(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt not in FP8_CAST_DTYPE:
        raise ValueError(f"fp8_grid_cast requires {list(FP8_CAST_DTYPE)}; got {fmt!r}")
    fp8_dtype = FP8_CAST_DTYPE[fmt]
    return x.to(fp8_dtype).to(x.dtype)


def _quantize_nv_scale(scale: torch.Tensor, scale_mode: str) -> torch.Tensor:
    if scale_mode == "none":
        return scale
    if scale_mode == "e4m3":
        return float_exmy_quantize_torch(scale, 4, 3, ceil=True)
    if scale_mode == "e4m3_plus":
        double_scale = scale.abs().amax().float() / 448.0
        q = float_exmy_quantize_torch(scale / double_scale, 4, 3, ceil=True)
        return q * double_scale
    raise ValueError(scale_mode)


def _symmetric_scale_from_absmax(
    absmax: torch.Tensor,
    fmt: str,
    kind: str,
    e_bit: int | None,
    m_bit: int | None,
    scale_mode: str,
    epsilon: float,
) -> torch.Tensor:
    if kind == "cast":
        scale = (absmax + epsilon) / FP8_CAST_MAX[fmt]
    else:
        assert e_bit is not None and m_bit is not None
        qp = _qp_symmetric(e_bit, m_bit)
        scale = (2.0 * absmax + epsilon) / (2.0 * qp)
    return _quantize_nv_scale(scale, scale_mode)


def _apply_value_quant(
    normalized: torch.Tensor,
    fmt: str,
    kind: str,
    e_bit: int | None,
    m_bit: int | None,
    stochastic: bool,
) -> torch.Tensor:
    if kind == "cast":
        return fp8_grid_cast_torch(normalized, fmt)
    assert e_bit is not None and m_bit is not None
    return float_exmy_quantize_torch(normalized, e_bit, m_bit, stochastic)


def symmetric_fake_quant(
    x: torch.Tensor,
    absmax: torch.Tensor,
    fmt: str,
    epsilon: float = 1e-12,
    stochastic: bool = False,
) -> torch.Tensor:
    """Symmetric per-slice fake quant given precomputed absmax (same shape as x or broadcastable)."""
    fmt, kind, em_bits, scale_mode = _parse_format(fmt)
    e_bit, m_bit = (em_bits if em_bits is not None else (None, None))
    scales = _symmetric_scale_from_absmax(
        absmax, fmt, kind, e_bit, m_bit, scale_mode, epsilon
    )
    normalized = x / scales
    q = _apply_value_quant(normalized, fmt, kind, e_bit, m_bit, stochastic)
    return q * scales
