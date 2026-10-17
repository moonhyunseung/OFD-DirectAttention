from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PackedInt4:
    """Frozen offset-binary groupwise INT4 representation."""

    packed: torch.Tensor
    scale: torch.Tensor
    orig_shape: tuple[int, ...]
    group_size: int = 32


def quantize_int4_groupwise(x: torch.Tensor, group_size: int = 32) -> PackedInt4:
    """Reference quantizer matching the OFD offset-binary storage contract.

    The quantizer uses symmetric absmax/7 scaling. Signed q is stored as
    unsigned nibble code q+8, with the low nibble first in each byte.
    """
    if x.shape[-1] % group_size != 0:
        raise ValueError("last dimension must be divisible by group_size")
    if group_size % 2:
        raise ValueError("group_size must be even")

    orig_shape = tuple(x.shape)
    groups = orig_shape[-1] // group_size
    xf = x.float().reshape(-1, groups, group_size)

    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(xf / scale).clamp(-8, 7).to(torch.int16)
    code = (q + 8).to(torch.uint8)

    lo = code[..., 0::2]
    hi = code[..., 1::2]
    packed = (lo | (hi << 4)).contiguous()

    return PackedInt4(
        packed=packed,
        scale=scale.squeeze(-1).to(torch.float16).contiguous(),
        orig_shape=orig_shape,
        group_size=group_size,
    )


def unpack_codes(obj: PackedInt4) -> torch.Tensor:
    """Return unsigned nibble codes C in {0,...,15}, without dequantizing."""
    lo = obj.packed & 0x0F
    hi = (obj.packed >> 4) & 0x0F
    code = torch.stack((lo, hi), dim=-1).reshape(
        *obj.packed.shape[:-1], obj.group_size
    )
    return code


def unpack_signed_int4(obj: PackedInt4) -> torch.Tensor:
    """Return the signed q represented by the offset-binary nibbles."""
    return unpack_codes(obj).to(torch.int16) - 8


def offset_bytes_to_twos_complement(packed: torch.Tensor) -> torch.Tensor:
    """Exact bytewise recoding: flip the sign bit in both nibbles."""
    return packed ^ 0x88


def dequantize_int4_groupwise(
    obj: PackedInt4,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode the stored mathematical state in the requested compute dtype.

    Scale values are already frozen/stored as FP16. Casting those stored
    values to `out_dtype` before multiplication allows high-precision tests
    to evaluate the same decoded-quantized state without adding an unrelated
    float32 multiplication roundoff.
    """
    q = unpack_signed_int4(obj).to(out_dtype)
    scale = obj.scale.to(out_dtype).unsqueeze(-1)
    x = q * scale
    return x.reshape(obj.orig_shape)
