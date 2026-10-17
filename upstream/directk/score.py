from __future__ import annotations

import torch

from upstream.directk.quant import PackedInt4, unpack_codes


def direct_packed_score(
    q: torch.Tensor,
    k_packed: PackedInt4,
    alpha: float,
    *,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Algebraic DirectK reference.

    This function intentionally does not materialize dense dequantized K.
    It still unpacks nibbles into logical integer codes, so it is a
    correctness reference rather than a hardware-optimized packed kernel.

    Shapes:
      q: [r, d]
      codes: [L, G, m]
      scales: [L, G]
    """
    if q.ndim != 2:
        raise ValueError("q must have shape [r,d]")

    dtype = q.dtype if compute_dtype is None else compute_dtype
    group_size = k_packed.group_size
    d = q.shape[-1]
    if d % group_size:
        raise ValueError("q feature dimension must be divisible by group_size")
    groups = d // group_size

    codes = unpack_codes(k_packed)
    if codes.ndim != 3:
        raise ValueError("reference expects packed K originating from [L,d]")
    if codes.shape[1:] != (groups, group_size):
        raise ValueError("packed K geometry does not match q")

    qg = q.to(dtype).reshape(q.shape[0], groups, group_size)
    cg = codes.to(dtype)
    scale = k_packed.scale.to(dtype)

    # A[h,g] = sum_j Q[h,g,j]
    a = qg.sum(dim=-1)

    # R[h,t,g] = sum_j Q[h,g,j] * C[t,g,j]
    r = torch.einsum("hgj,tgj->htg", qg, cg)

    # Z[h,t] = alpha * sum_g S[t,g] * (R[h,t,g] - 8*A[h,g])
    z = ((r - 8.0 * a[:, None, :]) * scale[None, :, :]).sum(dim=-1)
    return z * alpha


def direct_signed_score_from_qint(
    q: torch.Tensor,
    signed_qint: torch.Tensor,
    scale: torch.Tensor,
    alpha: float,
    *,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Exact signed normal form from logical signed INT4 values.

    This exposes the correction-free algebraic form. It does not claim a
    particular machine representation or instruction lowering.
    """
    if q.ndim != 2 or signed_qint.ndim != 3 or scale.ndim != 2:
        raise ValueError("expected q=[r,d], signed_qint=[L,G,m], scale=[L,G]")

    dtype = q.dtype if compute_dtype is None else compute_dtype
    groups, group_size = signed_qint.shape[1:]
    if q.shape[-1] != groups * group_size:
        raise ValueError("feature dimensions do not match")

    qg = q.to(dtype).reshape(q.shape[0], groups, group_size)
    qi = signed_qint.to(dtype)
    s = scale.to(dtype)

    r = torch.einsum("hgj,tgj->htg", qg, qi)
    return (r * s[None, :, :]).sum(dim=-1) * alpha
