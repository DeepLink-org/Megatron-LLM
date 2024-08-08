# Extracted from: https://github.com/facebookresearch/llama

from typing import Optional
import torch
from einops import rearrange

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, scaling_factor: float = 1.0
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device).float() / scaling_factor  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
#     position_ids: Optional[torch.Tensor] = None,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

#     freqs_cis = freqs_cis.to(xq.device)
#     if position_ids is None:
#         # we assume position_ids to be torch.arange(seq_len)
#         freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#         # freqs_cis: [seq_len, 1, 1, head_dim//2] (complex64)
#     else:
#         # use specified position_ids, possibly not monotonically increasing
#         # tensor shapes & types:
#         # xq_: [seq_len, batch_size, heads, head_dim//2] (complex64)
#         # position_ids: [batch_size, seq_len] (long)
#         position_ids = position_ids.to(xq.device)   # normally already on correct device
#         assert position_ids.shape == (xq_.shape[1], xq_.shape[0])
#         assert (freqs_cis.shape[1] == xq_.shape[-1])
#         freqs_cis = freqs_cis[position_ids].transpose(0, 1).unsqueeze(-2)
#         # freqs_cis: [seq_len, batch_size, 1, head_dim//2] (complex64)

#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)

def _torch_apply_rotary_func(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool = False,
):
    assert (
        x1.device == x2.device == cos.device == sin.device
    ), "All inputs must be on the same device"
    assert (
        x1.dtype == x2.dtype == cos.dtype == sin.dtype
    ), "All inputs must have the same dtype"
    assert x1.size() == x2.size(), "Input x1 and x2 must have the same sizes"
    assert cos.size() == sin.size(), "Input cos and sin must have the same sizes"

    x1, x2, cos, sin = x1.float(), x2.float(), cos.float(), sin.float()

    if conj:
        out1.copy_(x1 * cos + x2 * sin)
        out2.copy_(-x1 * sin + x2 * cos)
    else:
        out1.copy_(x1 * cos - x2 * sin)
        out2.copy_(x1 * sin + x2 * cos)

    return out1, out2

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    xq_reshape = xq.permute(1, 0, 2, 3)
    xk_reshape = xk.permute(1, 0, 2, 3)
    _, seqlen, _, headdim = xq_reshape.shape
    rotary_seqlen, rotary_dim = freqs_cis.shape
    freqs_cis = freqs_cis.to(xq.device)
    cos = torch.real(freqs_cis).to(dtype=xq.dtype, device=freqs_cis.device)
    sin = torch.imag(freqs_cis).to(dtype=xq.dtype, device=freqs_cis.device)
    rotary_dim *= 2
    assert rotary_dim <= headdim
    assert seqlen <= rotary_seqlen
    assert sin.shape == (rotary_seqlen, rotary_dim // 2)
    xq_ro = xq_reshape[..., :rotary_dim]
    xq1, xq2 = xq_ro[..., ::2], xq_ro[..., 1::2]
    xk_ro = xk_reshape[..., :rotary_dim]
    xk1, xk2 = xk_ro[..., ::2], xk_ro[..., 1::2]

    outq = torch.empty_like(xq_reshape)
    outq_ro = outq[..., :rotary_dim]
    oq1, oq2 = outq_ro[..., ::2], outq_ro[..., 1::2]
    oq1 = oq1.clone()
    oq2 = oq2.clone()

    outk = torch.empty_like(xk_reshape)
    outk_ro = outk[..., :rotary_dim]
    ok1, ok2 = outk_ro[..., ::2], outk_ro[..., 1::2]
    ok1 = ok1.clone()
    ok2 = ok2.clone()

    _torch_apply_rotary_func(
        xq1,
        xq2,
        rearrange(cos[:seqlen], "s d -> s 1 d"),
        rearrange(sin[:seqlen], "s d -> s 1 d"),
        oq1,
        oq2,
        False,
    )

    _torch_apply_rotary_func(
        xk1,
        xk2,
        rearrange(cos[:seqlen], "s d -> s 1 d"),
        rearrange(sin[:seqlen], "s d -> s 1 d"),
        ok1,
        ok2,
        False,
    )

    tempq = torch.empty_like(xq_reshape)
    tempk = torch.empty_like(xk_reshape)

    tempq[..., 0::2] = oq1
    tempq[..., 1::2] = oq2
    tempk[..., 0::2] = ok1
    tempk[..., 1::2] = ok2

    if rotary_dim < headdim:
        tempq[..., rotary_dim:].copy_(xq_reshape[..., rotary_dim:])
        tempk[..., rotary_dim:].copy_(xk_reshape[..., rotary_dim:])

    resq = tempq.permute(1, 0, 2, 3)
    resk = tempk.permute(1, 0, 2, 3)

    return resq.type_as(xq), resk.type_as(xk)
