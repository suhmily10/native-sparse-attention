# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.common.utils import (prepare_chunk_indices, prepare_lens,
                                  prepare_token_indices)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from native_sparse_attention.ops.utils import argsort


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK'],
)
@triton.jit
def parallel_nsa_kernel_compression(
    q,
    k,
    scale,
    block_indices,
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    C = tl.cdiv(T, BS)
    ################################
    # 1. lse computation
    ################################
    # max scores for the current block
    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    # lse = log(acc) + m
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i_s in range(0, tl.cdiv(i_t, BS), BS):
        o_s = i_s + tl.arange(0, BS)

        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, C), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [G, BS]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((i_t >= (o_s * BS))[None, :], b_s, float('-inf'))

        # [G]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        # [G, BS]
        b_p = tl.exp(b_s - b_m[:, None])
        # [G]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)

        b_mp = b_m
    # [G]
    b_lse = b_m + tl.log(b_acc)

    ################################
    # 2. topk selection
    ################################
    # [BS]
    b_i = tl.full([BS], -1, dtype=tl.float32)
    o_i = tl.zeros([BS], dtype=tl.int32)
    m_i = tl.arange(0, BS) < BS // 2
    for i_s in range(0, tl.cdiv(i_t, BS), BS):
        o_s = i_s + tl.arange(0, BS)

        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, C), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [G, BS]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((i_t >= (o_s * BS))[None, :], b_s, float('-inf'))

        # [G, BS]
        b_p = tl.exp(b_s - b_lse[:, None])
        # the importance scores of the current block
        # [BS]
        b_i, b_ip = tl.sum(b_p, 0), b_i
        o_i, o_ip = tl.where(o_s <= i_t//BS, o_s, 0), o_i

        b_i, o_i = argsort(b_i, o_i.to(tl.int32), dim=0, descending=False)
        b_i = b_ip * m_i + b_i * (1 - m_i)
        o_i = o_ip * m_i + o_i * (1 - m_i)
        b_i, o_i = argsort(b_i, o_i.to(tl.int32), dim=0, descending=True)

    m_top = tl.arange(0, BS // S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BS // S, S]), 0)

    p_b = tl.make_block_ptr(block_indices + (bos + i_t) * H*S, (H*S,), (1,), (i_h * S,), (S,), (0,))
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o_slc,
    lse_slc,
    o_swa,
    lse_swa,
    scale,
    block_indices,
    block_counts,
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    WS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H*S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o_slc = tl.make_block_ptr(o_slc + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse_slc = lse_slc + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o_slc = tl.zeros([G, BV], dtype=tl.float32)

    b_m_slc = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc_slc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t:
            p_k_slc = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v_slc = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k_slc = tl.load(p_k_slc, boundary_check=(0, 1))
            # [BS, BV]
            b_v_slc = tl.load(p_v_slc, boundary_check=(0, 1))
            # [G, BS]
            b_s_slc = tl.dot(b_q, b_k_slc)
            b_s_slc = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_slc, float('-inf'))

            # [G]
            b_m_slc, b_mp_slc = tl.maximum(b_m_slc, tl.max(b_s_slc, 1)), b_m_slc
            b_r_slc = tl.exp(b_mp_slc - b_m_slc)
            # [G, BS]
            b_p_slc = tl.exp(b_s_slc - b_m_slc[:, None])
            # [G]
            b_acc_slc = b_acc_slc * b_r_slc + tl.sum(b_p_slc, 1)
            # [G, BV]
            b_o_slc = b_o_slc * b_r_slc[:, None] + tl.dot(b_p_slc.to(b_q.dtype), b_v_slc)

            b_mp_slc = b_m_slc
    b_o_slc = b_o_slc / b_acc_slc[:, None]
    b_m_slc += tl.log(b_acc_slc)
    tl.store(p_o_slc, b_o_slc.to(p_o_slc.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse_slc, b_m_slc.to(p_lse_slc.dtype.element_ty))

    if WS > 0:
        p_o_swa = tl.make_block_ptr(o_swa + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
        p_lse_swa = lse_swa + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_o_swa = tl.zeros([G, BV], dtype=tl.float32)

        b_m_swa = tl.full([G], float('-inf'), dtype=tl.float32)
        b_acc_swa = tl.zeros([G], dtype=tl.float32)
        for i_s in range(max(0, i_t - WS + 1), i_t + 1, BS):
            p_k_swa = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v_swa = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k_swa = tl.load(p_k_swa, boundary_check=(0, 1))
            # [BS, BV]
            b_v_swa = tl.load(p_v_swa, boundary_check=(0, 1))
            # [G, BS]
            b_s_swa = tl.dot(b_q, b_k_swa)
            b_s_swa = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_swa, float('-inf'))

            # [G]
            b_m_swa, b_mp_swa = tl.maximum(b_m_swa, tl.max(b_s_swa, 1)), b_m_swa
            b_r_swa = tl.exp(b_mp_swa - b_m_swa)
            # [G, BS]
            b_p_swa = tl.exp(b_s_swa - b_m_swa[:, None])
            # [G]
            b_acc_swa = b_acc_swa * b_r_swa + tl.sum(b_p_swa, 1)
            # [G, BV]
            b_o_swa = b_o_swa * b_r_swa[:, None] + tl.dot(b_p_swa.to(b_q.dtype), b_v_swa)

            b_mp_swa = b_m_swa
        b_o_swa = b_o_swa / b_acc_swa[:, None]
        b_m_swa += tl.log(b_acc_swa)
        tl.store(p_o_swa, b_o_swa.to(p_o_swa.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_lse_swa, b_m_swa.to(p_lse_swa.dtype.element_ty))


@triton.heuristics({
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.jit
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
    i_b, i_t, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))


@triton.jit
def parallel_nsa_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    block_counts,
    offsets,
    token_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ*K
    do += (bos + i_t) * HQ*V
    dq += (i_v * B * T + bos + i_t) * HQ*K
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    block_indices += (bos + i_t) * H*S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BS]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    offsets,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * B*T*H + bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H*M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
            p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)

            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            # [G, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [G]
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

def compression(
    k: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    ### Currently, we set mean pooling as our basic compression function.
    assert k.shape[1] % block_size == 0, "sequence length must be divisible by block size"
    k_cmp = k.view(k.shape[0], block_size, k.shape[1] // block_size, *k.shape[2:]).mean(dim=1)
    return k_cmp

def parallel_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    S = triton.next_power_of_2(block_counts if isinstance(block_counts, int) else block_counts.max().item())
    BS = block_size
    BK = triton.next_power_of_2(K)

    grid = (T, B * H)
    k_cmp = compression(k, BS)
    block_indices = torch.zeros(B, T, H, S, dtype=torch.long, device=q.device)

    parallel_nsa_kernel_compression[grid](
        q=q,
        k=k_cmp,
        scale=scale,
        block_indices=block_indices,
        offsets=offsets,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        S=S,
        BS=BS,
        BK=BK
    )
    return block_indices


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    window_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    WS = window_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (NV, T, B * H)
    o_slc = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse_slc = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    o_swa = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device) if window_size > 0 else None
    lse_swa = torch.empty(B, T, HQ, dtype=torch.float, device=q.device) if window_size > 0 else None

    parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o_slc=o_slc,
        lse_slc=lse_slc,
        o_swa=o_swa,
        lse_swa=lse_swa,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        WS=WS,
        BK=BK,
        BV=BV,
    )
    return o_slc, lse_slc, o_swa, lse_swa


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    offsets: torch.LongTensor,
    block_size: int
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if offsets is not None:
        NS = triton.cdiv(prepare_lens(offsets).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(B, T, H*S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS
    )
    return block_mask


def parallel_nsa_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float32)
    parallel_nsa_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)

    delta = parallel_nsa_bwd_preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (NV, T, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV
    )
    dq = dq.sum(0)

    if offsets is not None:
        chunk_indices = prepare_chunk_indices(offsets, BS)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)

    # [B, T, H, M]
    block_mask = parallel_nsa_block_mask(block_indices, block_counts, offsets, block_size)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        offsets=offsets,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV
    )
    dk = dk.sum(0)
    return dq, dk, dv


class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, window_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o_slc, lse_slc, o_swa, lse_swa = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            window_size=window_size,
            scale=scale,
            offsets=offsets,
            token_indices=token_indices)
        ctx.save_for_backward(q, k, v, o_slc, lse_slc, o_swa, lse_swa)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.window_size = window_size
        ctx.scale = scale
        return o_slc.to(q.dtype), o_swa.to(q.dtype) if o_swa is not None else o_swa

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do_slc, do_swa):
        q, k, v, o_slc, lse_slc, o_swa, lse_swa = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o_slc,
            lse=lse_slc,
            do=do_slc,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            scale=ctx.scale,
            offsets=ctx.offsets,
            token_indices=ctx.token_indices)
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None, None, None, None


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Optional[torch.LongTensor] = None,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (torch.LongTensor):
            Number of selected blocks for each token with shape `[B, T, H]` if `head_first=False` else `[B, H, T]`,
            If not provided, it will defaults to `S` blocks for each token.
            Default: `None`.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_slc, g_swa))
        if block_counts is not None:
            block_counts = rearrange(block_counts, 'b h t -> b t h')

    o_slc, o_swa = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size, window_size, scale, cu_seqlens)
    o = o_slc * g_slc.unsqueeze(-1) + o_swa * g_swa.unsqueeze(-1) if window_size > 0 else o_slc * g_slc.unsqueeze(-1)
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o


def parallel_nsa_with_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_slc: torch.Tensor,  # 添加必需的门控参数
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    window_size: int = 0,
    g_swa: Optional[torch.Tensor] = None,  # 添加可选的门控参数
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_counts (Optional[Union[torch.LongTensor, int]]):
            Number of selected blocks for each query.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=False` else `[B, H, T]`,
            each query can select the same number of blocks.
        block_size (int):
            Selected block size. Default: 64.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    assert block_counts is not None, "block counts must be provided for selection"

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if head_first:
        q, k, v = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v))
        g_slc = rearrange(g_slc, 'b h t -> b t h') if g_slc is not None else None
        g_swa = rearrange(g_swa, 'b h t -> b t h') if window_size > 0 else None
        if not isinstance(block_counts, int):
            block_counts = rearrange(block_counts, 'b h t -> b t h')


    # 添加压缩索引生成步骤
    token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    block_indices = parallel_nsa_compression(
        q=q,
        k=k,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
        offsets=cu_seqlens,
        token_indices=token_indices
    )
    
    # 修改函数调用，传递window_size参数
    o_slc, o_swa = ParallelNSAFunction.apply(
        q, k, v, block_indices, block_counts, 
        block_size, window_size, scale, cu_seqlens
    )
    
    # 根据window_size决定输出合并方式
    if window_size > 0:
        assert g_swa is not None, "当window_size>0时必须提供g_swa参数"
        o = o_slc * g_slc.unsqueeze(-1) + o_swa * g_swa.unsqueeze(-1)
    else:
        o = o_slc * g_slc.unsqueeze(-1) if g_slc is not None else o_slc


    token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    block_indices = parallel_nsa_compression(q, k, block_counts, block_size, scale, cu_seqlens, token_indices)
    o_slc, o_swa = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size, 0, scale, cu_seqlens)
    o = o_slc
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o