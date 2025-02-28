# -*- coding: utf-8 -*-

import torch
import triton
from flash_attn import flash_attn_func

from native_sparse_attention.ops.parallel import parallel_nsa


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[2048 * 2 ** i for i in range(0, 3)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['nsa', 'flash', 'nsa_bwd', 'flash_bwd'],
        # label name for the lines
        line_names=['nsa', 'flash', 'nsa_bwd', 'flash_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'),
                ('blue', 'dotted'), ('red', 'dotted'), ('cyan', '-'), ('cyan', 'dotted')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    device = 'cuda'
    dtype = torch.bfloat16
    requires_grad = True
    
    # 修改为run_local.sh中的模型配置
    B, H, HQ, D, S = 2, 8, 128, 64, 8       
    block_size = 32                         # 匹配 --nsa-block-size 32
    window_size = 128                         # 匹配 --nsa-sliding-window 0

    
    q = torch.randn(B, T, HQ, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    do = torch.ones_like(q, dtype=dtype)

    # Make sure T is a power of 2 for block indices calculation
    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                # Limit the maximum number of blocks to avoid out-of-bounds issues
                max_blocks = min(triton.cdiv(t, block_size), S)
                if max_blocks > 0:
                    i_i = torch.randperm(max_blocks)[:min(S, max_blocks)]
                    block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (B, T, H), device=device)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'nsa':
        results = triton.testing.do_bench(
            lambda: parallel_nsa(q, k, v, g_slc, g_swa, block_indices, block_counts, block_size, window_size),
            quantiles=quantiles
        )
    elif provider == 'nsa_bwd':
        # Wrap in a try-except to handle any errors more gracefully
        try:
            results = triton.testing.do_bench(
                lambda: parallel_nsa(q, k, v, g_slc, g_swa, block_indices, block_counts, block_size, window_size).backward(do),
                quantiles=quantiles
            )
        except Exception as e:
            print(f"Error in NSA backward: {e}")
            results = float('inf'), float('inf'), float('inf')
    elif provider == 'flash':
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True),
            quantiles=quantiles
        )
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True).backward(do),
            quantiles=quantiles
        )

    torch.cuda.empty_cache()
    
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
