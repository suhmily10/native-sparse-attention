import torch
import time
import pandas as pd
import triton
from native_sparse_attention.ops.parallel import parallel_nsa_with_compression
from flash_attn import flash_attn_func

# ... existing code ...

def benchmark_attention(B=4, H=4, HQ=64, D=64, S=16, block_size=64, num_runs=100):
    results = []
    dtype = torch.float16
    device = 'cuda'
    window_size = 256
    
    # Test different sequence lengths
    for T in [128, 256, 512, 1024]:
        # Initialize inputs
        q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
        
        # Set number of blocks to select for each query
        block_counts = S  # Using fixed number S for all queries
        
        # Benchmark NSA
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            g_slc = torch.randn((B, T, HQ), dtype=dtype, device=device)
            g_swa = torch.randn((B, T, HQ), dtype=dtype, device=device) if window_size > 0 else None
    
            out_nsa = parallel_nsa_with_compression(
                q=q, k=k, v=v,
                block_counts=block_counts,
                block_size=block_size,
                scale=None,
                cu_seqlens=None,  # Not using variable length sequences
                head_first=False,
                window_size=window_size,
                g_slc=g_slc,
                g_swa=g_swa
            )
        torch.cuda.synchronize()
        nsa_time = (time.perf_counter() - start) / num_runs


        # Benchmark NSA backward
        grad = torch.randn_like(out_nsa)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            out_nsa.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
        nsa_bwd_time = (time.perf_counter() - start) / num_runs

        # Reshape inputs for Flash Attention
        q_flash = q.transpose(1, 2)  # [B, HQ, T, D]
        k_flash = k.transpose(1, 2)  # [B, H, T, D]
        v_flash = v.transpose(1, 2)  # [B, H, T, D]

        # Benchmark Flash Attention
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            out_flash = flash_attn_func(q_flash, k_flash, v_flash, causal=True)
        torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / num_runs

        # Benchmark Flash Attention backward
        grad_flash = grad.transpose(1, 2)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            out_flash.backward(grad_flash, retain_graph=True)
        torch.cuda.synchronize()
        flash_bwd_time = (time.perf_counter() - start) / num_runs

        results.append({
            'T': T,
            'nsa': nsa_time * 1000,  # Convert to milliseconds
            'nsa_bwd': nsa_bwd_time * 1000,
            'flash': flash_time * 1000,
            'flash_bwd': flash_bwd_time * 1000
        })

    # Create DataFrame and display results
    df = pd.DataFrame(results)
    print("\nPerformance Comparison (time in milliseconds):")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    df = benchmark_attention()