# -*- coding: utf-8 -*-

import torch
import triton
import pandas as pd
import matplotlib.pyplot as plt
from flash_attn import flash_attn_func
import time
import threading
import numpy as np
from native_sparse_attention.ops.parallel import parallel_nsa_with_compression

# 导入GPU监控所需的库
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not available. GPU monitoring will be disabled.")
    print("Install with: pip install pynvml")
    PYNVML_AVAILABLE = False

class GPUMonitor:
    def __init__(self, device_id=0, interval=0.1):
        if not PYNVML_AVAILABLE:
            self.enabled = False
            return
            
        self.device_id = device_id
        self.interval = interval
        self.enabled = True
        self.running = False
        self.memory_usage = []
        self.power_usage = []
        
        # 初始化NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
    def _monitor(self):
        while self.running:
            try:
                # 获取内存信息 (以MB为单位)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.memory_usage.append(mem_info.used / 1024 / 1024)
                
                # 获取功率信息 (以瓦特为单位)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.power_usage.append(power)
            except Exception as e:
                print(f"Error in GPU monitoring: {e}")
            
            time.sleep(self.interval)
    
    def start(self):
        if not self.enabled:
            return
            
        self.memory_usage = []
        self.power_usage = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        if not self.enabled:
            return
            
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
    
    def get_stats(self):
        if not self.enabled or not self.memory_usage:
            return {"memory_max_MB": 0, "memory_avg_MB": 0, "power_avg_W": 0, "power_max_W": 0}
            
        mem_max = max(self.memory_usage)
        mem_avg = sum(self.memory_usage) / len(self.memory_usage)
        power_avg = sum(self.power_usage) / len(self.power_usage) if self.power_usage else 0
        power_max = max(self.power_usage) if self.power_usage else 0
        
        return {
            "memory_max_MB": mem_max,
            "memory_avg_MB": mem_avg,
            "power_avg_W": power_avg,
            "power_max_W": power_max
        }
    
    def __del__(self):
        if self.enabled:
            self.stop()
            pynvml.nvmlShutdown()


# 创建全局变量来存储所有数据
gpu_stats_report = {}
perf_data = {}

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
    window_size = 128                       # 匹配 --nsa-sliding-window 0

    
    q = torch.randn(B, T, HQ, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    g_cmp = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
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

    # 初始化GPU监控
    gpu_monitor = GPUMonitor(device_id=0)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    gpu_stats = {}

    if provider == 'nsa':
        gpu_monitor.start()
        results = triton.testing.do_bench(
            lambda: parallel_nsa_with_compression(q, k, v, g_cmp, g_slc, g_swa, block_counts, block_size, window_size)[0],
            quantiles=quantiles
        )
        gpu_monitor.stop()
        gpu_stats = gpu_monitor.get_stats()
    elif provider == 'nsa_bwd':
        # Wrap in a try-except to handle any errors more gracefully
        try:
            gpu_monitor.start()
            results = triton.testing.do_bench(
                lambda: parallel_nsa_with_compression(q, k, v, g_cmp, g_slc, g_swa, block_counts, block_size, window_size)[0].backward(do),
                quantiles=quantiles
            )
            gpu_monitor.stop()
            gpu_stats = gpu_monitor.get_stats()
        except Exception as e:
            gpu_monitor.stop()
            print(f"Error in NSA backward: {e}")
            results = float('inf'), float('inf'), float('inf')
    elif provider == 'flash':
        gpu_monitor.start()
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True),
            quantiles=quantiles
        )
        gpu_monitor.stop()
        gpu_stats = gpu_monitor.get_stats()
    elif provider == 'flash_bwd':
        gpu_monitor.start()
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True).backward(do),
            quantiles=quantiles
        )
        gpu_monitor.stop()
        gpu_stats = gpu_monitor.get_stats()

    torch.cuda.empty_cache()
    
    # 打印GPU监控统计信息
    if gpu_stats:
        print(f"{provider} (T={T}) GPU Stats - "
              f"Memory: {gpu_stats['memory_max_MB']:.1f}MB max, {gpu_stats['memory_avg_MB']:.1f}MB avg | "
              f"Power: {gpu_stats['power_max_W']:.1f}W max, {gpu_stats['power_avg_W']:.1f}W avg")
        
        # 将结果添加到全局统计信息中
        key = f"{provider}_{T}"
        gpu_stats_report[key] = gpu_stats
    
    # 记录性能结果
    perf_time = float('inf')
    if results:
        # 取中位数作为性能指标
        perf_time = results[0]
    
    # 添加到性能数据字典
    perf_data[key] = perf_time
    
    return results


def generate_combined_report():
    """生成合并的性能和GPU统计信息报告"""
    if not gpu_stats_report or not perf_data:
        print("No performance or GPU statistics available")
        return

    # 将所有数据转换为DataFrame
    data = []
    for key, stats in gpu_stats_report.items():
        # 解析键
        parts = key.split('_')
        seq_len = parts[-1]
        provider = '_'.join(parts[:-1])
        
        # 获取对应的性能数据
        perf_time = perf_data.get(key, float('inf'))
        
        data.append({
            'Provider': provider,
            'Sequence Length': int(seq_len),
            'Time (ms)': perf_time,
            'Memory Max (MB)': stats['memory_max_MB'],
            'Memory Avg (MB)': stats['memory_avg_MB'],
            'Power Max (W)': stats['power_max_W'],
            'Power Avg (W)': stats['power_avg_W']
        })
    
    df = pd.DataFrame(data)
    
    # 排序数据
    df = df.sort_values(['Provider', 'Sequence Length'])
    
    print("\n=== Combined Performance and GPU Report ===")
    print(df.to_string(index=False))
    
    # 为每个序列长度创建对比报告
    print("\n=== Performance Comparison by Sequence Length ===")
    
    # 按序列长度分组
    for seq_len in sorted(df['Sequence Length'].unique()):
        seq_data = df[df['Sequence Length'] == seq_len]
        
        # 提取前向传播数据
        nsa_data = seq_data[seq_data['Provider'] == 'nsa']
        flash_data = seq_data[seq_data['Provider'] == 'flash']
        
        # 提取反向传播数据
        nsa_bwd_data = seq_data[seq_data['Provider'] == 'nsa_bwd']
        flash_bwd_data = seq_data[seq_data['Provider'] == 'flash_bwd']
        
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # 前向传播对比
        if not nsa_data.empty and not flash_data.empty:
            nsa_time = nsa_data['Time (ms)'].values[0]
            flash_time = flash_data['Time (ms)'].values[0]
            nsa_mem = nsa_data['Memory Max (MB)'].values[0]
            flash_mem = flash_data['Memory Max (MB)'].values[0]
            nsa_power = nsa_data['Power Max (W)'].values[0]
            flash_power = flash_data['Power Max (W)'].values[0]
            
            time_ratio = nsa_time / flash_time
            mem_ratio = nsa_mem / flash_mem
            power_ratio = nsa_power / flash_power
            
            print(f"Forward Pass Comparison (NSA vs Flash):")
            print(f"  Time: {nsa_time:.4f}ms vs {flash_time:.4f}ms (NSA/Flash = {time_ratio:.2f}x)")
            print(f"  Memory: {nsa_mem:.1f}MB vs {flash_mem:.1f}MB (NSA/Flash = {mem_ratio:.2f}x)")
            print(f"  Power: {nsa_power:.1f}W vs {flash_power:.1f}W (NSA/Flash = {power_ratio:.2f}x)")
        
        # 反向传播对比
        if not nsa_bwd_data.empty and not flash_bwd_data.empty:
            nsa_bwd_time = nsa_bwd_data['Time (ms)'].values[0]
            flash_bwd_time = flash_bwd_data['Time (ms)'].values[0]
            nsa_bwd_mem = nsa_bwd_data['Memory Max (MB)'].values[0]
            flash_bwd_mem = flash_bwd_data['Memory Max (MB)'].values[0]
            nsa_bwd_power = nsa_bwd_data['Power Max (W)'].values[0]
            flash_bwd_power = flash_bwd_data['Power Max (W)'].values[0]
            
            time_bwd_ratio = nsa_bwd_time / flash_bwd_time
            mem_bwd_ratio = nsa_bwd_mem / flash_bwd_mem
            power_bwd_ratio = nsa_bwd_power / flash_bwd_power
            
            print(f"Backward Pass Comparison (NSA vs Flash):")
            print(f"  Time: {nsa_bwd_time:.4f}ms vs {flash_bwd_time:.4f}ms (NSA/Flash = {time_bwd_ratio:.2f}x)")
            print(f"  Memory: {nsa_bwd_mem:.1f}MB vs {flash_bwd_mem:.1f}MB (NSA/Flash = {mem_bwd_ratio:.2f}x)")
            print(f"  Power: {nsa_bwd_power:.1f}W vs {flash_bwd_power:.1f}W (NSA/Flash = {power_bwd_ratio:.2f}x)")


if __name__ == '__main__':
    # 设置为不打印原始数据，我们将使用自己的报告
    benchmark.run(print_data=False)
    # 生成合并报告
    generate_combined_report()
