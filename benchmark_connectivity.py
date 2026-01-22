"""
Benchmark: Procedural Connectivity vs Dense Matrix Storage

Compares:
1. Procedural (sparse): scatter_add with cached target indices
2. Dense: Full matrix multiplication (traditional approach)
"""

import torch
import torch.nn as nn
import time


class ProceduralLayerCached(nn.Module):
    """Sparse connectivity with CACHED targets (current approach)."""

    def __init__(self, in_features, out_features, fan_out_ratio=0.5, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_out = max(1, int(out_features * fan_out_ratio))
        self.device = device or torch.device('cpu')

        # Weights: [in_features, fan_out]
        self.weights = nn.Parameter(torch.randn(in_features, self.fan_out, device=device) * 0.1)

        # Generate and cache targets
        targets = self._generate_targets()
        self.register_buffer('targets', targets)

        # Memory usage
        self.weight_memory = self.weights.numel() * 4  # float32
        self.target_memory = self.targets.numel() * 8  # int64
        self.total_memory = self.weight_memory + self.target_memory

    def _generate_targets(self):
        """Generate deterministic targets for each input neuron."""
        targets = torch.zeros(self.in_features, self.fan_out, dtype=torch.long, device=self.device)
        for i in range(self.in_features):
            torch.manual_seed(i * 12345)
            perm = torch.randperm(self.out_features, device=self.device)[:self.fan_out]
            targets[i] = perm
        return targets

    def forward(self, x):
        """Forward pass using scatter_add with cached targets."""
        weighted = x.unsqueeze(1) * self.weights
        out = torch.zeros(self.out_features, device=self.device)
        out.scatter_add_(0, self.targets.reshape(-1), weighted.reshape(-1))
        return out


class ProceduralLayerOnTheFly(nn.Module):
    """Sparse connectivity with ON-THE-FLY target generation (true procedural)."""

    def __init__(self, in_features, out_features, fan_out_ratio=0.5, device=None, seed=42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_out = max(1, int(out_features * fan_out_ratio))
        self.device = device or torch.device('cpu')
        self.seed = seed

        # Weights ONLY: [in_features, fan_out]
        self.weights = nn.Parameter(torch.randn(in_features, self.fan_out, device=device) * 0.1)

        # Memory usage - NO target storage!
        self.total_memory = self.weights.numel() * 4  # float32 only

    def _hash_target(self, src, conn_idx):
        """Deterministic hash to get target neuron."""
        x = src * 2654435761 + conn_idx * 2246822519 + self.seed
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x = (x >> 16) ^ x
        return x % self.out_features

    def _generate_targets_for_neuron(self, src):
        """Generate targets for one source neuron."""
        targets = []
        seen = set()
        conn_idx = 0
        while len(targets) < self.fan_out:
            t = self._hash_target(src, conn_idx)
            if t not in seen:
                targets.append(t)
                seen.add(t)
            conn_idx += 1
        return targets

    def forward(self, x):
        """Forward pass regenerating targets on-the-fly."""
        out = torch.zeros(self.out_features, device=self.device)

        # Regenerate targets each forward pass
        targets = torch.zeros(self.in_features, self.fan_out, dtype=torch.long, device=self.device)
        for i in range(self.in_features):
            t = self._generate_targets_for_neuron(i)
            targets[i] = torch.tensor(t, device=self.device)

        weighted = x.unsqueeze(1) * self.weights
        out.scatter_add_(0, targets.reshape(-1), weighted.reshape(-1))
        return out


# Alias for backward compatibility
ProceduralLayer = ProceduralLayerCached


class DenseLayer(nn.Module):
    """Traditional dense fully-connected layer."""

    def __init__(self, in_features, out_features, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device or torch.device('cpu')

        # Full weight matrix: [out_features, in_features]
        self.weights = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.1)

        # Memory usage
        self.total_memory = self.weights.numel() * 4  # float32

    def forward(self, x):
        """Forward pass using matrix multiply."""
        return torch.mv(self.weights, x)


def benchmark_forward(layer, x, num_iterations=1000, warmup=100):
    """Benchmark forward pass."""
    # Warmup
    for _ in range(warmup):
        _ = layer(x)

    torch.cuda.synchronize() if x.is_cuda else None
    start = time.perf_counter()

    for _ in range(num_iterations):
        _ = layer(x)

    torch.cuda.synchronize() if x.is_cuda else None
    elapsed = time.perf_counter() - start

    return elapsed / num_iterations * 1000  # ms


def benchmark_backward(layer, x, num_iterations=1000, warmup=100):
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        out = layer(x)
        loss = out.sum()
        loss.backward()

    torch.cuda.synchronize() if x.is_cuda else None
    start = time.perf_counter()

    for _ in range(num_iterations):
        out = layer(x)
        loss = out.sum()
        loss.backward()

    torch.cuda.synchronize() if x.is_cuda else None
    elapsed = time.perf_counter() - start

    return elapsed / num_iterations * 1000  # ms


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 80)

    # Test configurations: (in_features, out_features, fan_out_ratio)
    configs = [
        (40, 256, 0.5),    # 2-digit addition input->hidden
        (100, 1000, 0.1),  # Sparse
        (100, 1000, 0.5),  # Medium
        (1000, 1000, 0.1), # Large sparse
    ]

    for in_f, out_f, ratio in configs:
        print(f"\n{'='*80}")
        print(f"Config: {in_f} -> {out_f} (fan_out_ratio={ratio:.0%})")
        print(f"{'='*80}")

        # Create layers
        dense = DenseLayer(in_f, out_f, device)
        cached = ProceduralLayerCached(in_f, out_f, ratio, device)
        onthefly = ProceduralLayerOnTheFly(in_f, out_f, ratio, device)

        # Input
        x = torch.randn(in_f, device=device)

        # Memory
        dense_mem = dense.total_memory / 1024
        cached_mem = cached.total_memory / 1024
        onthefly_mem = onthefly.total_memory / 1024

        # Benchmark (fewer iterations for on-the-fly since it's slow)
        n_iter = 500
        dense_fwd = benchmark_forward(dense, x, n_iter, 50)
        cached_fwd = benchmark_forward(cached, x, n_iter, 50)
        onthefly_fwd = benchmark_forward(onthefly, x, 100, 10)  # Fewer iterations

        print(f"\n{'Method':<25} {'Memory':<15} {'Forward (ms)':<15} {'Mem vs Dense':<15}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
        print(f"{'Dense':<25} {dense_mem:>8.1f} KB    {dense_fwd:>10.4f}     {'baseline':>12}")
        print(f"{'Procedural (cached)':<25} {cached_mem:>8.1f} KB    {cached_fwd:>10.4f}     {cached_mem/dense_mem:>10.2f}x")
        print(f"{'Procedural (on-the-fly)':<25} {onthefly_mem:>8.1f} KB    {onthefly_fwd:>10.4f}     {onthefly_mem/dense_mem:>10.2f}x")

    print(f"\n{'='*80}")
    print("SUMMARY:")
    print("- Dense: Fastest (optimized matmul), but O(in × out) memory")
    print("- Cached: Medium speed, memory = weights + int64 targets")
    print("- On-the-fly: Slowest, but minimum memory (weights only)")
    print("  → Only stores in × fan_out floats, regenerates targets each pass")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
