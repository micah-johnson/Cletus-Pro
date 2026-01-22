"""
Procedural Connectivity for Spiking Neural Networks.

Instead of storing explicit target indices, regenerates connections
deterministically from (src_neuron, seed). Only stores learned weights.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ProceduralConnectivity(nn.Module):
    """
    Procedural connectivity that regenerates target indices deterministically.

    For each source neuron, generates a fixed set of target neurons using
    a deterministic hash function. Only the weights are stored and learned.
    """

    def __init__(
        self,
        num_src: int,
        num_dst: int,
        fan_out: int,
        seed: int = 42,
        device: torch.device = None
    ):
        """
        Args:
            num_src: Number of source neurons
            num_dst: Number of destination neurons
            fan_out: Number of outgoing connections per source neuron
            seed: Random seed for deterministic connection generation
            device: PyTorch device
        """
        super().__init__()

        self.num_src = num_src
        self.num_dst = num_dst
        self.fan_out = min(fan_out, num_dst)  # Can't connect to more than exist
        self.seed = seed
        self.device = device or torch.device('cpu')

        # Learnable weights: [num_src, fan_out]
        # Initialize with Xavier/Glorot initialization scaled for spiking
        self.weights = nn.Parameter(
            torch.randn(num_src, self.fan_out, device=self.device) * 0.5
        )

        # Pre-compute all target indices for efficiency (registered as buffer)
        self._precompute_targets()

    def _hash_neuron(self, src_neuron: int, connection_idx: int) -> int:
        """
        Deterministic hash function to generate target neuron index.

        Uses a simple but effective mixing function that ensures
        good distribution across target neurons.
        """
        # Combine source, connection index, and seed
        x = src_neuron * 2654435761 + connection_idx * 2246822519 + self.seed

        # Mix bits (similar to MurmurHash finalizer)
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x = (x >> 16) ^ x

        return x % self.num_dst

    def _generate_targets_for_neuron(self, src_neuron: int) -> torch.Tensor:
        """
        Generate target indices for a single source neuron.

        Uses rejection sampling to ensure unique targets.
        """
        targets = []
        seen = set()
        connection_idx = 0

        while len(targets) < self.fan_out:
            target = self._hash_neuron(src_neuron, connection_idx)
            if target not in seen:
                targets.append(target)
                seen.add(target)
            connection_idx += 1

            # Safety check to avoid infinite loop (shouldn't happen with proper hash)
            if connection_idx > self.fan_out * 10:
                # Fill remaining with sequential targets
                for t in range(self.num_dst):
                    if t not in seen and len(targets) < self.fan_out:
                        targets.append(t)
                break

        return torch.tensor(targets, dtype=torch.long, device=self.device)

    def _precompute_targets(self) -> None:
        """Pre-compute all target indices for faster lookup."""
        targets_list = []
        for src in range(self.num_src):
            targets_list.append(self._generate_targets_for_neuron(src))
        # Register as buffer so it moves with the module to GPU
        self.register_buffer('_cached_targets', torch.stack(targets_list))

    def get_targets(self, src_neuron: int) -> torch.Tensor:
        """
        Get target neuron indices for a source neuron.

        Args:
            src_neuron: Index of source neuron

        Returns:
            Tensor of target neuron indices [fan_out]
        """
        if self._cached_targets is not None:
            return self._cached_targets[src_neuron]
        return self._generate_targets_for_neuron(src_neuron)

    def get_targets_batch(self, src_neurons: torch.Tensor) -> torch.Tensor:
        """
        Get targets for multiple source neurons.

        Args:
            src_neurons: Tensor of source neuron indices [batch_size]

        Returns:
            Tensor of target indices [batch_size, fan_out]
        """
        if self._cached_targets is not None:
            return self._cached_targets[src_neurons]

        return torch.stack([
            self._generate_targets_for_neuron(int(src))
            for src in src_neurons
        ])

    def get_weights(self, src_neuron: int) -> torch.Tensor:
        """
        Get weights for connections from a source neuron.

        Args:
            src_neuron: Index of source neuron

        Returns:
            Tensor of weights [fan_out]
        """
        return self.weights[src_neuron]

    def get_weights_batch(self, src_neurons: torch.Tensor) -> torch.Tensor:
        """
        Get weights for multiple source neurons.

        Args:
            src_neurons: Tensor of source neuron indices [batch_size]

        Returns:
            Tensor of weights [batch_size, fan_out]
        """
        return self.weights[src_neurons]

    def forward(self, src_neurons: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: get targets and weights for spiking neurons.

        Args:
            src_neurons: Tensor of source neuron indices that are spiking

        Returns:
            Tuple of (targets, weights) where:
                targets: [num_spikes, fan_out] target neuron indices
                weights: [num_spikes, fan_out] connection weights
        """
        if len(src_neurons) == 0:
            empty = torch.tensor([], device=self.device)
            return empty.view(0, self.fan_out).long(), empty.view(0, self.fan_out)

        targets = self.get_targets_batch(src_neurons)
        weights = self.get_weights_batch(src_neurons)

        return targets, weights


class LayeredConnectivity(nn.Module):
    """
    Manages procedural connectivity between multiple layers.

    For a network with layers [input, hidden, output], creates
    connectivity between adjacent layers.
    """

    def __init__(
        self,
        layer_sizes: list,
        fan_out_ratio: float = 0.5,
        seed: int = 42,
        device: torch.device = None
    ):
        """
        Args:
            layer_sizes: List of neuron counts per layer [input, hidden..., output]
            fan_out_ratio: Fraction of next layer each neuron connects to
            seed: Random seed
            device: PyTorch device
        """
        super().__init__()

        self.layer_sizes = layer_sizes
        self.device = device or torch.device('cpu')

        # Create connectivity for each layer transition
        self.connections = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            num_src = layer_sizes[i]
            num_dst = layer_sizes[i + 1]
            fan_out = max(1, int(num_dst * fan_out_ratio))

            conn = ProceduralConnectivity(
                num_src=num_src,
                num_dst=num_dst,
                fan_out=fan_out,
                seed=seed + i,  # Different seed per layer
                device=device
            )
            self.connections.append(conn)

        # Compute layer offsets for global neuron indexing
        self.layer_offsets = [0]
        for size in layer_sizes[:-1]:
            self.layer_offsets.append(self.layer_offsets[-1] + size)

    def get_layer_for_neuron(self, global_idx: int) -> int:
        """Get which layer a global neuron index belongs to."""
        for i, offset in enumerate(self.layer_offsets[1:]):
            if global_idx < offset:
                return i
        return len(self.layer_sizes) - 1

    def global_to_local(self, global_idx: int) -> Tuple[int, int]:
        """Convert global neuron index to (layer, local_idx)."""
        layer = self.get_layer_for_neuron(global_idx)
        local_idx = global_idx - self.layer_offsets[layer]
        return layer, local_idx

    def local_to_global(self, layer: int, local_idx: int) -> int:
        """Convert (layer, local_idx) to global neuron index."""
        return self.layer_offsets[layer] + local_idx

    def get_connection(self, layer: int) -> ProceduralConnectivity:
        """Get connectivity object for a specific layer transition."""
        return self.connections[layer]
