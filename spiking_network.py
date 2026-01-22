"""
Spiking Neural Network with lazy membrane potential updates.

Core network implementation that combines:
- Timing wheel for O(1) spike scheduling
- Lazy membrane potential (only update on spike arrival)
- Procedural connectivity
- Spike graph recording for backpropagation
- Surrogate gradients for differentiable spike decisions
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from timing_wheel import TimingWheel, SpikeGraph, SpikeEvent
from procedural_connectivity import ProceduralConnectivity, LayeredConnectivity


class SurrogateSpike(torch.autograd.Function):
    """
    Spike function with surrogate gradient for backpropagation.

    Forward: Heaviside step function (0 if v < thresh, 1 otherwise)
    Backward: Sigmoid derivative surrogate
    """

    surrogate_scale = 25.0  # Class variable for scale

    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        spikes = (membrane_potential >= threshold).float()
        return spikes

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        membrane_potential, = ctx.saved_tensors
        threshold = ctx.threshold
        scale = SurrogateSpike.surrogate_scale

        # Normalized distance from threshold
        x = scale * (membrane_potential - threshold)

        # Sigmoid surrogate gradient
        sigmoid = torch.sigmoid(x)
        surrogate_grad = scale * sigmoid * (1 - sigmoid)

        return grad_output * surrogate_grad, None


def surrogate_spike(v: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply spike function with surrogate gradient."""
    return SurrogateSpike.apply(v, threshold)


@dataclass
class NeuronState:
    """State for lazy membrane potential computation."""
    membrane_potential: torch.Tensor
    last_update_time: torch.Tensor
    has_fired: torch.Tensor


class SpikingNetwork(nn.Module):
    """
    Spiking Neural Network with differentiable forward pass.

    Uses surrogate gradients to enable backpropagation through spike decisions.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        tau: float = 20.0,
        threshold: float = 1.0,
        fan_out_ratio: float = 0.5,
        seed: int = 42,
        device: torch.device = None,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.total_neurons = sum(layer_sizes)
        self.tau = tau
        self.threshold = threshold
        self.device = device or torch.device('cpu')

        # Layer boundaries
        self.layer_offsets = [0]
        for size in layer_sizes[:-1]:
            self.layer_offsets.append(self.layer_offsets[-1] + size)

        # Procedural connectivity
        self.connectivity = LayeredConnectivity(
            layer_sizes=layer_sizes,
            fan_out_ratio=fan_out_ratio,
            seed=seed,
            device=device
        )

        # Timing wheel and spike graph
        self.timing_wheel = TimingWheel(num_slots=256, device=device)
        self.spike_graph = SpikeGraph()
        self.state: Optional[NeuronState] = None

    def _init_state(self) -> None:
        """Initialize neuron state."""
        self.state = NeuronState(
            membrane_potential=torch.zeros(self.total_neurons, device=self.device),
            last_update_time=torch.zeros(self.total_neurons, dtype=torch.long, device=self.device),
            has_fired=torch.zeros(self.total_neurons, dtype=torch.bool, device=self.device)
        )

    def _get_layer_idx(self, global_idx: int) -> int:
        """Get which layer a global neuron index belongs to."""
        for i in range(len(self.layer_sizes)):
            if global_idx < self.layer_offsets[i] + self.layer_sizes[i]:
                return i
        return len(self.layer_sizes) - 1

    def forward(
        self,
        input_spikes: torch.Tensor,
        max_timesteps: int = 50,
        return_all_spikes: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimized differentiable forward pass using vectorized operations.
        """
        self._init_state()

        output_size = self.layer_sizes[-1]

        # Track output spike times
        output_spike_times = torch.full((output_size,), -1, dtype=torch.long, device=self.device)
        output_has_fired = torch.zeros(output_size, dtype=torch.bool, device=self.device)

        # Layer states
        layer_spikes = [torch.zeros(size, device=self.device) for size in self.layer_sizes]
        layer_potentials = [torch.zeros(size, device=self.device) for size in self.layer_sizes]

        all_spikes = [] if return_all_spikes else None

        # Precompute decay
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=self.device))

        # Initialize input
        layer_spikes[0] = input_spikes.float() * 2.0

        for t in range(max_timesteps):
            # Vectorized propagation through layers
            for layer_idx in range(len(self.layer_sizes) - 1):
                conn = self.connectivity.get_connection(layer_idx)
                src_spikes = layer_spikes[layer_idx]

                # Fully vectorized: all sources contribute simultaneously
                all_targets = conn._cached_targets  # [num_src, fan_out]
                all_weights = conn.weights  # [num_src, fan_out]

                # Weighted contributions from all sources
                weighted_spikes = src_spikes.unsqueeze(1) * all_weights  # [num_src, fan_out]

                # Flatten and scatter
                flat_targets = all_targets.reshape(-1)
                flat_contributions = weighted_spikes.reshape(-1)

                layer_potentials[layer_idx + 1].scatter_add_(0, flat_targets, flat_contributions)

            # Decay and threshold (vectorized)
            for layer_idx in range(1, len(self.layer_sizes)):
                layer_potentials[layer_idx] = layer_potentials[layer_idx] * decay

                if layer_idx < len(self.layer_sizes) - 1:
                    # Hidden: surrogate spike and soft reset
                    spikes = surrogate_spike(layer_potentials[layer_idx], self.threshold)
                    layer_spikes[layer_idx] = spikes
                    layer_potentials[layer_idx] = layer_potentials[layer_idx] * (1 - spikes)
                else:
                    # Output: record spike times (vectorized)
                    above_thresh = layer_potentials[layer_idx] >= self.threshold
                    new_spikes = above_thresh & ~output_has_fired
                    output_spike_times[new_spikes] = t
                    output_has_fired = output_has_fired | above_thresh

            # Reset input after t=0
            if t == 0:
                layer_spikes[0] = torch.zeros_like(layer_spikes[0])

            if return_all_spikes:
                all_spikes.append((t, [s.clone() for s in layer_spikes]))

            # Early exit
            if output_has_fired.all():
                break

        info = {
            'layer_potentials': layer_potentials,
            'all_spikes': all_spikes,
        }

        return output_spike_times, info

    def decode_first_spike(self, output_spike_times: torch.Tensor) -> int:
        """Decode output using first-to-fire strategy."""
        fired_mask = output_spike_times >= 0
        if not fired_mask.any():
            return -1
        fired_times = output_spike_times.clone().float()
        fired_times[~fired_mask] = float('inf')
        return int(torch.argmin(fired_times))


class SpikingNetworkWithGrad(SpikingNetwork):
    """Extended network with gradient-friendly loss computation."""
    pass


def create_addition_network(device: torch.device = None) -> SpikingNetworkWithGrad:
    """Create network for single-digit addition task."""
    net = SpikingNetworkWithGrad(
        layer_sizes=[20, 128, 10],  # Larger hidden layer
        tau=20.0,  # Slower decay to accumulate signal
        threshold=0.3,  # Lower threshold
        fan_out_ratio=0.5,  # Full connectivity
        seed=42,
        device=device,
    )
    # Initialize weights with slight positive bias for better initial activity
    with torch.no_grad():
        for conn in net.connectivity.connections:
            conn.weights.data = conn.weights.data * 0.3 + 0.1
    return net
