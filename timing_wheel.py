"""
Timing Wheel Scheduler for Spiking Neural Networks.

Provides O(1) event scheduling and retrieval using a circular buffer of buckets.
Each bucket contains neurons scheduled to fire at that timestep.
"""

import torch
from typing import List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class SpikeEvent:
    """Represents a spike event for backprop tracking."""
    time: int
    src_neuron: int
    targets: torch.Tensor  # Target neuron indices
    weight_indices: torch.Tensor  # Indices into weight matrix for backprop
    weights: torch.Tensor  # The actual weight values (for gradient computation)


class TimingWheel:
    """
    O(1) timing wheel scheduler for spike events.

    Uses a circular buffer where each slot represents one timestep.
    Neurons scheduled to fire are stored in the corresponding bucket.
    """

    def __init__(self, num_slots: int = 256, device: torch.device = None):
        """
        Args:
            num_slots: Number of slots in the circular buffer (should be power of 2)
            device: PyTorch device for tensor operations
        """
        self.num_slots = num_slots
        self.device = device or torch.device('cpu')
        self.current_time = 0

        # Each bucket is a set of neuron indices scheduled to fire at that time
        self.buckets: List[Set[int]] = [set() for _ in range(num_slots)]

        # For batch operations, we also maintain tensor-based scheduling
        self._pending_spikes: Optional[torch.Tensor] = None

    def _slot_index(self, time: int) -> int:
        """Get bucket index for a given time (O(1) with modulo)."""
        return time % self.num_slots

    def schedule(self, neuron_id: int, fire_time: int) -> None:
        """
        Schedule a neuron to fire at a specific time. O(1) operation.

        Args:
            neuron_id: Index of the neuron to schedule
            fire_time: Absolute timestep when the neuron should fire
        """
        if fire_time < self.current_time:
            return  # Don't schedule in the past

        slot = self._slot_index(fire_time)
        self.buckets[slot].add(neuron_id)

    def schedule_batch(self, neuron_ids: torch.Tensor, fire_time: int) -> None:
        """
        Schedule multiple neurons to fire at the same time. O(k) for k neurons.

        Args:
            neuron_ids: Tensor of neuron indices to schedule
            fire_time: Absolute timestep when neurons should fire
        """
        if fire_time < self.current_time:
            return

        slot = self._slot_index(fire_time)
        for nid in neuron_ids.tolist():
            self.buckets[slot].add(nid)

    def get_current_spikes(self) -> torch.Tensor:
        """
        Get all neurons scheduled to fire at current timestep. O(k) for k neurons.

        Returns:
            Tensor of neuron indices firing at current time
        """
        slot = self._slot_index(self.current_time)
        spikes = list(self.buckets[slot])

        if len(spikes) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        return torch.tensor(spikes, dtype=torch.long, device=self.device)

    def advance(self) -> torch.Tensor:
        """
        Advance time by one step, returning and clearing current bucket.

        Returns:
            Tensor of neuron indices that fired at the (previous) current time
        """
        spikes = self.get_current_spikes()

        # Clear the current bucket
        slot = self._slot_index(self.current_time)
        self.buckets[slot].clear()

        # Advance time
        self.current_time += 1

        return spikes

    def reset(self) -> None:
        """Reset the timing wheel to initial state."""
        self.current_time = 0
        for bucket in self.buckets:
            bucket.clear()

    def peek_next_spike_time(self) -> Optional[int]:
        """
        Look ahead to find the next time with scheduled spikes.
        Useful for sparse networks where many timesteps have no activity.

        Returns:
            Next timestep with spikes, or None if no spikes scheduled
        """
        for offset in range(self.num_slots):
            slot = self._slot_index(self.current_time + offset)
            if self.buckets[slot]:
                return self.current_time + offset
        return None

    def skip_to_next_spike(self) -> Optional[torch.Tensor]:
        """
        Skip to the next timestep with spikes and return them.

        Returns:
            Tuple of (new_time, spiking_neurons) or None if no spikes
        """
        next_time = self.peek_next_spike_time()
        if next_time is None:
            return None

        # Advance to that time
        self.current_time = next_time
        return self.advance()


class SpikeGraph:
    """
    Records spike events during forward pass for backpropagation.

    Maintains a list of spike events that can be traversed in reverse
    during the backward pass.
    """

    def __init__(self):
        self.events: List[SpikeEvent] = []
        self.clear()

    def record(self, time: int, src_neuron: int, targets: torch.Tensor,
               weight_indices: torch.Tensor, weights: torch.Tensor) -> None:
        """
        Record a spike event for later backpropagation.

        Args:
            time: Timestep of the spike
            src_neuron: Index of the source neuron
            targets: Target neuron indices
            weight_indices: Indices into weight matrix
            weights: Weight values (must have grad enabled)
        """
        event = SpikeEvent(
            time=time,
            src_neuron=src_neuron,
            targets=targets,
            weight_indices=weight_indices,
            weights=weights
        )
        self.events.append(event)

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events = []

    def get_events_reversed(self) -> List[SpikeEvent]:
        """Get events in reverse order for backpropagation."""
        return list(reversed(self.events))

    def __len__(self) -> int:
        return len(self.events)
