"""
Two-digit addition with carry for Spiking Neural Network.

Task: Add two 2-digit numbers (00-99 + 00-99 = 000-198)

Input encoding (40 neurons):
- Neurons 0-9: First number's tens digit (one-hot)
- Neurons 10-19: First number's ones digit (one-hot)
- Neurons 20-29: Second number's tens digit (one-hot)
- Neurons 30-39: Second number's ones digit (one-hot)

Output encoding (22 neurons):
- Neurons 0-1: Hundreds digit (0 or 1)
- Neurons 2-11: Tens digit (0-9)
- Neurons 12-21: Ones digit (0-9)

Decoding: First-to-fire within each digit group.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random

from spiking_network import SpikingNetwork, SpikingNetworkWithGrad, surrogate_spike
from timing_wheel import TimingWheel, SpikeGraph
from procedural_connectivity import LayeredConnectivity


@dataclass
class TwoDigitExample:
    """A two-digit addition example."""
    num1: int  # 0-99
    num2: int  # 0-99
    result: int  # 0-198
    input_spikes: torch.Tensor  # [40] boolean
    target_hundreds: int  # 0 or 1
    target_tens: int  # 0-9
    target_ones: int  # 0-9


def generate_two_digit_dataset(
    num_examples: int = 500,
    device: torch.device = None
) -> List[TwoDigitExample]:
    """Generate random two-digit addition examples."""
    device = device or torch.device('cpu')
    examples = []
    seen = set()

    while len(examples) < num_examples:
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)

        if (num1, num2) in seen:
            continue
        seen.add((num1, num2))

        result = num1 + num2

        # Parse digits
        d1_tens, d1_ones = num1 // 10, num1 % 10
        d2_tens, d2_ones = num2 // 10, num2 % 10
        r_hundreds = result // 100
        r_tens = (result // 10) % 10
        r_ones = result % 10

        # Create input spikes (one-hot encoding)
        input_spikes = torch.zeros(40, dtype=torch.bool, device=device)
        input_spikes[d1_tens] = True         # Tens of first number
        input_spikes[10 + d1_ones] = True    # Ones of first number
        input_spikes[20 + d2_tens] = True    # Tens of second number
        input_spikes[30 + d2_ones] = True    # Ones of second number

        examples.append(TwoDigitExample(
            num1=num1,
            num2=num2,
            result=result,
            input_spikes=input_spikes,
            target_hundreds=r_hundreds,
            target_tens=r_tens,
            target_ones=r_ones
        ))

    return examples


class TwoDigitAdditionNetwork(nn.Module):
    """
    SNN for two-digit addition with carry.

    Architecture:
    - Input: 40 neurons (4 one-hot digits)
    - Hidden: 256 neurons
    - Output: 22 neurons (hundreds[2] + tens[10] + ones[10])
    """

    def __init__(
        self,
        hidden_size: int = 512,
        tau: float = 20.0,
        threshold: float = 0.3,
        device: torch.device = None,
    ):
        super().__init__()

        self.input_size = 40
        self.hidden_size = hidden_size
        self.output_size = 22  # 2 + 10 + 10
        self.tau = tau
        self.threshold = threshold
        self.device = device or torch.device('cpu')

        layer_sizes = [self.input_size, hidden_size, self.output_size]
        self.layer_sizes = layer_sizes
        self.total_neurons = sum(layer_sizes)

        # Layer offsets
        self.layer_offsets = [0]
        for size in layer_sizes[:-1]:
            self.layer_offsets.append(self.layer_offsets[-1] + size)

        # Connectivity
        self.connectivity = LayeredConnectivity(
            layer_sizes=layer_sizes,
            fan_out_ratio=0.5,  # Full connectivity
            seed=42,
            device=device
        )

        # Initialize weights
        with torch.no_grad():
            for conn in self.connectivity.connections:
                # Xavier-like initialization with positive bias
                fan_in = conn.num_src
                std = 1.0 / (fan_in ** 0.5)
                conn.weights.data = torch.randn_like(conn.weights) * std + 0.05

        self.timing_wheel = TimingWheel(num_slots=256, device=device)
        self.spike_graph = SpikeGraph()

    def forward(
        self,
        input_spikes: torch.Tensor,
        max_timesteps: int = 50,
    ) -> Tuple[torch.Tensor, Dict]:
        """Optimized forward pass with vectorized operations."""

        # Layer potentials (differentiable)
        layer_potentials = [torch.zeros(size, device=self.device) for size in self.layer_sizes]

        # Current spikes per layer
        layer_spikes = [torch.zeros(size, device=self.device) for size in self.layer_sizes]

        # Output spike times
        output_spike_times = torch.full((self.output_size,), -1, dtype=torch.long, device=self.device)
        output_has_fired = torch.zeros(self.output_size, dtype=torch.bool, device=self.device)

        # Precompute decay factor
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=self.device))

        # Initialize input
        layer_spikes[0] = input_spikes.float() * 2.0

        for t in range(max_timesteps):
            # Propagate through each layer transition (vectorized)
            for layer_idx in range(len(self.layer_sizes) - 1):
                conn = self.connectivity.get_connection(layer_idx)
                src_spikes = layer_spikes[layer_idx]

                # Get all source neurons (use all, weighted by spike value)
                # targets: [num_src, fan_out], weights: [num_src, fan_out]
                all_targets = conn._cached_targets  # [num_src, fan_out]
                all_weights = conn.weights  # [num_src, fan_out]

                # Compute contributions: spike_val * weight for each connection
                # src_spikes: [num_src], all_weights: [num_src, fan_out]
                weighted_spikes = src_spikes.unsqueeze(1) * all_weights  # [num_src, fan_out]

                # Flatten and scatter add
                flat_targets = all_targets.reshape(-1)  # [num_src * fan_out]
                flat_contributions = weighted_spikes.reshape(-1)  # [num_src * fan_out]

                # Accumulate into destination layer
                layer_potentials[layer_idx + 1].scatter_add_(0, flat_targets, flat_contributions)

            # Apply decay and threshold (vectorized)
            for layer_idx in range(1, len(self.layer_sizes)):
                layer_potentials[layer_idx] = layer_potentials[layer_idx] * decay

                if layer_idx < len(self.layer_sizes) - 1:
                    # Hidden layer: generate spikes
                    spikes = surrogate_spike(layer_potentials[layer_idx], self.threshold)
                    layer_spikes[layer_idx] = spikes
                    # Soft reset
                    layer_potentials[layer_idx] = layer_potentials[layer_idx] * (1 - spikes)
                else:
                    # Output layer: record spike times (vectorized)
                    above_thresh = layer_potentials[layer_idx] >= self.threshold
                    new_spikes = above_thresh & ~output_has_fired
                    output_spike_times[new_spikes] = t
                    output_has_fired = output_has_fired | above_thresh

            # Reset input after first timestep
            if t == 0:
                layer_spikes[0] = torch.zeros_like(layer_spikes[0])

            # Early exit if all outputs have fired
            if output_has_fired.all():
                break

        info = {
            'layer_potentials': layer_potentials,
        }

        return output_spike_times, info

    def decode_output(self, output_spike_times: torch.Tensor, output_potentials: torch.Tensor) -> Tuple[int, int, int]:
        """
        Decode the three output digits using first-to-fire within each group.
        Falls back to max potential if no spike.
        """
        def decode_group(times, potentials, start, end):
            group_times = times[start:end]
            group_pots = potentials[start:end]

            fired_mask = group_times >= 0
            if fired_mask.any():
                # Use first to fire
                fired_times = group_times.clone().float()
                fired_times[~fired_mask] = float('inf')
                return int(torch.argmin(fired_times))
            else:
                # Use highest potential
                return int(torch.argmax(group_pots))

        hundreds = decode_group(output_spike_times, output_potentials, 0, 2)
        tens = decode_group(output_spike_times, output_potentials, 2, 12)
        ones = decode_group(output_spike_times, output_potentials, 12, 22)

        return hundreds, tens, ones

    def decode_result(self, output_spike_times: torch.Tensor, output_potentials: torch.Tensor) -> int:
        """Decode full numeric result."""
        h, t, o = self.decode_output(output_spike_times, output_potentials)
        return h * 100 + t * 10 + o


class MultiGroupLoss(nn.Module):
    """
    Loss for multi-group first-to-fire classification.
    Vectorized for speed.
    """

    def __init__(self, temperature: float = 1.0, margin: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        output_potentials: torch.Tensor,
        target_hundreds: int,
        target_tens: int,
        target_ones: int
    ) -> torch.Tensor:
        device = output_potentials.device

        # Split into groups
        hundreds_pots = output_potentials[0:2]
        tens_pots = output_potentials[2:12]
        ones_pots = output_potentials[12:22]

        # Targets as tensors
        t_h = torch.tensor([target_hundreds], device=device)
        t_t = torch.tensor([target_tens], device=device)
        t_o = torch.tensor([target_ones], device=device)

        # Cross-entropy for each group (vectorized)
        ce_h = torch.nn.functional.cross_entropy(hundreds_pots.unsqueeze(0) / self.temperature, t_h)
        ce_t = torch.nn.functional.cross_entropy(tens_pots.unsqueeze(0) / self.temperature, t_t)
        ce_o = torch.nn.functional.cross_entropy(ones_pots.unsqueeze(0) / self.temperature, t_o)

        return ce_h + ce_t + ce_o


def train_two_digit(
    num_epochs: int = 200,
    num_train: int = 500,
    num_test: int = 100,
    hidden_size: int = 256,
    learning_rate: float = 0.02,
    max_timesteps: int = 10,
    device: torch.device = None,
    verbose: bool = True
) -> Tuple[TwoDigitAdditionNetwork, Dict]:
    """Train the two-digit addition network."""

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate data
    print(f"Generating {num_train} training and {num_test} test examples...")
    train_examples = generate_two_digit_dataset(num_train, device)
    test_examples = generate_two_digit_dataset(num_test, device)

    # Create network
    network = TwoDigitAdditionNetwork(
        hidden_size=hidden_size,
        tau=20.0,
        threshold=0.25,
        device=device
    )

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {num_params}")

    # Training setup
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = MultiGroupLoss(temperature=1.0, margin=0.3)

    history = {'loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        random.shuffle(train_examples)

        total_loss = 0.0
        correct = 0

        for example in train_examples:
            optimizer.zero_grad()

            output_times, info = network.forward(example.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['layer_potentials'][-1]

            loss = loss_fn(output_pots, example.target_hundreds, example.target_tens, example.target_ones)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            # Check accuracy
            pred = network.decode_result(output_times, output_pots)
            if pred == example.result:
                correct += 1

        avg_loss = total_loss / len(train_examples)
        train_acc = correct / len(train_examples)

        # Test accuracy
        test_correct = 0
        with torch.no_grad():
            for example in test_examples:
                output_times, info = network.forward(example.input_spikes, max_timesteps=max_timesteps)
                output_pots = info['layer_potentials'][-1]
                pred = network.decode_result(output_times, output_pots)
                if pred == example.result:
                    test_correct += 1

        test_acc = test_correct / len(test_examples)

        history['loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if verbose and (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Loss = {avg_loss:.4f}, "
                  f"Train = {train_acc:.1%}, "
                  f"Test = {test_acc:.1%}")

    print(f"\nTraining complete!")
    print(f"Final train accuracy: {history['train_acc'][-1]:.1%}")
    print(f"Final test accuracy: {history['test_acc'][-1]:.1%}")

    # Show sample predictions
    print("\nSample test predictions:")
    for example in test_examples[:15]:
        with torch.no_grad():
            output_times, info = network.forward(example.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['layer_potentials'][-1]
            pred = network.decode_result(output_times, output_pots)
            status = "✓" if pred == example.result else "✗"
            print(f"  {example.num1:2d} + {example.num2:2d} = {example.result:3d}, predicted: {pred:3d} {status}")

    return network, history


if __name__ == "__main__":
    network, history = train_two_digit(
        num_epochs=300,
        num_train=5000,
        num_test=1000,
        hidden_size=512,
        learning_rate=0.01
    )
