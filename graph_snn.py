"""
Graph-based Spiking Neural Network.

Unlike layered SNNs, this allows arbitrary connectivity between any neurons.
Spikes propagate through the graph based on timing, not layer order.

Architecture:
- Flat array of N neurons (no layer distinction internally)
- Each neuron has fan_out outgoing connections to any other neuron
- Timing wheel schedules and processes spikes by time
- Event-driven: only active neurons are processed
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


class SurrogateSpike(torch.autograd.Function):
    """Spike with surrogate gradient for backprop."""

    scale = 25.0

    @staticmethod
    def forward(ctx, v: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        v, = ctx.saved_tensors
        x = SurrogateSpike.scale * (v - ctx.threshold)
        sig = torch.sigmoid(x)
        grad = SurrogateSpike.scale * sig * (1 - sig)
        return grad_output * grad, None


def surrogate_spike(v: torch.Tensor, threshold: float) -> torch.Tensor:
    return SurrogateSpike.apply(v, threshold)


class GraphSNN(nn.Module):
    """
    Graph-based Spiking Neural Network.

    All neurons are in a flat array. Connections can go from any neuron
    to any other neuron (except input neurons don't receive connections).
    """

    def __init__(
        self,
        num_input: int,
        num_hidden: int,
        num_output: int,
        fan_out: int = 32,
        tau: float = 20.0,
        threshold: float = 0.3,
        seed: int = 42,
        device: torch.device = None,
    ):
        super().__init__()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.num_neurons = num_input + num_hidden + num_output
        self.fan_out = min(fan_out, self.num_neurons - 1)
        self.tau = tau
        self.threshold = threshold
        self.seed = seed
        self.device = device or torch.device('cpu')

        # Neuron ranges
        self.input_slice = slice(0, num_input)
        self.hidden_slice = slice(num_input, num_input + num_hidden)
        self.output_slice = slice(num_input + num_hidden, self.num_neurons)

        # Learnable weights: [num_neurons, fan_out]
        self.weights = nn.Parameter(
            torch.randn(self.num_neurons, self.fan_out, device=device) * 0.3
        )

        # Generate and store random connection targets (not procedural)
        self._generate_random_targets()

    def _generate_random_targets(self):
        """
        Generate random targets with structured connectivity.
        Stored in memory (not procedurally regenerated).

        - Input neurons -> hidden + output
        - Hidden neurons -> hidden + output (allows recurrence)
        - Output neurons -> hidden (feedback)
        """
        torch.manual_seed(self.seed)

        hidden_start = self.num_input
        hidden_end = self.num_input + self.num_hidden
        output_end = self.num_neurons

        targets = torch.zeros(self.num_neurons, self.fan_out, dtype=torch.long, device=self.device)

        for src in range(self.num_neurons):
            # Determine valid target range based on source type
            if src < self.num_input:
                # Input -> HIDDEN ONLY (forces processing before output)
                valid = torch.arange(hidden_start, hidden_end, device=self.device)
            elif src < hidden_end:
                # Hidden -> hidden + output (can be recurrent)
                valid = torch.arange(hidden_start, output_end, device=self.device)
                # Remove self-connection
                valid = valid[valid != src]
            else:
                # Output -> hidden (feedback)
                valid = torch.arange(hidden_start, hidden_end, device=self.device)

            # Randomly select fan_out unique targets
            n_targets = min(self.fan_out, len(valid))
            perm = torch.randperm(len(valid), device=self.device)[:n_targets]
            targets[src, :n_targets] = valid[perm]

            # If not enough valid targets, repeat some
            if n_targets < self.fan_out:
                for i in range(n_targets, self.fan_out):
                    targets[src, i] = valid[i % len(valid)]

        self.register_buffer('targets', targets)

    def forward(
        self,
        input_spikes: torch.Tensor,
        max_timesteps: int = 20,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Event-driven forward pass.

        Args:
            input_spikes: [num_input] boolean tensor
            max_timesteps: Maximum simulation time

        Returns:
            output_spike_times: [num_output] first spike time per output (-1 if none)
            info: Dictionary with layer_potentials for loss computation
        """
        # State
        potentials = torch.zeros(self.num_neurons, device=self.device)
        has_fired = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)

        # Output tracking
        output_spike_times = torch.full((self.num_output,), -1, dtype=torch.long, device=self.device)

        # Current spikes (which neurons are spiking this timestep)
        current_spikes = torch.zeros(self.num_neurons, device=self.device)

        # Initialize: input neurons spike at t=0
        current_spikes[self.input_slice] = input_spikes.float() * 2.0
        has_fired[self.input_slice] = input_spikes

        # Precompute decay
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=self.device))

        for t in range(max_timesteps):
            # 1. Decay potentials from previous timestep (not on t=0)
            if t > 0:
                potentials = potentials * decay

            # 2. Propagate current spikes through graph
            weighted_spikes = current_spikes.unsqueeze(1) * self.weights
            flat_targets = self.targets.reshape(-1)
            flat_contributions = weighted_spikes.reshape(-1)
            potentials.scatter_add_(0, flat_targets, flat_contributions)

            # 3. Clear input spikes after first propagation
            if t == 0:
                current_spikes = current_spikes.clone()
                current_spikes[self.input_slice] = 0

            # 4. Determine which neurons spike (above threshold, haven't fired yet)
            can_spike = ~has_fired
            spikes = surrogate_spike(potentials, self.threshold) * can_spike.float()

            # 5. Record which neurons fired
            new_fired = spikes > 0.5
            has_fired = has_fired | new_fired

            # 6. Record output spike times
            output_fired = new_fired[self.output_slice]
            for i in range(self.num_output):
                if output_fired[i] and output_spike_times[i] < 0:
                    output_spike_times[i] = t

            # 7. Soft reset spiked neurons (but NOT output neurons - let them accumulate for loss)
            reset_mask = spikes.clone()
            reset_mask[self.output_slice] = 0  # Don't reset output neurons
            potentials = potentials * (1 - reset_mask)

            # 8. Update current spikes for next iteration
            current_spikes = spikes

        # Output potentials now accumulate (not reset) so use them directly for loss
        output_potentials = potentials[self.output_slice]
        info = {
            'layer_potentials': [None, None, output_potentials],  # Compatible format
            'potentials': potentials,
            'has_fired': has_fired,
        }

        return output_spike_times, info

    def decode_first_spike(self, output_spike_times: torch.Tensor, output_potentials: torch.Tensor = None) -> int:
        """Decode using first-to-fire, with fallback to max potential."""
        fired_mask = output_spike_times >= 0
        if fired_mask.any():
            times = output_spike_times.float()
            times[~fired_mask] = float('inf')
            return int(torch.argmin(times))
        elif output_potentials is not None:
            # Fallback: use highest potential if no spikes
            return int(torch.argmax(output_potentials))
        else:
            return -1


class GraphAdditionLoss(nn.Module):
    """Loss function for graph-based SNN."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, output_potentials: torch.Tensor, target: int) -> torch.Tensor:
        logits = output_potentials / self.temperature
        target_t = torch.tensor([target], device=output_potentials.device)
        return torch.nn.functional.cross_entropy(logits.unsqueeze(0), target_t)


def create_graph_addition_network(device: torch.device = None) -> GraphSNN:
    """Create graph SNN for single-digit addition."""
    return GraphSNN(
        num_input=20,    # Two one-hot digits
        num_hidden=128,  # Hidden neurons
        num_output=10,   # Output digits 0-9
        fan_out=64,      # Connections per neuron (higher works better)
        tau=20.0,        # Slower decay for accumulation
        threshold=0.3,   # Threshold
        seed=42,
        device=device,
    )


def train_graph_addition(
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    device: torch.device = None,
):
    """Train graph SNN on single-digit addition."""
    import random
    import torch.optim as optim
    from training import generate_addition_dataset

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create network
    network = create_graph_addition_network(device)
    print(f"Graph SNN: {network.num_neurons} neurons, {network.fan_out} fan-out")
    print(f"Parameters: {sum(p.numel() for p in network.parameters())}")

    # Data
    examples = generate_addition_dataset(device)
    print(f"Training examples: {len(examples)}")

    # Training
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_fn = GraphAdditionLoss(temperature=1.0)

    for epoch in range(num_epochs):
        random.shuffle(examples)
        total_loss = 0
        correct = 0

        for ex in examples:
            optimizer.zero_grad()

            output_times, info = network.forward(ex.input_spikes, max_timesteps=20)
            output_pots = info['layer_potentials'][-1]

            loss = loss_fn(output_pots, ex.result)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            pred = network.decode_first_spike(output_times, output_pots)
            if pred == ex.result:
                correct += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            acc = correct / len(examples)
            print(f"Epoch {epoch+1}: Loss={total_loss/len(examples):.4f}, Acc={acc:.1%}")

    # Final evaluation
    print("\nSample predictions:")
    for ex in examples[:10]:
        with torch.no_grad():
            output_times, info = network.forward(ex.input_spikes)
            output_pots = info['layer_potentials'][-1]
            pred = network.decode_first_spike(output_times, output_pots)
            mark = "OK" if pred == ex.result else "X"
            print(f"  {ex.a} + {ex.b} = {ex.result}, pred={pred} [{mark}]")

    return network


# ============================================================================
# Two-Digit Addition Support
# ============================================================================

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
    import random
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


class GraphTwoDigitLoss(nn.Module):
    """
    Loss for two-digit addition with three output groups.
    - Hundreds: neurons 0-1
    - Tens: neurons 2-11
    - Ones: neurons 12-21
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

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

        # Cross-entropy for each group
        ce_h = torch.nn.functional.cross_entropy(hundreds_pots.unsqueeze(0) / self.temperature, t_h)
        ce_t = torch.nn.functional.cross_entropy(tens_pots.unsqueeze(0) / self.temperature, t_t)
        ce_o = torch.nn.functional.cross_entropy(ones_pots.unsqueeze(0) / self.temperature, t_o)

        return ce_h + ce_t + ce_o


def decode_two_digit_output(
    output_spike_times: torch.Tensor,
    output_potentials: torch.Tensor
) -> Tuple[int, int, int]:
    """
    Decode three output digit groups using first-to-fire within each group.
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


def create_graph_two_digit_network(device: torch.device = None) -> GraphSNN:
    """Create graph SNN for two-digit addition."""
    return GraphSNN(
        num_input=40,    # Four one-hot digits (tens1, ones1, tens2, ones2)
        num_hidden=256,  # More hidden for harder task
        num_output=22,   # 2 (hundreds) + 10 (tens) + 10 (ones)
        fan_out=32,      # Connections per neuron
        tau=20.0,
        threshold=0.3,
        seed=42,
        device=device,
    )


def train_graph_two_digit(
    num_epochs: int = 300,
    num_train: int = 2000,
    num_test: int = 500,
    learning_rate: float = 0.01,
    max_timesteps: int = 25,
    device: torch.device = None,
):
    """Train graph SNN on two-digit addition."""
    import random
    import torch.optim as optim

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create network
    network = create_graph_two_digit_network(device)
    print(f"Graph SNN: {network.num_neurons} neurons, {network.fan_out} fan-out")
    print(f"Parameters: {sum(p.numel() for p in network.parameters())}")

    # Data
    print(f"Generating {num_train} training and {num_test} test examples...")
    train_examples = generate_two_digit_dataset(num_train, device)
    test_examples = generate_two_digit_dataset(num_test, device)

    # Training
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_fn = GraphTwoDigitLoss(temperature=1.0)

    best_test_acc = 0.0

    for epoch in range(num_epochs):
        random.shuffle(train_examples)
        total_loss = 0
        correct = 0

        for ex in train_examples:
            optimizer.zero_grad()

            output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['layer_potentials'][-1]

            loss = loss_fn(output_pots, ex.target_hundreds, ex.target_tens, ex.target_ones)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Decode prediction
            h, t, o = decode_two_digit_output(output_times, output_pots)
            pred = h * 100 + t * 10 + o
            if pred == ex.result:
                correct += 1

        scheduler.step()
        train_acc = correct / len(train_examples)

        # Test accuracy
        if (epoch + 1) % 1 == 0:
            test_correct = 0
            with torch.no_grad():
                for ex in test_examples:
                    output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
                    output_pots = info['layer_potentials'][-1]
                    h, t, o = decode_two_digit_output(output_times, output_pots)
                    pred = h * 100 + t * 10 + o
                    if pred == ex.result:
                        test_correct += 1

            test_acc = test_correct / len(test_examples)
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_examples):.4f}, "
                  f"Train={train_acc:.1%}, Test={test_acc:.1%}")

    print(f"\nBest test accuracy: {best_test_acc:.1%}")

    # Final evaluation
    print("\nSample test predictions:")
    for ex in test_examples[:15]:
        with torch.no_grad():
            output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['layer_potentials'][-1]
            h, t, o = decode_two_digit_output(output_times, output_pots)
            pred = h * 100 + t * 10 + o
            mark = "OK" if pred == ex.result else "X"
            print(f"  {ex.num1:2d} + {ex.num2:2d} = {ex.result:3d}, pred={pred:3d} [{mark}]")

    return network


if __name__ == "__main__":
    # Run two-digit addition (more interesting task)
    train_graph_two_digit(
        num_epochs=300,
        num_train=2000,
        num_test=500,
        learning_rate=0.015
    )
