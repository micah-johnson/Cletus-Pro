"""
Lazy Membrane Graph-based Spiking Neural Network.

True event-driven computation where neurons are only updated when spikes arrive.
Key innovations:
1. Per-neuron state tracking (membrane, last_update time)
2. Retroactive decay: compute decay only when spike arrives
3. Event-driven loop: no fixed timestep iteration
4. Fully differentiable for backprop through surrogate gradients

Gradient Flow:
- When spike arrives at neuron j at time t:
  1. decay_factor = exp(-(t - last_update[j]) / tau)  [differentiable]
  2. membrane[j] = membrane[j] * decay_factor + weight  [differentiable]
  3. spike = surrogate(membrane[j] - threshold)  [differentiable via surrogate]
- The computation graph follows spike causality
- Gradients flow backwards through the spike chain
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass
import heapq


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


class EventQueue:
    """
    Priority queue for spike events.

    Uses a min-heap to process events in time order.
    Supports batched retrieval of all events at the same timestep.
    """

    def __init__(self):
        self.heap: List[Tuple[int, int]] = []  # (time, neuron_id)
        self.scheduled: Set[Tuple[int, int]] = set()  # avoid duplicates

    def schedule(self, neuron_id: int, time: int):
        """Schedule a neuron to spike at given time."""
        key = (time, neuron_id)
        if key not in self.scheduled:
            heapq.heappush(self.heap, key)
            self.scheduled.add(key)

    def pop_next_batch(self) -> Tuple[int, List[int]]:
        """
        Pop all events at the earliest timestep.
        Returns (time, list of neuron_ids).
        """
        if not self.heap:
            return -1, []

        t, first_neuron = heapq.heappop(self.heap)
        self.scheduled.discard((t, first_neuron))
        neurons = [first_neuron]

        # Get all other neurons at same timestep
        while self.heap and self.heap[0][0] == t:
            _, neuron = heapq.heappop(self.heap)
            self.scheduled.discard((t, neuron))
            neurons.append(neuron)

        return t, neurons

    def has_events(self) -> bool:
        return len(self.heap) > 0

    def peek_time(self) -> int:
        """Peek at the next event time without popping."""
        return self.heap[0][0] if self.heap else -1

    def clear(self):
        self.heap.clear()
        self.scheduled.clear()


class LazyGraphSNN(nn.Module):
    """
    Graph-based SNN with lazy membrane evaluation.

    Neurons are only updated when spikes arrive, not every timestep.
    This is more biologically plausible and computationally efficient
    for sparse activity.

    State tracked per neuron:
    - membrane: Current membrane potential (last known value)
    - last_update: Timestep when membrane was last computed
    - has_fired: Whether neuron has fired (for refractory period)
    """

    def __init__(
        self,
        num_input: int,
        num_hidden: int,
        num_output: int,
        fan_out: int = 32,
        tau: float = 20.0,
        threshold: float = 0.3,
        spike_delay: int = 1,  # Delay before spike arrives at target
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
        self.spike_delay = spike_delay
        self.seed = seed
        self.device = device or torch.device('cpu')

        # Neuron ranges
        self.input_start = 0
        self.input_end = num_input
        self.hidden_start = num_input
        self.hidden_end = num_input + num_hidden
        self.output_start = num_input + num_hidden
        self.output_end = self.num_neurons

        # Learnable weights: [num_neurons, fan_out]
        self.weights = nn.Parameter(
            torch.randn(self.num_neurons, self.fan_out, device=device) * 0.3
        )

        # Generate and store random connection targets
        self._generate_random_targets()

        # Precompute decay constant (for use in lazy decay)
        # decay_per_step = exp(-1/tau)
        self.register_buffer(
            'decay_base',
            torch.exp(torch.tensor(-1.0 / self.tau, device=device))
        )

    def _generate_random_targets(self):
        """Generate random targets with structured connectivity."""
        torch.manual_seed(self.seed)

        targets = torch.zeros(self.num_neurons, self.fan_out, dtype=torch.long, device=self.device)

        for src in range(self.num_neurons):
            if src < self.input_end:
                # Input -> hidden only (forces two-hop processing)
                valid = torch.arange(self.hidden_start, self.hidden_end, device=self.device)
            elif src < self.hidden_end:
                # Hidden -> hidden + output (can be recurrent, no self)
                valid = torch.arange(self.hidden_start, self.output_end, device=self.device)
                valid = valid[valid != src]
            else:
                # Output -> hidden (feedback)
                valid = torch.arange(self.hidden_start, self.hidden_end, device=self.device)

            n_targets = min(self.fan_out, len(valid))
            perm = torch.randperm(len(valid), device=self.device)[:n_targets]
            targets[src, :n_targets] = valid[perm]

            if n_targets < self.fan_out:
                for i in range(n_targets, self.fan_out):
                    targets[src, i] = valid[i % len(valid)]

        self.register_buffer('targets', targets)

    def _lazy_decay(
        self,
        membrane: torch.Tensor,
        last_update: torch.Tensor,
        current_time: int,
        neuron_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply retroactive decay to specified neurons.

        This is the key lazy computation:
        membrane[i] = membrane[i] * exp(-(current_time - last_update[i]) / tau)

        Args:
            membrane: Current membrane potentials [N]
            last_update: Last update time per neuron [N]
            current_time: Current simulation time
            neuron_indices: Which neurons to update [K]

        Returns:
            Updated membrane tensor (maintains gradient)
        """
        if len(neuron_indices) == 0:
            return membrane

        # Compute time since last update for each target neuron
        dt = current_time - last_update[neuron_indices]  # [K]

        # Compute decay factor: exp(-dt / tau) = decay_base^dt
        # Using pow maintains gradient flow
        decay_factors = torch.pow(self.decay_base, dt.float())  # [K]

        # Apply decay to selected neurons
        # We need to do this in a way that maintains gradients
        membrane = membrane.clone()
        membrane[neuron_indices] = membrane[neuron_indices] * decay_factors

        return membrane

    def forward(
        self,
        input_spikes: torch.Tensor,
        max_timesteps: int = 50,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Hybrid lazy-timestep forward pass.

        Uses timestep iteration (for easier gradient flow) but with lazy
        decay computation (only update neurons when they receive input).

        Gradient flow:
        - Input spikes -> input weights -> hidden potentials
        - Hidden potentials -> surrogate spike -> hidden weights -> output potentials
        - Output potentials -> loss

        Args:
            input_spikes: [num_input] boolean tensor
            max_timesteps: Maximum simulation time

        Returns:
            output_spike_times: [num_output] first spike time per output (-1 if none)
            info: Dictionary with output_potentials for loss computation
        """
        # All potentials in one tensor for proper gradient flow
        potentials = torch.zeros(self.num_neurons, device=self.device)
        last_update = torch.zeros(self.num_neurons, dtype=torch.long, device=self.device)
        has_fired = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)

        output_spike_times = torch.full(
            (self.num_output,), -1, dtype=torch.long, device=self.device
        )

        # Current spikes (soft, differentiable)
        current_spikes = torch.zeros(self.num_neurons, device=self.device)

        # Initialize input spikes
        current_spikes[:self.input_end] = input_spikes.float() * 2.0
        has_fired[:self.input_end] = input_spikes

        # Track which neurons have pending input (for lazy updates)
        pending_input = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)

        for t in range(max_timesteps):
            # Find neurons that are spiking
            spiking_mask = current_spikes > 0.1

            if not spiking_mask.any():
                continue  # Skip timestep but don't break - allow later activity

            spiking_indices = torch.where(spiking_mask)[0]

            # Get all targets and weights for spiking neurons
            batch_targets = self.targets[spiking_indices]  # [B, fan_out]
            batch_weights = self.weights[spiking_indices]  # [B, fan_out]
            spike_vals = current_spikes[spiking_indices]   # [B]

            # Compute weighted contributions
            contributions = spike_vals.unsqueeze(1) * batch_weights  # [B, fan_out]

            flat_targets = batch_targets.reshape(-1)
            flat_contribs = contributions.reshape(-1)

            # Get unique targets
            unique_targets = torch.unique(flat_targets)

            # LAZY DECAY: Apply decay only to neurons receiving input
            # dt = t - last_update[target]
            dt = t - last_update[unique_targets]
            decay_factors = torch.pow(self.decay_base, dt.float())

            # Apply decay (creates new tensor, maintains gradient)
            potentials = potentials.clone()
            potentials[unique_targets] = potentials[unique_targets] * decay_factors
            last_update[unique_targets] = t

            # Accumulate contributions
            potentials.scatter_add_(0, flat_targets, flat_contribs)

            # Clear input spikes after first propagation
            if t == 0:
                current_spikes = current_spikes.clone()
                current_spikes[:self.input_end] = 0

            # Check thresholds with surrogate gradient
            can_spike = ~has_fired
            spike_probs = surrogate_spike(potentials, self.threshold) * can_spike.float()

            # Record new spikes
            new_fired = spike_probs > 0.5
            has_fired = has_fired | new_fired

            # Record output spike times
            output_fired = new_fired[self.output_start:self.output_end]
            for i in range(self.num_output):
                if output_fired[i] and output_spike_times[i] < 0:
                    output_spike_times[i] = t

            # Soft reset for hidden neurons only (outputs accumulate for loss)
            reset_mask = spike_probs.clone()
            reset_mask[self.output_start:] = 0  # Don't reset outputs
            potentials = potentials * (1 - reset_mask)

            # Update current spikes for next iteration
            current_spikes = spike_probs

        # Final lazy decay for all neurons to end time
        all_indices = torch.arange(self.num_neurons, device=self.device)
        final_dt = max_timesteps - last_update
        final_decay = torch.pow(self.decay_base, final_dt.float())
        potentials = potentials * final_decay

        output_potentials = potentials[self.output_start:self.output_end]

        info = {
            'layer_potentials': [None, None, output_potentials],
            'output_potentials': output_potentials,
            'has_fired': has_fired,
        }

        return output_spike_times, info

    def decode_first_spike(
        self,
        output_spike_times: torch.Tensor,
        output_potentials: torch.Tensor = None
    ) -> int:
        """Decode using first-to-fire, with fallback to max potential."""
        fired_mask = output_spike_times >= 0
        if fired_mask.any():
            times = output_spike_times.float()
            times[~fired_mask] = float('inf')
            return int(torch.argmin(times))
        elif output_potentials is not None:
            return int(torch.argmax(output_potentials))
        else:
            return -1


# ============================================================================
# Two-Digit Addition Support (same as graph_snn.py)
# ============================================================================

from dataclasses import dataclass

@dataclass
class TwoDigitExample:
    """A two-digit addition example."""
    num1: int
    num2: int
    result: int
    input_spikes: torch.Tensor
    target_hundreds: int
    target_tens: int
    target_ones: int


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
        d1_tens, d1_ones = num1 // 10, num1 % 10
        d2_tens, d2_ones = num2 // 10, num2 % 10
        r_hundreds = result // 100
        r_tens = (result // 10) % 10
        r_ones = result % 10

        input_spikes = torch.zeros(40, dtype=torch.bool, device=device)
        input_spikes[d1_tens] = True
        input_spikes[10 + d1_ones] = True
        input_spikes[20 + d2_tens] = True
        input_spikes[30 + d2_ones] = True

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


class LazyGraphTwoDigitLoss(nn.Module):
    """Loss for two-digit addition with three output groups."""

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

        hundreds_pots = output_potentials[0:2]
        tens_pots = output_potentials[2:12]
        ones_pots = output_potentials[12:22]

        t_h = torch.tensor([target_hundreds], device=device)
        t_t = torch.tensor([target_tens], device=device)
        t_o = torch.tensor([target_ones], device=device)

        ce_h = torch.nn.functional.cross_entropy(
            hundreds_pots.unsqueeze(0) / self.temperature, t_h
        )
        ce_t = torch.nn.functional.cross_entropy(
            tens_pots.unsqueeze(0) / self.temperature, t_t
        )
        ce_o = torch.nn.functional.cross_entropy(
            ones_pots.unsqueeze(0) / self.temperature, t_o
        )

        return ce_h + ce_t + ce_o


def decode_two_digit_output(
    output_spike_times: torch.Tensor,
    output_potentials: torch.Tensor
) -> Tuple[int, int, int]:
    """Decode three output digit groups."""
    def decode_group(times, potentials, start, end):
        group_times = times[start:end]
        group_pots = potentials[start:end]

        fired_mask = group_times >= 0
        if fired_mask.any():
            fired_times = group_times.clone().float()
            fired_times[~fired_mask] = float('inf')
            return int(torch.argmin(fired_times))
        else:
            return int(torch.argmax(group_pots))

    hundreds = decode_group(output_spike_times, output_potentials, 0, 2)
    tens = decode_group(output_spike_times, output_potentials, 2, 12)
    ones = decode_group(output_spike_times, output_potentials, 12, 22)

    return hundreds, tens, ones


def create_lazy_two_digit_network(device: torch.device = None) -> LazyGraphSNN:
    """Create lazy graph SNN for two-digit addition."""
    return LazyGraphSNN(
        num_input=40,
        num_hidden=256,
        num_output=22,
        fan_out=96,      # More connections
        tau=20.0,
        threshold=0.25,  # Lower threshold
        spike_delay=1,
        seed=42,
        device=device,
    )


def train_lazy_two_digit(
    num_epochs: int = 300,
    num_train: int = 2000,
    num_test: int = 500,
    learning_rate: float = 0.03,
    max_timesteps: int = 20,
    device: torch.device = None,
):
    """Train lazy graph SNN on two-digit addition."""
    import random
    import torch.optim as optim

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("LAZY MEMBRANE EVALUATION - Event-Driven SNN")
    print("=" * 60)

    network = create_lazy_two_digit_network(device)
    print(f"Neurons: {network.num_neurons} (input={network.num_input}, "
          f"hidden={network.num_hidden}, output={network.num_output})")
    print(f"Fan-out: {network.fan_out}, Tau: {network.tau}, Threshold: {network.threshold}")
    print(f"Parameters: {sum(p.numel() for p in network.parameters())}")

    print(f"\nGenerating {num_train} training and {num_test} test examples...")
    train_examples = generate_two_digit_dataset(num_train, device)
    test_examples = generate_two_digit_dataset(num_test, device)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_fn = LazyGraphTwoDigitLoss(temperature=1.0)

    best_test_acc = 0.0

    for epoch in range(num_epochs):
        random.shuffle(train_examples)
        total_loss = 0
        correct = 0

        for ex in train_examples:
            optimizer.zero_grad()

            output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['output_potentials']

            loss = loss_fn(output_pots, ex.target_hundreds, ex.target_tens, ex.target_ones)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            h, t, o = decode_two_digit_output(output_times, output_pots)
            pred = h * 100 + t * 10 + o
            if pred == ex.result:
                correct += 1

        scheduler.step()
        train_acc = correct / len(train_examples)

        if (epoch + 1) % 1 == 0:
            test_correct = 0
            with torch.no_grad():
                for ex in test_examples:
                    output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
                    output_pots = info['output_potentials']
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

    print("\nSample test predictions:")
    for ex in test_examples[:15]:
        with torch.no_grad():
            output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['output_potentials']
            h, t, o = decode_two_digit_output(output_times, output_pots)
            pred = h * 100 + t * 10 + o
            mark = "OK" if pred == ex.result else "X"
            print(f"  {ex.num1:2d} + {ex.num2:2d} = {ex.result:3d}, pred={pred:3d} [{mark}]")

    return network


# ============================================================================
# Single-Digit Addition (for quick testing)
# ============================================================================

@dataclass
class AdditionExample:
    """A single addition training example."""
    a: int
    b: int
    result: int
    input_spikes: torch.Tensor


def generate_addition_dataset(device: torch.device = None) -> List[AdditionExample]:
    """Generate all valid single-digit addition examples where a + b < 10."""
    device = device or torch.device('cpu')
    examples = []

    for a in range(10):
        for b in range(10):
            if a + b < 10:
                result = a + b
                input_spikes = torch.zeros(20, dtype=torch.bool, device=device)
                input_spikes[a] = True
                input_spikes[10 + b] = True
                examples.append(AdditionExample(a=a, b=b, result=result, input_spikes=input_spikes))

    return examples


class LazyGraphAdditionLoss(nn.Module):
    """Loss function for single-digit addition."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, output_potentials: torch.Tensor, target: int) -> torch.Tensor:
        logits = output_potentials / self.temperature
        target_t = torch.tensor([target], device=output_potentials.device)
        return torch.nn.functional.cross_entropy(logits.unsqueeze(0), target_t)


def create_lazy_addition_network(device: torch.device = None) -> LazyGraphSNN:
    """Create lazy graph SNN for single-digit addition."""
    return LazyGraphSNN(
        num_input=20,
        num_hidden=128,
        num_output=10,
        fan_out=64,
        tau=20.0,
        threshold=0.3,
        spike_delay=1,
        seed=42,
        device=device,
    )


def train_lazy_addition(
    num_epochs: int = 300,
    learning_rate: float = 0.03,
    max_timesteps: int = 20,
    device: torch.device = None,
):
    """Train lazy graph SNN on single-digit addition."""
    import random
    import torch.optim as optim

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("LAZY MEMBRANE - Single Digit Addition")
    print("=" * 60)

    network = create_lazy_addition_network(device)
    print(f"Neurons: {network.num_neurons}")
    print(f"Parameters: {sum(p.numel() for p in network.parameters())}")

    examples = generate_addition_dataset(device)
    print(f"Training examples: {len(examples)}")

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_fn = LazyGraphAdditionLoss(temperature=1.0)

    for epoch in range(num_epochs):
        random.shuffle(examples)
        total_loss = 0
        correct = 0

        for ex in examples:
            optimizer.zero_grad()

            output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['output_potentials']

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

    print("\nSample predictions:")
    for ex in examples[:10]:
        with torch.no_grad():
            output_times, info = network.forward(ex.input_spikes, max_timesteps=max_timesteps)
            output_pots = info['output_potentials']
            pred = network.decode_first_spike(output_times, output_pots)
            mark = "OK" if pred == ex.result else "X"
            print(f"  {ex.a} + {ex.b} = {ex.result}, pred={pred} [{mark}]")

    return network


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Quick test with single-digit addition
        train_lazy_addition(num_epochs=200, learning_rate=0.02)
    else:
        # Full two-digit addition test
        train_lazy_two_digit(
            num_epochs=300,
            num_train=2000,
            num_test=500,
            learning_rate=0.015
        )
