"""
Training module for Spiking Neural Network.

Implements:
- Data generation for single-digit addition task
- Custom loss function for first-to-fire classification
- Training loop with spike-graph backpropagation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random

from spiking_network import SpikingNetworkWithGrad, create_addition_network


@dataclass
class AdditionExample:
    """A single addition training example."""
    a: int
    b: int
    result: int
    input_spikes: torch.Tensor  # [20] boolean tensor


def generate_addition_dataset(device: torch.device = None) -> List[AdditionExample]:
    """
    Generate all valid single-digit addition examples where a + b < 10.

    There are 45 such examples:
    - 0+0, 0+1, ..., 0+9 (10 examples)
    - 1+0, 1+1, ..., 1+8 (9 examples)
    - ...
    - 9+0 (1 example)

    Actually for a + b < 10:
    (0,0) through (0,9): 10
    (1,0) through (1,8): 9
    (2,0) through (2,7): 8
    ...
    (9,0): 1
    Total: 10+9+8+7+6+5+4+3+2+1 = 55

    But we want a + b < 10, so actually:
    (0,0) to (0,9): but 0+9=9 which is < 10, so 10 examples
    (1,0) to (1,8): 1+8=9, so 9 examples
    ...
    (9,0): 9+0=9, so 1 example

    Total = 55 examples where sum <= 9

    Wait, the user said 45 examples. Let me recalculate for a + b < 10:
    That means sum can be 0,1,2,3,4,5,6,7,8,9 (sum < 10 means sum <= 9)

    For each sum s from 0 to 9, number of pairs (a,b) where a,b in [0,9] and a+b=s:
    s=0: (0,0) = 1
    s=1: (0,1), (1,0) = 2
    s=2: (0,2), (1,1), (2,0) = 3
    ...
    s=9: (0,9), (1,8), ..., (9,0) = 10

    Total = 1+2+3+4+5+6+7+8+9+10 = 55

    The user said 45, but mathematically there are 55 valid pairs.
    Let me generate all 55, or we could exclude duplicates like (a,b) and (b,a)?

    Actually let's just generate all pairs where a + b < 10.
    """
    device = device or torch.device('cpu')
    examples = []

    for a in range(10):
        for b in range(10):
            if a + b < 10:
                result = a + b

                # Create input spike pattern
                # First 10 neurons: one-hot encoding of a
                # Next 10 neurons: one-hot encoding of b
                input_spikes = torch.zeros(20, dtype=torch.bool, device=device)
                input_spikes[a] = True       # First digit
                input_spikes[10 + b] = True  # Second digit

                examples.append(AdditionExample(
                    a=a,
                    b=b,
                    result=result,
                    input_spikes=input_spikes
                ))

    return examples


class FirstSpikeLoss(nn.Module):
    """
    Loss function for first-to-fire classification.

    Encourages the target neuron to fire first while penalizing
    incorrect neurons that fire early.
    """

    def __init__(
        self,
        max_time: float = 50.0,
        margin: float = 3.0,
        non_fire_penalty: float = 10.0
    ):
        """
        Args:
            max_time: Maximum time for normalization
            margin: Desired time gap between correct and incorrect
            non_fire_penalty: Penalty when target doesn't fire
        """
        super().__init__()
        self.max_time = max_time
        self.margin = margin
        self.non_fire_penalty = non_fire_penalty

    def forward(
        self,
        output_potentials: torch.Tensor,
        output_spike_times: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            output_potentials: Final membrane potentials of output neurons
            output_spike_times: When each output neuron fired (-1 if didn't fire)
            target_class: Index of correct class

        Returns:
            Scalar loss tensor
        """
        device = output_potentials.device
        num_classes = len(output_potentials)

        # Create soft spike times that are differentiable
        # For neurons that fired: use spike time
        # For neurons that didn't fire: use max_time - potential (closer to threshold = earlier potential spike)
        soft_times = torch.zeros(num_classes, device=device)

        for i in range(num_classes):
            if output_spike_times[i] >= 0:
                # Use spike time, scaled
                soft_times[i] = output_spike_times[i].float()
            else:
                # Didn't fire: higher potential = lower effective time
                # This creates gradient to increase potential
                soft_times[i] = self.max_time - output_potentials[i].clamp(min=0)

        target_time = soft_times[target_class]

        # Multi-margin loss: target should fire before all others by margin
        loss = torch.tensor(0.0, device=device)

        for i in range(num_classes):
            if i != target_class:
                # Hinge loss: penalize if target fires too late
                hinge = torch.relu(target_time - soft_times[i] + self.margin)
                loss = loss + hinge

        # Additional penalty if target didn't fire at all
        if output_spike_times[target_class] < 0:
            # Strong penalty + gradient to increase potential
            loss = loss + self.non_fire_penalty * (1.0 - output_potentials[target_class].clamp(max=1.0))

        return loss / num_classes


class SpikeTimingLoss(nn.Module):
    """
    Alternative loss based on spike timing directly.

    Uses soft-argmin style loss over spike times.
    """

    def __init__(self, temperature: float = 1.0, max_time: float = 50.0):
        super().__init__()
        self.temperature = temperature
        self.max_time = max_time

    def forward(
        self,
        output_potentials: torch.Tensor,
        output_spike_times: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        device = output_potentials.device
        num_classes = len(output_potentials)

        # Convert spike times to soft scores (lower time = higher score)
        scores = torch.zeros(num_classes, device=device)

        for i in range(num_classes):
            if output_spike_times[i] >= 0:
                # Fired: score based on how early
                scores[i] = self.max_time - output_spike_times[i].float()
            else:
                # Didn't fire: score based on potential (higher = better)
                scores[i] = output_potentials[i].clamp(min=-10, max=10)

        # Cross-entropy style loss
        log_probs = torch.log_softmax(scores / self.temperature, dim=0)
        loss = -log_probs[target_class]

        return loss


class PotentialBasedLoss(nn.Module):
    """
    Loss based directly on membrane potentials for better gradient flow.
    """

    def __init__(self, threshold: float = 0.5, temperature: float = 1.0, margin: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        output_potentials: torch.Tensor,
        output_spike_times: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """
        Combined loss: cross-entropy + margin ranking.

        Encourages target to have highest potential with margin over others.
        """
        device = output_potentials.device

        # Ensure requires_grad
        if not output_potentials.requires_grad:
            output_potentials = output_potentials.clone().requires_grad_(True)

        # Cross-entropy component
        logits = output_potentials / self.temperature
        target = torch.tensor([target_class], device=device)
        ce_loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)

        # Margin ranking: target should be higher than all others by margin
        target_potential = output_potentials[target_class]
        margin_loss = torch.tensor(0.0, device=device)

        for i in range(len(output_potentials)):
            if i != target_class:
                # Hinge loss: penalize if target is not margin higher than other
                diff = output_potentials[i] - target_potential + self.margin
                margin_loss = margin_loss + torch.relu(diff)

        # Combine losses
        loss = ce_loss + 0.5 * margin_loss

        return loss


def train_epoch(
    network: SpikingNetworkWithGrad,
    examples: List[AdditionExample],
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        network: The spiking network
        examples: Training examples
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        device: PyTorch device

    Returns:
        Tuple of (average_loss, accuracy)
    """
    random.shuffle(examples)

    total_loss = 0.0
    correct = 0

    for example in examples:
        optimizer.zero_grad()

        # Forward pass
        output_spike_times, info = network.forward(
            example.input_spikes,
            max_timesteps=50
        )

        # Get output potentials from layer_potentials (has gradient)
        output_potentials = info['layer_potentials'][-1]

        # Compute loss
        loss = loss_fn(output_potentials, output_spike_times, example.result)

        # Backward pass
        if loss.requires_grad:
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

            optimizer.step()

        total_loss += loss.item()

        # Check accuracy
        prediction = network.decode_first_spike(output_spike_times)
        if prediction == example.result:
            correct += 1

    avg_loss = total_loss / len(examples)
    accuracy = correct / len(examples)

    return avg_loss, accuracy


def evaluate(
    network: SpikingNetworkWithGrad,
    examples: List[AdditionExample],
    device: torch.device,
    verbose: bool = False
) -> float:
    """
    Evaluate network accuracy.

    Args:
        network: The spiking network
        examples: Test examples
        device: PyTorch device
        verbose: Print individual predictions

    Returns:
        Accuracy as fraction
    """
    correct = 0

    with torch.no_grad():
        for example in examples:
            output_spike_times, _ = network.forward(
                example.input_spikes,
                max_timesteps=50
            )

            prediction = network.decode_first_spike(output_spike_times)

            if verbose:
                print(f"{example.a} + {example.b} = {example.result}, "
                      f"predicted: {prediction}, "
                      f"spike_times: {output_spike_times.tolist()}")

            if prediction == example.result:
                correct += 1

    return correct / len(examples)


def train_network(
    network: SpikingNetworkWithGrad,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    device: torch.device = None,
    verbose: bool = True
) -> Dict:
    """
    Full training loop.

    Args:
        network: Network to train
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: PyTorch device
        verbose: Print progress

    Returns:
        Dictionary with training history
    """
    # Infer device from network if not provided
    if device is None:
        device = next(network.parameters()).device

    # Generate dataset
    examples = generate_addition_dataset(device)
    print(f"Generated {len(examples)} training examples")

    # Setup optimizer and loss
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = PotentialBasedLoss(threshold=1.0, temperature=0.5)

    # Training history
    history = {
        'loss': [],
        'accuracy': [],
    }

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Train
        avg_loss, accuracy = train_epoch(
            network, examples, optimizer, loss_fn, device
        )

        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Loss = {avg_loss:.4f}, "
                  f"Accuracy = {accuracy:.2%}")

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.2%}")

    # Final evaluation with verbose output
    if verbose:
        print("\nSample predictions:")
        sample_examples = random.sample(examples, min(10, len(examples)))
        evaluate(network, sample_examples, device, verbose=True)

    return history


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    network = create_addition_network(device)
    history = train_network(network, num_epochs=50, learning_rate=0.01, device=device)
