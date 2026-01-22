"""
Main entry point for the Spiking Neural Network implementation.

Novel SNN architecture featuring:
- Timing Wheel Scheduler: O(1) event scheduling and retrieval
- Lazy Membrane Potential: Only compute when spikes arrive
- Procedural Connectivity: Deterministic connection regeneration
- Spike-Graph Backprop: Record spike events for gradient computation

Training task: Single digit addition (a + b where a + b < 10)
"""

import torch
import argparse
from typing import Optional

from spiking_network import SpikingNetworkWithGrad, create_addition_network
from training import (
    generate_addition_dataset,
    train_network,
    evaluate,
    AdditionExample
)


def demo_components():
    """Demonstrate individual components working."""
    print("=" * 60)
    print("COMPONENT DEMONSTRATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 1. Timing Wheel Demo
    print("\n--- Timing Wheel Demo ---")
    from timing_wheel import TimingWheel
    tw = TimingWheel(num_slots=16, device=device)
    tw.schedule(0, 0)  # Neuron 0 fires at t=0
    tw.schedule(1, 0)  # Neuron 1 fires at t=0
    tw.schedule(5, 2)  # Neuron 5 fires at t=2

    print(f"Spikes at t=0: {tw.get_current_spikes().tolist()}")
    tw.advance()
    print(f"Spikes at t=1: {tw.get_current_spikes().tolist()}")
    tw.advance()
    print(f"Spikes at t=2: {tw.get_current_spikes().tolist()}")

    # 2. Procedural Connectivity Demo
    print("\n--- Procedural Connectivity Demo ---")
    from procedural_connectivity import ProceduralConnectivity
    conn = ProceduralConnectivity(
        num_src=10, num_dst=20, fan_out=5, seed=42, device=device
    )
    print(f"Targets for neuron 0: {conn.get_targets(0).tolist()}")
    print(f"Targets for neuron 1: {conn.get_targets(1).tolist()}")
    print(f"Weights shape: {conn.weights.shape}")
    print(f"Weights for neuron 0: {conn.get_weights(0).tolist()[:3]}...")

    # Verify determinism
    targets_again = conn.get_targets(0).tolist()
    print(f"Same targets on second call: {conn.get_targets(0).tolist() == targets_again}")

    # 3. Network Forward Pass Demo
    print("\n--- Network Forward Pass Demo ---")
    network = create_addition_network(device)
    print(f"Network layers: {network.layer_sizes}")
    print(f"Total neurons: {network.total_neurons}")
    print(f"Layer offsets: {network.layer_offsets}")

    # Test with input 3 + 5 = 8
    input_spikes = torch.zeros(20, dtype=torch.bool, device=device)
    input_spikes[3] = True    # First digit: 3
    input_spikes[10 + 5] = True  # Second digit: 5

    output_times, info = network.forward(input_spikes, max_timesteps=30)
    print(f"Input: 3 + 5 = 8")
    print(f"Output spike times: {output_times.tolist()}")
    print(f"Total spike events recorded: {info['total_spikes']}")
    print(f"Prediction: {network.decode_first_spike(output_times)}")


def run_training(
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    seed: Optional[int] = None
):
    """Run the full training pipeline."""
    print("=" * 60)
    print("TRAINING SPIKING NEURAL NETWORK")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"Random seed: {seed}")

    # Create network
    network = create_addition_network(device)
    print(f"\nNetwork Architecture:")
    print(f"  Input:  {network.layer_sizes[0]} neurons (two 10-digit one-hot encodings)")
    print(f"  Hidden: {network.layer_sizes[1]} neurons")
    print(f"  Output: {network.layer_sizes[2]} neurons (one per digit 0-9)")
    print(f"  Total:  {network.total_neurons} neurons")

    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"  Learnable parameters: {num_params}")

    # Train
    print(f"\nTraining for {num_epochs} epochs with lr={learning_rate}")
    print("-" * 40)

    history = train_network(
        network,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        verbose=True
    )

    return network, history


def interactive_test(network: SpikingNetworkWithGrad):
    """Interactive testing of trained network."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)
    print("Enter two single digits to add (or 'q' to quit)")

    device = next(network.parameters()).device

    while True:
        try:
            user_input = input("\nEnter 'a b' (e.g., '3 5'): ").strip()
            if user_input.lower() == 'q':
                break

            parts = user_input.split()
            if len(parts) != 2:
                print("Please enter exactly two numbers")
                continue

            a, b = int(parts[0]), int(parts[1])

            if not (0 <= a <= 9 and 0 <= b <= 9):
                print("Please enter digits 0-9")
                continue

            if a + b >= 10:
                print(f"Sum {a + b} >= 10, not in training distribution")
                continue

            # Create input
            input_spikes = torch.zeros(20, dtype=torch.bool, device=device)
            input_spikes[a] = True
            input_spikes[10 + b] = True

            # Run network
            with torch.no_grad():
                output_times, info = network.forward(input_spikes, max_timesteps=50)

            prediction = network.decode_first_spike(output_times)
            correct = a + b

            print(f"  {a} + {b} = {correct}")
            print(f"  Network prediction: {prediction}")
            print(f"  Output spike times: {output_times.tolist()}")
            print(f"  {'CORRECT' if prediction == correct else 'WRONG'}")

        except ValueError:
            print("Invalid input, please enter two digits")
        except KeyboardInterrupt:
            break


def benchmark_speed():
    """Benchmark the network speed."""
    print("\n" + "=" * 60)
    print("SPEED BENCHMARK")
    print("=" * 60)

    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    network = create_addition_network(device)
    examples = generate_addition_dataset(device)

    # Warmup
    for example in examples[:5]:
        network.forward(example.input_spikes, max_timesteps=30)

    # Benchmark
    num_iterations = 100
    start_time = time.time()

    for i in range(num_iterations):
        example = examples[i % len(examples)]
        network.forward(example.input_spikes, max_timesteps=30)

    elapsed = time.time() - start_time

    print(f"Forward passes: {num_iterations}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Time per pass: {elapsed/num_iterations*1000:.2f}ms")
    print(f"Passes per second: {num_iterations/elapsed:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Spiking Neural Network for digit addition"
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['demo', 'train', 'benchmark'],
        help='Mode to run: demo, train, or benchmark'
    )
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=0.02,
        help='Learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--interactive', action='store_true',
        help='Run interactive testing after training'
    )

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_components()

    elif args.mode == 'train':
        network, history = run_training(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed
        )

        if args.interactive:
            interactive_test(network)

    elif args.mode == 'benchmark':
        benchmark_speed()


if __name__ == "__main__":
    main()
