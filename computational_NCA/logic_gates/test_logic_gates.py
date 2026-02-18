"""
Filename: test_logic_gates.py

Purpose: Test trained NCA logic gates on various input sizes.
         Measures generalization from 4-8 bit training to 128 bit testing.

Key Parameters:
 - Test sizes: 4, 8, 16, 32, 64, 128 bits
 - Samples per size: 500
 - Steps per example: 20

Architecture:
 - Same as training: 16 channels, 3×3 conv, tanh activation
 - Loads weights from {gate_name}_weights.pth

Expected Results:
 - All gates: 100% accuracy at all test sizes
 - 16x generalization (8-bit training → 128-bit testing)

Inputs Required:
 - and_weights.pth
 - or_weights.pth
 - xor_weights.pth
 - nand_weights.pth
 - nor_weights.pth
 - xnor_weights.pth
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")


def step(grid, conv):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-15.
    
    Args:
        grid: Current grid state (1, 16, 2, width)
        conv: The convolution layer
    
    Returns:
        Updated grid with channel 0 unchanged, channels 1-15 incremented
    """
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:16, :, :] = grid[0, 1:16, :, :] + update[0, 0:15, :, :]
    return newGrid


def load_model(gate_name):
    """
    Load trained weights for a specific gate.
    
    Args:
        gate_name: Name of the gate (lowercase for filename)
    
    Returns:
        conv: Conv2d layer with loaded weights
    """
    conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
    conv.load_state_dict(torch.load(f'{gate_name.lower()}_weights.pth'))
    conv.eval()
    return conv


def test_gate(conv, gate_name, gate_func, test_bits_list=[4, 8, 16, 32, 64, 128], samples=500):
    """
    Test NCA generalization on a specific logic gate.
    
    Args:
        conv: Trained convolution layer
        gate_name: Name of the gate (for logging)
        gate_func: Lambda function implementing the gate
        test_bits_list: List of bit widths to test
        samples: Number of samples per bit width
    
    Returns:
        results: Dict mapping bit width to accuracy percentage
    """
    print(f"\nTesting {gate_name}:")
    
    results = {}
    
    for bits in test_bits_list:
        correct = 0
        
        for _ in range(samples):
            # Generate random binary inputs
            a = [random.randint(0, 1) for _ in range(bits)]
            b = [random.randint(0, 1) for _ in range(bits)]
            
            # Compute expected output
            result = [gate_func(a[j], b[j]) for j in range(bits)]
            
            # Initialize grid
            grid = torch.zeros(1, 16, 2, bits).to(device)
            for j in range(bits):
                grid[0, 0, 0, j] = float(a[j])
                grid[0, 0, 1, j] = float(b[j])
            
            # Forward pass
            with torch.no_grad():
                for _ in range(20):
                    grid = step(grid, conv)
            
            # Get predictions
            output = torch.sigmoid(grid[0, 1, 0, :]).cpu().numpy()
            predicted = [1 if x > 0.5 else 0 for x in output]
            
            # Check if correct
            if predicted == result:
                correct += 1
        
        accuracy = 100 * correct / samples
        results[bits] = accuracy
        print(f"  {bits} bits: {correct}/{samples} = {accuracy:.1f}%")
    
    return results


def test_all_gates():
    """
    Test all fundamental logic gates and print summary.
    
    Returns:
        all_results: Dict mapping gate name to results dict
    """
    # Define all logic gates
    gates = {
        "AND":  lambda a, b: a & b,
        "OR":   lambda a, b: a | b,
        "XOR":  lambda a, b: a ^ b,
        "NAND": lambda a, b: 1 - (a & b),
        "NOR":  lambda a, b: 1 - (a | b),
        "XNOR": lambda a, b: 1 - (a ^ b),
    }
    
    all_results = {}
    
    for gate_name, gate_func in gates.items():
        try:
            conv = load_model(gate_name)
            results = test_gate(conv, gate_name, gate_func)
            all_results[gate_name] = results
        except FileNotFoundError:
            print(f"\nWarning: {gate_name.lower()}_weights.pth not found. Skipping.")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: All Logic Gates Generalization")
    print('='*70)
    print(f"{'Gate':<6} | {'4-bit':<8} | {'8-bit':<8} | {'16-bit':<8} | {'32-bit':<8} | {'64-bit':<8} | {'128-bit':<8}")
    print('-'*70)
    
    for gate_name, results in all_results.items():
        row = f"{gate_name:<6}"
        for bits in [4, 8, 16, 32, 64, 128]:
            row += f" | {results[bits]:<8.1f}"
        print(row)
    
    print('='*70)
    
    return all_results


if __name__ == "__main__":
    all_results = test_all_gates()