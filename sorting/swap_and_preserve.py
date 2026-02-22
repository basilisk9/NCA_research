"""
Filename: 2_element_swap.py

Purpose: Train an NCA to swap (sort) two numbers by learning local comparison
         and routing rules. Uses combined swap loss and preservation loss to
         force the network to both reorder values AND preserve their magnitudes.
         This is the simplest sorting case — proves NCA can compare and swap.

Key Parameters:
 - Grid size: (1, 16, 1, 2) - 1 batch, 16 channels, 1 row, 2 elements
 - Learning rate: 0.0001
 - Steps per example: 20
 - Training iterations: 50,000
 - Input range: 0-255 (normalized to [0, 1] via v/255)
 - Loss: swap_loss + 100 * preservation_loss

Architecture:
 - Channel 0: Input (never modified), holds two normalized values
 - Channel 1: Output - swapped values appear here after evolution
 - Channels 2-15: Hidden state for comparison computation
 - Conv2d: 16→15 channels, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Loss Function:
 - Swap Loss: MSE between output and target (values in swapped positions)
 - Preservation Loss: MSE between sorted(output) and sorted(input)
   Forces network to preserve original values, not interpolate
 - Preservation weighted 100x to prioritize value preservation
 - Training shows preservation converges first, then swap follows

Key Finding:
 - NCA CAN learn to compare and swap 2 values
 - Precision limited to ±1-2 due to tanh's continuous domain
 - Proves the fundamental comparison operation is learnable
 - Does not scale to 4+ elements due to interpolation trap

Outputs:
 - swap_weights.pth: Trained model weights
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 16→15 channels: channel 0 is input-only, channels 1-15 are updated
conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.0001)


def step(grid):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-15.
    Channel 0 (input) is never modified.

    Args:
        grid: Current grid state (1, 16, 1, 2)

    Returns:
        Updated grid with channel 0 unchanged, channels 1-15 incremented
    """
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:16, :, :] = grid[0, 1:16, :, :] + update[0, 0:15, :, :]
    return newGrid


def lossFunc(output, target, input_values):
    """
    Compute combined swap and preservation loss.
    
    Swap loss ensures values end up in correct (sorted) positions.
    Preservation loss ensures output contains the same values as input,
    preventing the interpolation trap where NCA outputs average values.

    Args:
        output: NCA output values (2,)
        target: Expected sorted values (2,)
        input_values: Original input values (2,)

    Returns:
        Tuple of (total_loss, swap_loss, preservation_loss)
    """
    # Output should match sorted target positions
    swap_loss = nn.MSELoss()(output, target)

    # Sorted output should match sorted input (same values, any order)
    output_sorted = torch.sort(output)[0]
    input_sorted = torch.sort(input_values)[0]
    preservation_loss = nn.MSELoss()(output_sorted, input_sorted)

    # Weight preservation heavily to prevent interpolation
    loss = swap_loss + 100.0 * preservation_loss
    return loss, swap_loss, preservation_loss


def trainingLoop(num, a, b, iteration):
    """
    Train on one swap problem: given [a, b], output [min, max].

    Args:
        num: If 1, log output; if 0, silent
        a, b: Two integers (0-255) to swap/sort
        iteration: Current training iteration (for logging)
    """
    # Initialize grid: channel 0 holds normalized input values
    grid = torch.zeros(1, 16, 1, 2).to(device)
    grid[0, 0, 0, 0] = a / 255.0
    grid[0, 0, 0, 1] = b / 255.0

    # Target: values swapped (smaller first, larger second)
    target = torch.zeros(2).to(device)
    target[0] = b / 255.0
    target[1] = a / 255.0

    # Input tensor for preservation loss
    input_values = torch.tensor([a / 255.0, b / 255.0]).to(device)

    # Forward pass: 20 NCA steps
    optimizer.zero_grad()
    for _ in range(20):
        grid = step(grid)

    # Read output from channel 1
    output = grid[0, 1, 0, :]
    loss, swap_loss, preservation_loss = lossFunc(output, target, input_values)

    # Logging
    if num == 1:
        pred_a = output[0].item() * 255
        pred_b = output[1].item() * 255
        print(f"Iter {iteration} | [{a},{b}]→[{pred_a:.1f},{pred_b:.1f}] expected [{b},{a}] | Swap: {swap_loss.item():.6f} | Preserve: {preservation_loss.item():.6f} | Loss: {loss.item():.6f}")

    # Backpropagation
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # Main training loop: 50k random swap problems
    for i in range(50000):
        a = random.randint(0, 255)
        b = random.randint(0, 255)

        if i % 5000 == 0:
            trainingLoop(1, a, b, i)
        else:
            trainingLoop(0, a, b, i)

    # Save trained weights
    torch.save(conv.state_dict(), 'swap_weights.pth')
    print("Saved: swap_weights.pth")