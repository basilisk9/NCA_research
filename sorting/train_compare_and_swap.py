"""
Filename: 2_element_sort_extended.py

Purpose: Train an NCA to sort two numbers using expanded capacity.
         Compared to 2_element_swap.py, this uses more channels (32 vs 16),
         more steps (30 vs 20), more iterations (500k vs 50k), and
         simpler MSE loss without explicit preservation term.
         Tests whether raw capacity can solve sorting without loss engineering.

Key Parameters:
 - Grid size: (1, 32, 1, 2) - 1 batch, 32 channels, 1 row, 2 elements
 - Learning rate: 0.0001
 - Steps per example: 30
 - Training iterations: 500,000
 - Input range: 0-255 (normalized to [0, 1] via v/255)
 - Loss: Pure MSE on sorted output (no preservation term)

Architecture:
 - Channel 0: Input (never modified), holds two normalized values
 - Channel 1: Output - sorted values appear here (small, big)
 - Channels 2-31: Hidden state (doubled from 16-channel version)
 - Conv2d: 32→31 channels, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Why 32 Channels and 30 Steps:
 - 2-element swap worked with 16 channels but precision was ±1-2
 - Hypothesis: more hidden state allows better value preservation
 - More steps allow finer convergence to target values
 - Tests the capacity scaling law discovered in subtraction experiments

Key Finding:
 - Works but requires ~500k iterations to converge
 - Still limited to ±1-2 precision due to tanh
 - Proves sorting is possible but slow with pure MSE
 - Does not scale to 4+ elements (interpolation trap)

Outputs:
 - comapre_swap_weights.pth: Trained model weights
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 32→31 channels: channel 0 is input-only, channels 1-31 are updated
# Doubled capacity compared to base 16-channel architecture
conv = nn.Conv2d(32, 31, kernel_size=(3, 3), padding=(1, 1)).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.0001)


def step(grid):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-31.
    Channel 0 (input) is never modified.

    Args:
        grid: Current grid state (1, 32, 1, 2)

    Returns:
        Updated grid with channel 0 unchanged, channels 1-31 incremented
    """
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:32, :, :] = grid[0, 1:32, :, :] + update[0, 0:31, :, :]
    return newGrid


def trainingLoop(num, a, b, iteration):
    """
    Train on one sorting problem: given [a, b], output [min(a,b), max(a,b)].
    Uses pure MSE loss without explicit preservation term.

    Args:
        num: If 1, log output; if 0, silent
        a, b: Two integers (0-255) to sort
        iteration: Current training iteration (for logging)
    """
    # Initialize grid: channel 0 holds normalized input values
    grid = torch.zeros(1, 32, 1, 2).to(device)
    grid[0, 0, 0, 0] = a / 255.0
    grid[0, 0, 0, 1] = b / 255.0
    
    # Target: sorted values (smallest first)
    small = min(a, b)
    big = max(a, b)
    
    target = torch.zeros(2).to(device)
    target[0] = small / 255.0
    target[1] = big / 255.0

    # Forward pass: 30 NCA steps (more than base 20)
    optimizer.zero_grad()
    for _ in range(30):
        grid = step(grid)

    # Read output from channel 1
    output = grid[0, 1, 0, :]
    
    # Pure MSE loss - no preservation term
    loss = nn.MSELoss()(output, target)

    # Logging
    if num == 1:
        pred_small = round(output[0].item() * 255)
        pred_big = round(output[1].item() * 255)
        correct = (pred_small == small and pred_big == big)
        print(f"Iter {iteration} | [{a},{b}]→[{pred_small},{pred_big}] expected [{small},{big}] | {'✓' if correct else '✗'} | Loss: {loss.item():.6f}")

    # Backpropagation
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # Main training loop: 500k random sorting problems
    # Much longer than 50k to allow pure MSE to converge
    for i in range(500000):
        a = random.randint(0, 255)
        b = random.randint(0, 255)

        if i % 10000 == 0:
            trainingLoop(1, a, b, i)
        else:
            trainingLoop(0, a, b, i)

    # Save trained weights
    torch.save(conv.state_dict(), 'comapre_swap_weights.pth')
    print("Saved: comapre_swap_weights.pth")