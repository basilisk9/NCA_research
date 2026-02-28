"""
Filename: train_heat_diffusion.py

Purpose: Train a 2D NCA to learn heat diffusion physics from examples.
         NCA observes input temperature distributions and learns to predict
         the result after 5 steps of heat diffusion.
         No physics equations given - NCA discovers the rules from data.

Key Parameters:
 - Grid size: (1, 16, 1, width) - 1 batch, 16 channels, 1 row, variable width
 - Learning rate: 0.0001
 - Steps per example: 20
 - Training iterations: 50,000
 - Training width: 8-16 cells
 - Physics steps to predict: 5
 - Loss: MSELoss (continuous values)

Architecture:
 - Channel 0: Input (initial temperature distribution, never modified)
 - Channel 1: Output (predicted temperature after diffusion)
 - Channels 2-15: Hidden state for computation
 - Conv2d: 16→15 channels, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Physics Being Learned:
 - Heat equation: T_new[i] = 0.5 * T[i] + 0.25 * T[i-1] + 0.25 * T[i+1]
 - Each cell averages with its neighbors
 - Purely local operation
 - Applied 5 times to get target

Training Strategy:
 - Random initial temperature distributions (uniform 0-1)
 - Random width between 8-16 cells
 - Target = result of 5 physics steps

Expected Results:
 - Loss decreases to ~0.001 or lower
 - NCA discovers heat diffusion without being told the equation

Outputs:
 - heat_diffusion_weights.pth: Trained model weights
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available (for Colab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 16→15 channels: channel 0 is input-only, channels 1-15 are updated
conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.0001)

def true_heat_step(grid):
    """
    Apply one step of true heat diffusion physics.
    
    Heat equation (discrete):
    T_new[i] = 0.5 * T[i] + 0.25 * T[i-1] + 0.25 * T[i+1]
    
    Each cell becomes average of itself and neighbors.
    Uses periodic boundary conditions (wraps around).
    
    Args:
        grid: Temperature distribution tensor (1, 1, 1, width)
    
    Returns:
        Updated temperature distribution after one diffusion step
    """
    left = torch.roll(grid, 1, dims=-1)
    right = torch.roll(grid, -1, dims=-1)
    return 0.5 * grid + 0.25 * left + 0.25 * right

def step(grid):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-15.
    
    Args:
        grid: Current grid state (1, 16, 1, width)
    
    Returns:
        Updated grid with channel 0 unchanged, channels 1-15 incremented
    """
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:16, :, :] = grid[0, 1:16, :, :] + update[0, 0:15, :, :]
    return newGrid

def lossFunc(finalGrid, target):
    """
    Compute MSE loss between output (channel 1, row 0) and target.
    
    Uses MSE instead of BCE because temperature is continuous (0-1),
    not binary like logic gates or addition.
    
    Args:
        finalGrid: NCA state after 20 steps
        target: Expected temperature distribution after physics
    
    Returns:
        MSE loss (mean squared error)
    """
    output = finalGrid[0, 1, 0, :]
    target_slice = target[0, 0, 0, :]
    loss = nn.MSELoss()(output, target_slice)
    return loss

def trainingLoop(num, width, iteration, steps):
    """
    Train on one heat diffusion problem.
    
    Args:
        num: If 1, log output; if 0, silent
        width: Grid width for this example
        iteration: Current training iteration (for logging)
    """
    # Random initial temperature distribution
    initial = torch.rand(1, 1, 1, width).to(device)
    
    # Apply true physics 5 times to get target
    target = initial.clone()
    for _ in range(steps):
        target = true_heat_step(target)
    
    # Initialize grid: channel 0 holds input temperature
    grid = torch.zeros(1, 16, 1, width).to(device)
    grid[0, 0, 0, :] = initial[0, 0, 0, :]
    
    # Forward pass: 20 NCA steps
    optimizer.zero_grad()
    for _ in range(steps):
        grid = step(grid)
    loss = lossFunc(grid, target)
    
    if num == 1:
        print(f"Iter {iteration} | width={width} | loss={loss.item():.6f}")
    
    # Backpropagation
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    
    # Main training loop: 100k random examples
    for i in range(100000):
        width = random.randint(8, 16)
        steps = random.randint(5, 20)
        if i % 5000 == 0:
            trainingLoop(1, width, i, steps)
        else:
            trainingLoop(0, width, i, steps)
    
    # Save final weights    
    torch.save(conv.state_dict(), 'heat_diffusion_through_time.pth')