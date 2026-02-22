"""
Filename: sort4_rank_crossentropy.py

Purpose: Train an NCA to sort 4 integers via ranking using cross-entropy loss.
         Instead of moving values, each cell classifies its rank (0-3) in the
         sorted order. Cross-entropy treats ranking as a classification problem
         with 4 classes per position.

Key Parameters:
 - Grid size: (1, 16, 1, 4) - 1 batch, 16 channels, 1 row, 4 elements
 - Learning rate: 0.001
 - Steps per example: 20
 - Training iterations: 500,000
 - Input range: 0-255 (normalized to [0, 1] via v/255)
 - Loss: CrossEntropyLoss (classification over 4 rank classes)

Architecture:
 - Channel 0: Input (never modified), holds normalized values
 - Channels 1-4: Output logits, one channel per possible rank (0-3)
   - Read as [width, num_ranks] after permutation for cross-entropy
 - Channels 5-15: Hidden state for computation
 - Conv2d: 16→15 channels, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Why Cross-Entropy Instead of MSE:
 - MSE on continuous rank values causes "interpolation trap" where
   NCA outputs average ramp instead of actual ranks
 - Cross-entropy forces discrete classification: each cell must pick
   exactly one rank from {0, 1, 2, 3}
 - Produces confidence scores per prediction via softmax

Why This Approach Has Limits:
 - Width 4 needs 4 output channels for 4 rank classes
 - Width 8 would need 8 output channels
 - Architecture must be redesigned for each width
 - Cannot generalize to unseen widths (structural limitation)
 - This is an encoding limitation, not an NCA limitation
   (NCA finds the local rule, but scaling is encoding-dependent)

Key Findings:
 - Converges reliably with high confidence predictions
 - Works well for fixed-width sorting
 - Proves NCA can learn comparison-based ranking as classification
 - Does not scale to variable widths without architecture change

Outputs:
 - sort4_rank_weights.pth: Trained model weights
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 16→15 channels: channel 0 is input-only, channels 1-15 are updated
conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
nn.init.xavier_uniform_(conv.weight, gain=0.1)
nn.init.zeros_(conv.bias)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.001)


def step(grid):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-15.
    Channel 0 (input) is never modified.

    Args:
        grid: Current grid state (1, 16, 1, 4)

    Returns:
        Updated grid with channel 0 unchanged, channels 1-15 incremented
    """
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:16, :, :] = grid[0, 1:16, :, :] + update[0, 0:15, :, :]
    return newGrid


def get_ranks(values):
    """
    Compute the rank of each value in the list.
    Rank 0 = smallest, Rank N-1 = largest.

    Example: [200, 50, 150] → [2, 0, 1]
    (50 is smallest=rank 0, 150 is middle=rank 1, 200 is largest=rank 2)

    Args:
        values: List of integers

    Returns:
        List of integer ranks, same length as input
    """
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank
    return ranks


def trainingLoop(num, values, iteration):
    """
    Train on one ranking problem: assign ranks 0-3 to 4 integers.

    Uses channels 1-4 as logits for 4 rank classes. Cross-entropy loss
    treats each position as an independent classification problem:
    "which rank does this position's value belong to?"

    Args:
        num: If 1, log output; if 0, silent
        values: List of 4 integers (0-255) to rank
        iteration: Current training iteration (for logging)
    """
    # Initialize grid: channel 0 holds normalized input values
    grid = torch.zeros(1, 16, 1, 4).to(device)
    for i in range(4):
        grid[0, 0, 0, i] = values[i] / 255.0  # normalize to [0, 1]

    # Compute target ranks
    ranks = get_ranks(values)
    target = torch.tensor(ranks, device=device).long()  # long for cross-entropy

    # Forward pass: 20 NCA steps
    optimizer.zero_grad()
    for _ in range(20):
        grid = step(grid)

    # Read logits from channels 1-4 (one channel per rank class)
    # Shape: [4 ranks, 4 positions] → permute to [4 positions, 4 ranks]
    logits = grid[0, 1:5, 0, :]
    logits = logits.permute(1, 0)

    # Cross-entropy: each position classifies into rank 0, 1, 2, or 3
    loss = nn.CrossEntropyLoss()(logits, target)

    # Logging
    if num == 1:
        pred_ranks = logits.argmax(dim=1).tolist()
        correct = pred_ranks == ranks
        # Softmax confidence for each prediction
        probs = torch.softmax(logits, dim=1)
        confidence = [round(probs[i][pred_ranks[i]].item(), 2) for i in range(4)]
        print(f"Iter {iteration} | Loss: {loss.item():.6f} | {'✓' if correct else '✗'}")
        print(f"  Input:      {values}")
        print(f"  Pred ranks: {pred_ranks}")
        print(f"  True ranks: {ranks}")
        print(f"  Confidence: {confidence}")
        print()

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(conv.parameters(), 1.0)
    optimizer.step()


if __name__ == "__main__":
    # Main training loop: 500k random ranking problems
    for i in range(500000):
        values = [random.randint(0, 255) for _ in range(4)]

        if i % 10000 == 0:
            trainingLoop(1, values, i)
        else:
            trainingLoop(0, values, i)

    # Save trained weights
    torch.save(conv.state_dict(), 'sort4_rank_weights.pth')
    print("Saved.")