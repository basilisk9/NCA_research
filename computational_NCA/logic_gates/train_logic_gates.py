"""
Filename: train_logic_gates.py

Purpose: Train a 2D NCA to learn all fundamental logic gates (AND, OR, XOR, NAND, NOR, XNOR)
         Uses 16 channels with 2D convolution.
         Saves trained weights for each gate.

Key Parameters:
 - Grid size: (1, 16, 2, width) - 1 batch, 16 channels, 2 rows, variable width
 - Learning rate: 0.0001
 - Steps per example: 20
 - Training iterations: 30,000 per gate
 - Training range: 4-8 bits
 - Loss: BCEWithLogitsLoss 

Architecture:
 - Channel 0: Input (never modified)
   - Row 0: First binary string
   - Row 1: Second binary string
 - Channel 1: Output - checked by loss function (row 0 only)
 - Channels 2-15: Hidden state for computation
 - Conv2d: 16→15 channels, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Training Strategy:
 - Random binary strings of length 4-8 bits
 - Each gate trained independently

Expected Results:
 - Loss decreases to ~0.0000 for all gates
 - Training accuracy: 100%

Outputs:
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


def create_model():
    """
    Create a fresh NCA model.
    
    Returns:
        conv: Conv2d layer (16→15 channels, 3x3 kernel)
        optimizer: Adam optimizer with lr=0.0001
    """
    conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
    optimizer = torch.optim.Adam(conv.parameters(), lr=0.0001)
    return conv, optimizer


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


def loss_func(final_grid, target):
    """
    Compute BCE loss between output (channel 1, row 0) and target.
    
    Args:
        final_grid: NCA state after N steps
        target: Expected output tensor
    
    Returns:
        BCE loss (expects logits, applies sigmoid internally)
    """
    output = final_grid[0, 1, 0, :]
    loss = nn.BCEWithLogitsLoss()(output, target)
    return loss


def train_gate(gate_name, gate_func, iterations=30000):
    """
    Train NCA to learn a specific logic gate.
    
    Args:
        gate_name: Name of the gate (for logging)
        gate_func: Lambda function implementing the gate (a, b) -> result
        iterations: Number of training iterations
    
    Returns:
        conv: Trained convolution layer
    """
    print(f"\n{'='*50}")
    print(f"Training {gate_name}")
    print('='*50)
    
    conv, optimizer = create_model()
    
    for i in range(iterations):
        # Random bit width between 4-8
        bits = random.randint(4, 8)
        
        # Generate random binary inputs
        a = [random.randint(0, 1) for _ in range(bits)]
        b = [random.randint(0, 1) for _ in range(bits)]
        
        # Compute expected output using gate function
        result = [gate_func(a[j], b[j]) for j in range(bits)]
        
        # Initialize grid: channel 0 holds inputs
        grid = torch.zeros(1, 16, 2, bits).to(device)
        for j in range(bits):
            grid[0, 0, 0, j] = float(a[j])
            grid[0, 0, 1, j] = float(b[j])
        
        # Create target tensor
        target = torch.tensor([float(x) for x in result]).to(device)
        
        # Forward pass: 20 NCA steps
        optimizer.zero_grad()
        for _ in range(20):
            grid = step(grid, conv)
        
        # Compute loss and backpropagate
        loss = loss_func(grid, target)
        loss.backward()
        optimizer.step()
        
        # Logging every 10000 iterations
        if i % 10000 == 0:
            print(f"Iter {i} | loss={loss.item():.4f}")
    
    # Save weights
    torch.save(conv.state_dict(), f'{gate_name.lower()}_weights.pth')
    print(f"Saved {gate_name.lower()}_weights.pth")
    
    return conv


def train_all_gates():
    """
    Train all fundamental logic gates.
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
    
    for gate_name, gate_func in gates.items():
        train_gate(gate_name, gate_func)
    
    print(f"\n{'='*50}")
    print("Training complete! All weights saved.")
    print('='*50)


if __name__ == "__main__":
    train_all_gates()