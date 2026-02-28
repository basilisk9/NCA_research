"""
Filename: single_step_NCA.py

Purpose: Train a 2D NCA to almost perfectly learn the heat diffusion kernel. 
         Single step and minimal channels allow NCA to actually learn and 
         not have too many parameters to worry about

Key Parameters:
 - Grid size: (1, 1, 1, width) - 1 batch, 1 channel, 1 row, variable width
 - Learning rate: 0.001
 - Steps per example: 1
 - Training iterations: 10,000
 - Training width: 8-16 cells
 - Physics steps to predict: 1
 - Loss: MSELoss (continuous values)

Architecture:
 - Channel 0: Input (initial temperature) 
 - Channel 0: Output (final temperature)
 - Conv2d: 1→1 channels, kernel=(1,3), padding=(1,1)
 - Activation: None

Physics Being Learned:
 - Heat equation: T_new[i] = 0.5 * T[i] + 0.25 * T[i-1] + 0.25 * T[i+1]
 - Each cell averages with its neighbors
 - Purely local operation
 - Applied 1 time to get target

Training Strategy:
 - Random initial temperature distributions (uniform 0-1)
 - Random width between 8-16 cells
 - Target = result of 1 physics step

Testing Strategy
 - using the conv in memory, test generalization across steps and width
 - training ranges from 1 step in 16 wide grid to 1000 steps in 128 wide
 - MSE loss is lower than 0.00000000 in every case

Expected Results:
 - Loss decreases to ~0.000000001 or lower
 - NCA discovers heat diffusion without being told the equation

Outputs:
 - single_step_heat_diffusion_weights.pth: Trained model weights
"""
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

conv = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1), padding_mode='circular', bias=False).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.001)

def true_heat_step(T):
    left = torch.roll(T, 1, dims=-1)
    right = torch.roll(T, -1, dims=-1)
    return 0.5 * T + 0.25 * left + 0.25 * right

# Train: single step
for i in range(10000):
    width = torch.randint(8, 17, (1,)).item()
    T = torch.rand(1, 1, 1, width).to(device)
    target = true_heat_step(T)

    optimizer.zero_grad()
    pred = T + conv(T)
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"Step {i}: loss = {loss.item():.12f}")

print("\nLearned kernel:", conv.weight.data.squeeze().cpu().numpy())
print("Target kernel:  [0.25, -0.5, 0.25]")
torch.save(conv.state_dict(), 'single_step_heat_diffusion_weights.pth')
print("Weights saved")

# Test: iterate many steps
print("\n=== Generalization ===\n")
print(f"{'Steps':<10} {'Width 16':<15} {'Width 32':<15} {'Width 64':<15} {'Width 128':<15}")
print("-" * 70)

for steps in [1, 5, 10, 50, 100, 500, 1000]:
    row = f"{steps:<10}"
    for width in [16, 32, 64, 128]:
        total_error = 0
        for _ in range(100):
            T_nca = torch.rand(1, 1, 1, width).to(device)
            T_true = T_nca.clone()

            with torch.no_grad():
                for _ in range(steps):
                    T_nca = T_nca + conv(T_nca)
                    T_true = true_heat_step(T_true)

            error = torch.mean((T_nca - T_true) ** 2).item()
            total_error += error

        avg = total_error / 100
        row += f"{avg:<15.8f}"
    print(row)