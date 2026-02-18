# Heat Diffusion with Neural Cellular Automata

## Overview
This experiment explores whether an NCA can learn the local rules of heat diffusion
when given only input states and expected outputs. Instead of hand-coding the heat
equation, the NCA discovers the underlying physics purely from data. By training on
small grids and testing on grids up to 8x larger, we can see if the learned local
rules generalize beyond training data.

## Key Question

**Can an NCA learn the local rules of heat diffusion from data alone and generalize
to grid sizes it has never seen?**

## Architecture

- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Channel 0: Temperature field (input state, modified during evolution)
  - Channels 1-15: Hidden state (for learning diffusion dynamics)
- **Convolution**: 2D Conv (16→16 channels, 3×3 kernel, padding=1)
- **Activation**: tanh (bounds updates to [-1, 1])
- **Steps**: 5 forward passes per example
- **Loss**: MSELoss (mean squared error between predicted and target temperature)
- **Target Generation**: Ground truth computed using discrete heat equation:
  `T_new[i][j] = T[i][j] + α * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] - 4*T[i][j])`

## Files

### Training Scripts

**`train_heat_diffusion.py`**
- Main training script for heat diffusion
- Generates random initial temperature fields as training data
- Target is calculated by applying discrete heat equation for 5 steps
- Training grid sizes: 8-16
- Outputs: trained weights

### Testing Scripts

**`test_heat_diffusion.py`**
- Tests if NCA generalizes beyond training grid sizes
- Requires: Weights trained on 8-16 grids
- Tests: 16, 32, 64, 128 width grids (up to 8x training size)
- Compares NCA output against ground truth heat equation
- **Core experiment for proving learned physics vs memorization**

## Experiments & Results

### Experiment 1: Training Data Performance
**Setup**: Train on grid sizes 8-16, test on grid size 16
**Result**: MSE = 0.001028, Accuracy = 99.90%
**Conclusion**: NCA learned heat diffusion on training-sized grids with high accuracy

### Experiment 2: Generalization (Core Result)
**Setup**: Train on grid sizes 8-16, test on 16, 32, 64, 128

**Results**:

| Grid Width | MSE | Accuracy |
|------------|-----|----------|
| 16 (seen) | 0.001028 | 99.90% |
| 32 (unseen) | 0.000627 | 99.94% |
| 64 (unseen) | 0.000420 | 99.96% |
| 128 (unseen) | 0.000317 | 99.97% |

**Conclusion**: ✓ The NCA **learned heat diffusion**, not just memorized patterns.
Accuracy actually improves on larger grids, demonstrating that the NCA discovered
genuine local diffusion rules that compose correctly at any scale.

### Why Accuracy Improves With Scale
**Observation**: MSE decreases as grid size increases
**Explanation**: Boundary effects have proportionally less influence on larger grids.
The interior cells follow pure local rules which the NCA learned correctly. More
interior cells relative to boundary cells means higher overall accuracy.