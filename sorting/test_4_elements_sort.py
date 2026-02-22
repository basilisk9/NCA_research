"""
Filename: test_4_elements_sort.py

Purpose: Evaluate the trained 5-phase NCA sorting model on three test suites:
         1. Random width-4 arrays (training distribution)
         2. Close-valued arrays (hardest discrimination cases)
         3. Width 5 and 6 arrays (never-seen generalization test)
         Uses ±5 tolerance to account for tanh precision limits.

Key Parameters:
 - Weights file: sort_5phase_no_clear_weights.pth
 - Channels: 32
 - Steps per phase: 20 (100 total across 5 phases)
 - Normalization: (value - 128) / 64 → input range [-2, 2]
 - De-normalization: output * 64 + 128 → back to [0, 255]

Evaluation Metrics:
 - Exact match: predicted values exactly equal sorted target
 - Within ±5 and ordered: each predicted value within 5 of target
   AND output is in ascending order. Accounts for tanh precision
   loss over 100 nonlinear steps.
 - Close value accuracy: tests on values spaced 1-15 apart,
   the hardest case for NCA discrimination

Expected Results:
 - Width 4 exact: ~0% (tanh precision prevents exact match)
 - Width 4 within ±5: ~60% (values close, order correct)
 - Close values: ~67% (harder but still works partially)
 - Width 5-6: 0% (no generalization, confirms axiom prediction)

Key Finding:
 - NCA learns value routing (actually moves values) but precision
   is limited by tanh accumulation over 100 steps
 - Ranking approach achieves higher accuracy (~91%) but doesn't
   move values, only computes position indices
 - Generalization to unseen widths fails for both approaches,
   confirming that scaling is encoding-dependent, not NCA-dependent

Outputs:
 - Console: detailed test results with accuracy summaries
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

channels = 32

# Load all 5 phase networks
conv1 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv3 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv4 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv5 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)

weights = torch.load('/kaggle/working/sort_5phase_no_clear_weights.pth', map_location=device)
conv1.load_state_dict(weights['conv1'])
conv2.load_state_dict(weights['conv2'])
conv3.load_state_dict(weights['conv3'])
conv4.load_state_dict(weights['conv4'])
conv5.load_state_dict(weights['conv5'])

# Set all phases to eval mode (disables dropout/batchnorm if present)
for conv in [conv1, conv2, conv3, conv4, conv5]:
    conv.eval()


def step(grid, conv):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-31.
    Channel 0 (input) is never modified.

    Args:
        grid: Current grid state (1, 32, 1, width)
        conv: Which phase's conv to apply

    Returns:
        Updated grid with channel 0 unchanged, channels 1-31 incremented
    """
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:channels, :, :] = grid[0, 1:channels, :, :] + update[0, :, :, :]
    return newGrid


def inference(values):
    """
    Run trained 5-phase NCA on input values to produce sorted output.
    No gradient computation. All 5 phases run sequentially.

    Args:
        values: List of integers (0-255) to sort

    Returns:
        List of predicted sorted integers, de-normalized back to [0, 255]
    """
    width = len(values)
    grid = torch.zeros(1, channels, 1, width).to(device)
    for i in range(width):
        grid[0, 0, 0, i] = (values[i] - 128.0) / 64.0  # normalize to [-2, 2]

    # Run all 5 phases without gradient tracking
    with torch.no_grad():
        for s in range(20):
            grid = step(grid, conv1)
        for s in range(20):
            grid = step(grid, conv2)
        for s in range(20):
            grid = step(grid, conv3)
        for s in range(20):
            grid = step(grid, conv4)
        for s in range(20):
            grid = step(grid, conv5)

    # Read channel 1 and de-normalize back to integer scale
    output = grid[0, 1, 0, :width]
    pred = [round(output[i].item() * 64.0 + 128.0) for i in range(width)]
    return pred


if __name__ == "__main__":

    # ==========================================
    # Test 1: 100 random width-4 cases
    # Tests overall sorting accuracy on training distribution
    # ==========================================
    print("=" * 60)
    print("TEST 1: 100 random width-4 cases")
    print("=" * 60)
    print()

    exact = 0
    within5 = 0
    for i in range(100):
        values = [random.randint(0, 255) for _ in range(4)]
        expected = sorted(values)
        got = inference(values)
        is_exact = got == expected
        is_close = all(abs(g - e) <= 5 for g, e in zip(got, expected))
        is_ordered = all(got[j] <= got[j + 1] for j in range(len(got) - 1))
        if is_exact:
            exact += 1
        if is_close and is_ordered:
            within5 += 1
        if not is_close or not is_ordered:
            print(f"  FAIL: {values} → {got} expected {expected}")

    print()
    print(f"  Exact: {exact}/100")
    print(f"  Within ±5 and ordered: {within5}/100")
    print()

    # ==========================================
    # Test 2: Close values (spacing 1-15)
    # Hardest test - values barely differ, NCA must discriminate precisely
    # ==========================================
    print("=" * 60)
    print("TEST 2: Close values (spacing 1-5)")
    print("=" * 60)
    print()

    close_cases = []
    for base in range(20, 240, 25):
        vals = [base, base + random.randint(1, 5), base + random.randint(6, 10), base + random.randint(11, 15)]
        random.shuffle(vals)
        close_cases.append(vals)

    close_correct = 0
    for values in close_cases:
        expected = sorted(values)
        got = inference(values)
        is_close = all(abs(g - e) <= 5 for g, e in zip(got, expected))
        is_ordered = all(got[j] <= got[j + 1] for j in range(len(got) - 1))
        status = "✓" if (is_close and is_ordered) else "✗"
        if is_close and is_ordered:
            close_correct += 1
        print(f"  {status} {values} → {got} expected {expected}")

    print()
    print(f"  Close accuracy: {close_correct}/{len(close_cases)}")
    print()

    # ==========================================
    # Test 3: Generalization to width 5 and 6
    # These widths were never seen during training
    # Tests whether the learned local rule transfers to longer arrays
    # ==========================================
    print("=" * 60)
    print("TEST 3: Generalization (NEVER TRAINED)")
    print("=" * 60)
    print()

    for width in [5, 6]:
        w_correct = 0
        print(f"  Width {width}:")
        for i in range(20):
            values = [random.randint(0, 255) for _ in range(width)]
            expected = sorted(values)
            got = inference(values)
            is_close = all(abs(g - e) <= 5 for g, e in zip(got, expected))
            is_ordered = all(got[j] <= got[j + 1] for j in range(len(got) - 1))
            if is_close and is_ordered:
                w_correct += 1
            else:
                print(f"    FAIL: {values} → {got} expected {expected}")
        print(f"    Accuracy (±5 and ordered): {w_correct}/20")
        print()