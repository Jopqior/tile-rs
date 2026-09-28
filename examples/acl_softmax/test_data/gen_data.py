import numpy as np

# Generate 16 random f32 values in [-2, 2) — a typical range for softmax inputs
np.random.seed(42)
input_data = np.random.uniform(-2.0, 2.0, 16).astype(np.float32)

# Compute golden reference (numerically stable softmax)
shifted = input_data - np.max(input_data)
exp_vals = np.exp(shifted)
golden = exp_vals / np.sum(exp_vals)

# Save input and length
input_data.tofile("./input.bin")
np.array([len(input_data)], dtype=np.uint32).tofile("./len.bin")
golden.tofile("./golden.bin")

print(f"input  = {input_data}")
print(f"softmax = {golden}")
print(f"sum     = {np.sum(golden)}")
