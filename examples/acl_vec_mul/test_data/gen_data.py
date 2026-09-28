import numpy as np

# Generate 16 random u16 values in [1, 10) to avoid overflow on multiply
input_x = np.random.uniform(1, 10, 16).astype(np.uint16)
input_y = np.random.uniform(1, 10, 16).astype(np.uint16)
golden = (input_x * input_y).astype(np.uint16)

input_x.tofile("./input_x.bin")
input_y.tofile("./input_y.bin")
golden.tofile("./golden.bin")

print(f"x = {input_x}")
print(f"y = {input_y}")
print(f"x * y = {golden}")
