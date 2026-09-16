import numpy as np

input_x = np.random.uniform(1, 100, 16).astype(np.uint16)
input_y = np.random.uniform(1, 100, 16).astype(np.uint16)
golden = (input_x + input_y).astype(np.uint16)

input_x.tofile("./input_x.bin")
input_y.tofile("./input_y.bin")
golden.tofile("./golden.bin")
