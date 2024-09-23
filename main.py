import numpy as np
import sys
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

print('\n\nSTART ---------------------\n')

import numpy as np
import time

def invert_matrix(n):
    A = np.random.rand(n, n)
    start_time = time.time()
    A_inv = np.linalg.inv(A)
    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds

sizes = [10, 100, 1000, 10000, 25000]
for size in sizes:
    inversion_time = invert_matrix(size)
    print(f"Inverting a {size}x{size} matrix took {inversion_time:.4f} ms")
