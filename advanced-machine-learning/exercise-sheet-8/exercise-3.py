# Direct full convolution
# Multiplications = (n+k-1) * k 
# Additions = (n+k-1)*(k-1)

# FFT-based convolution
# Multiplications = 3 * l * 2^l + 2^l
# Additions = 3 * l * 2^l


import numpy as np
import matplotlib.pyplot as plt

def direct_ops(n, k):
    return (n+k-1) * (2*k - 1)

def fft_ops(n, k):
    l = (int(n + k - 1) - 1).bit_length()
    return 3 * 2**l * l + 3 * 2**l * l + 2**l

# parameters
n = 2000
k_values = np.arange(1, 2001)

ops_direct = [direct_ops(n, k) for k in k_values]
ops_fft = [fft_ops(n, k) for k in k_values]

plt.figure(figsize=(8,5))
plt.plot(k_values, ops_direct, label="Direct Convolution")
plt.plot(k_values, ops_fft, label="FFT-based Convolution")
plt.xlabel("k")
plt.ylabel("Operation Count")
plt.title(f"Operation Count Comparison for n={n}")
plt.legend()
plt.grid(True)
plt.show()
