import numpy as np

class ReLU:
    def __init__(self):
        """Initialize the ReLU activation function (no parameters)."""
        pass

    def __call__(self, x):
        """Apply the ReLU function element-wise on a NumPy array or scalar."""
        return np.maximum(0, x)
