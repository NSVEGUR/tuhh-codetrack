from typing import Optional, Callable
import numpy as np

class HiddenMarkovModel:
    """
    If the observation space is discrete, `O` exists and is a list. In that case, also `psi` should be passed.
    If `O = None`, the observation space is assumed to be the real numbers, and `psi_fun` should be passed instead of `psi`.
    
    S: state space
    O: observation space (optional, if observation space is continuous)
    A: transition probability matrix
    psi: observation probability matrix
    psi_fun: observation probability function (optional, only used for continuous observation spaces)
    pi: prior distribution over states
    """
    def __init__(self, S: list, O: Optional[list], A: np.ndarray, psi: Optional[np.ndarray], psi_fun: Optional[Callable], pi: np.ndarray):
        assert A.shape == (len(S), len(S))
        if O is not None:
            assert psi is not None
            assert psi.shape == (len(S), len(O))
        assert pi.shape == (len(S),)
        self.S = S
        self.O = O
        self.A = A
        self.psi = psi
        self.psi_fun = psi_fun
        self.pi = pi

    def get_psi(self, x: np.ndarray):
        """
        In the discrete case, `x` is an array of indices of n observations.
        In the continuous case, `x` is an array of observation themselves.
        Returns a matrix, where each column represents p(z | x).
        """
        if self.O is not None:
            return self.psi[:, x]
        else:
            return self.psi_fun(x)

def make_grid_hmm(N: int, sigma: float):
    S_grid = [(i,j) for i in range(-N, N+1) for j in range(-N, N+1)]
    
    # Define the transition matrix. Due to exceptions at the edges and corners, this looks a bit messy.
    A_grid = []
    for z in S_grid:
        i,j = z
        if i in [-N,N]:
            if j in [-N,N]:
                a = [1/2 if abs(i-k) + abs(j-l) == 1 else 0 for k,l in S_grid]
            else:
                a = [1/3 if abs(i-k) + abs(j-l) == 1 else 0 for k,l in S_grid]
        elif j in [-N,N]:
            a = [1/3 if abs(i-k) + abs(j-l) == 1 else 0 for k,l in S_grid]
        else:
            a = [1/4 if abs(i-k) + abs(j-l) == 1 else 0 for k,l in S_grid]
        A_grid.append(a)
    A_grid = np.array(A_grid)
    
    # Helper function that assigns observations to their emission probabilities, for each possible state
    def psi_grid(x: float):
        return np.array([norm.pdf(x, loc=i+j, scale=sigma) for (i,j) in S_grid])
    
    # Initial distribution
    pi = np.ones(len(S_grid))
    pi = pi / pi.sum()
    
    return HiddenMarkovModel(S_grid, None, A_grid, None, psi_grid, pi)