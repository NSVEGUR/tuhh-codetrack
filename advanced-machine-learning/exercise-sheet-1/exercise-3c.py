import numpy as np
import time

def matrix_multiply_level1(A, B):
    """
    BLAS LEVEL 1: nk scalar products (dot products)
    Computes C = A @ B using individual dot products
    C[i,j] = A[i,:] @ B[:,j] for each i,j
    
    Args:
        A: matrix of shape (n, n)
        B: matrix of shape (n, k)
    Returns:
        C: matrix of shape (n, k)
    """
    n, k = A.shape[0], B.shape[1]
    C = np.zeros((n, k))
    
    # Compute each element as a scalar product
    for i in range(n):
        for j in range(k):
            # Scalar product: A[i,:] · B[:,j]
            C[i, j] = np.dot(A[i, :], B[:, j])
    
    return C


def matrix_multiply_level2(A, B):
    """
    BLAS LEVEL 2: k matrix-vector products
    Computes C = A @ B by computing A @ b_j for each column b_j of B
    C[:,j] = A @ B[:,j] for each j
    
    Args:
        A: matrix of shape (n, n)
        B: matrix of shape (n, k)
    Returns:
        C: matrix of shape (n, k)
    """
    n, k = A.shape[0], B.shape[1]
    C = np.zeros((n, k))
    
    # Compute each column as a matrix-vector product
    for j in range(k):
        # Matrix-vector product: A @ B[:,j]
        C[:, j] = A @ B[:, j]
    
    return C


def matrix_multiply_level3(A, B):
    """
    BLAS LEVEL 3: one matrix-matrix product
    Computes C = A @ B directly using NumPy's optimized matrix multiplication
    
    Args:
        A: matrix of shape (n, n)
        B: matrix of shape (n, k)
    Returns:
        C: matrix of shape (n, k)
    """
    return A @ B


def benchmark_methods(n, k, num_trials=3):
    """
    Benchmark all three methods and compare results
    
    Args:
        n: size of square matrix A (n x n)
        k: number of columns in matrix B (n x k)
        num_trials: number of trials for timing
    """
    print(f"\n{'='*70}")
    print(f"Matrix Multiplication Benchmark: A({n}×{n}) @ B({n}×{k})")
    print(f"{'='*70}\n")
    
    # Generate random matrices
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, k)
    
    results = {}
    
    # BLAS LEVEL 1: nk scalar products
    print("BLAS LEVEL 1: Computing using nk scalar products...")
    times_l1 = []
    for trial in range(num_trials):
        start = time.time()
        C1 = matrix_multiply_level1(A, B)
        end = time.time()
        times_l1.append(end - start)
        print(f"  Trial {trial+1}: {times_l1[-1]:.4f} seconds")
    
    avg_time_l1 = np.mean(times_l1)
    results['Level 1'] = (C1, avg_time_l1)
    print(f"  Average time: {avg_time_l1:.4f} seconds\n")
    
    # BLAS LEVEL 2: k matrix-vector products
    print("BLAS LEVEL 2: Computing using k matrix-vector products...")
    times_l2 = []
    for trial in range(num_trials):
        start = time.time()
        C2 = matrix_multiply_level2(A, B)
        end = time.time()
        times_l2.append(end - start)
        print(f"  Trial {trial+1}: {times_l2[-1]:.4f} seconds")
    
    avg_time_l2 = np.mean(times_l2)
    results['Level 2'] = (C2, avg_time_l2)
    print(f"  Average time: {avg_time_l2:.4f} seconds\n")
    
    # BLAS LEVEL 3: one matrix-matrix product
    print("BLAS LEVEL 3: Computing using one matrix-matrix product...")
    times_l3 = []
    for trial in range(num_trials):
        start = time.time()
        C3 = matrix_multiply_level3(A, B)
        end = time.time()
        times_l3.append(end - start)
        print(f"  Trial {trial+1}: {times_l3[-1]:.4f} seconds")
    
    avg_time_l3 = np.mean(times_l3)
    results['Level 3'] = (C3, avg_time_l3)
    print(f"  Average time: {avg_time_l3:.4f} seconds\n")
    
    # Verify all methods produce the same result
    print(f"{'='*70}")
    print("Verification:")
    print(f"{'='*70}")
    print(f"Max difference Level 1 vs Level 3: {np.max(np.abs(C1 - C3)):.2e}")
    print(f"Max difference Level 2 vs Level 3: {np.max(np.abs(C2 - C3)):.2e}")
    
    # Performance comparison
    print(f"\n{'='*70}")
    print("Performance Summary:")
    print(f"{'='*70}")
    print(f"BLAS Level 1: {avg_time_l1:.4f} seconds (baseline)")
    print(f"BLAS Level 2: {avg_time_l2:.4f} seconds ({avg_time_l1/avg_time_l2:.1f}x faster)")
    print(f"BLAS Level 3: {avg_time_l3:.4f} seconds ({avg_time_l1/avg_time_l3:.1f}x faster)")
    print(f"\nSpeedup Level 3 vs Level 2: {avg_time_l2/avg_time_l3:.1f}x")
    print(f"Speedup Level 3 vs Level 1: {avg_time_l1/avg_time_l3:.1f}x")
    
    # Compute theoretical operations
    ops_level1 = n * k * (2 * n - 1)  # nk dot products, each with n multiplications and n-1 additions
    ops_level2 = k * (2 * n * n - n)  # k matrix-vector products
    ops_level3 = 2 * n * n * k - n * k  # matrix-matrix product
    
    print(f"\n{'='*70}")
    print("Theoretical Analysis:")
    print(f"{'='*70}")
    print(f"Number of scalar products (Level 1): {n * k:,}")
    print(f"Number of matrix-vector products (Level 2): {k:,}")
    print(f"Number of matrix-matrix products (Level 3): 1")
    print(f"\nTotal operations are the same: ~{2*n*n*k:,} FLOPs")
    print(f"But memory access patterns and cache efficiency differ dramatically!")


if __name__ == "__main__":
    # Test with smaller matrices first
    print("Testing with small matrices (n=100, k=10):")
    benchmark_methods(n=100, k=10, num_trials=3)
    
    # Test with medium-sized matrices
    print("\n\nTesting with medium matrices (n=1000, k=50):")
    benchmark_methods(n=1000, k=50, num_trials=3)
    
    # Test with larger matrices (adjust based on your computer)
    # For n=10000, k=100, Level 1 might be very slow!
    print("\n\nTesting with larger matrices (n=2000, k=100):")
    print("(Note: n=10000 would take very long for Level 1, using n=2000 instead)")
    benchmark_methods(n=2000, k=100, num_trials=3)
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. BLAS Level 1: Slowest - poor cache utilization, element-by-element")
    print("2. BLAS Level 2: Faster - better cache usage with vector operations")
    print("3. BLAS Level 3: Fastest - optimal cache blocking and vectorization")
    print("\nThe speedup demonstrates why modern libraries use BLAS Level 3!")
    print("="*70)