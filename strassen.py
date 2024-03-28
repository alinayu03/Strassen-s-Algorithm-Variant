import numpy as np

def read_matrix(filename, dimension):
    """Read matrices from a file."""
    data = np.loadtxt(filename, dtype=int)
    A = data[:dimension**2].reshape(dimension, dimension)
    B = data[dimension**2:].reshape(dimension, dimension)
    return A, B

def standard_matrix_multiplication(A, B):
    """Perform standard matrix multiplication."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def add_matrices(A, B):
    """Add two matrices."""
    return A + B

def subtract_matrices(A, B):
    """Subtract matrix B from A."""
    return A - B

def split_matrix(A):
    """Split matrix into quadrants."""
    row, col = A.shape
    row2, col2 = row // 2, col // 2
    return A[:row2, :col2], A[:row2, col2:], A[row2:, :col2], A[row2:, col2:]

def strassen(A, B, n0):
    """Strassen's algorithm with crossover to the standard algorithm."""
    n = A.shape[0]
    if n <= n0:
        return standard_matrix_multiplication(A, B)
    if n % 2 != 0:
        A = np.pad(A, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        B = np.pad(B, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        n += 1
    
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    P1 = strassen(add_matrices(A11, A22), add_matrices(B11, B22), n0)
    P2 = strassen(add_matrices(A21, A22), B11, n0)
    P3 = strassen(A11, subtract_matrices(B12, B22), n0)
    P4 = strassen(A22, subtract_matrices(B21, B11), n0)
    P5 = strassen(add_matrices(A11, A12), B22, n0)
    P6 = strassen(subtract_matrices(A21, A11), add_matrices(B11, B12), n0)
    P7 = strassen(subtract_matrices(A12, A22), add_matrices(B21, B22), n0)

    C11 = add_matrices(subtract_matrices(add_matrices(P1, P4), P5), P7)
    C12 = add_matrices(P3, P5)
    C21 = add_matrices(P2, P4)
    C22 = add_matrices(subtract_matrices(add_matrices(P1, P3), P2), P6)

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    if C.shape[0] != n:
        C = C[:n, :n]
    return C[:dimension, :dimension]

def print_diagonal(C):
    """Print the diagonal of the matrix."""
    for i in range(C.shape[0]):
        print(C[i, i])

if __name__ == "__main__":
    import sys
    flag, dimension, inputfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    A, B = read_matrix(inputfile, dimension)
    n0 = 15 if dimension % 2 == 0 else 37
    C = strassen(A, B, n0)
    print_diagonal(C)
