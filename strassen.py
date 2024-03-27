def read_matrix(filename, d):
    with open(filename, 'r') as f:
        numbers = [int(line.strip()) for line in f]
    A = [numbers[i:i+d] for i in range(0, d**2, d)]
    B = [numbers[i:i+d] for i in range(d**2, 2*d**2, d)]
    return A, B

def standard_matrix_multiplication(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def add_matrices(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def subtract_matrices(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def get_matrix_quadrants(matrix):
    n = len(matrix)
    mid = n // 2
    A = [[matrix[i][j] for j in range(mid)] for i in range(mid)]
    B = [[matrix[i][j] for j in range(mid, n)] for i in range(mid)]
    C = [[matrix[i][j] for j in range(mid)] for i in range(mid, n)]
    D = [[matrix[i][j] for j in range(mid, n)] for i in range(mid, n)]
    return A, B, C, D

def strassen(A, B, n0):
    n = len(A)
    if n <= n0:
        return standard_matrix_multiplication(A, B)
    A11, A12, A21, A22 = get_matrix_quadrants(A)
    B11, B12, B21, B22 = get_matrix_quadrants(B)
    
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

    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n // 2):
        for j in range(n // 2):
            C[i][j] = C11[i][j]
            C[i][j + n // 2] = C12[i][j]
            C[i + n // 2][j] = C21[i][j]
            C[i + n // 2][j + n // 2] = C22[i][j]
    return C

def main(flag, dimension, inputfile):
    A, B = read_matrix(inputfile, dimension)
    n0 = 15 if dimension % 2 == 0 else 37  
    C = strassen(A, B, n0)
    for i in range(dimension):
        print(C[i][i])

if __name__ == "__main__":
    import sys
    flag, dimension, inputfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    main(flag, int(dimension), inputfile)

