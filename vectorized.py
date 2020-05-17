import numpy as np

from utils import timing


@timing
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = np.zeros([A.shape[0], B.shape[1]])
    for k in range(0, B.shape[1]):
        for i in range(0, A.shape[0]):
            temp = 0
            for j in range(0, A.shape[1]):
                temp += A[i, j] * B[j, k]
            C[i, k] = temp
    return C

@timing
def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = np.zeros([A.shape[0], B.shape[1]])
    for k in range(0, B.shape[1]):
        for i in range(0, A.shape[0]):
            C[i, k] = np.dot(A[i, :], B[:, k])
    return C


def main():
    epsilon = 1e-7
    A = np.random.randn(4, 5)
    B = np.random.randn(5, 8)

    C = matmul(A, B)
    C_vec = matmul_vectorized(A, B)
    C_check = np.matmul(A, B)

    assert np.all(np.linalg.norm(C - C_check) < epsilon)
    assert np.all(np.linalg.norm(C_vec - C_check) < epsilon)



if __name__ == '__main__':
    main()
