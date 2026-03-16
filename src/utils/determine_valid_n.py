import numpy as np


def interfaces_are_face_aligned(a: float, b: float, N: int, interfaces: list[float], tol: float = 1e-12):
    L = b - a
    M = N - 1
    for x in interfaces:
        val = 2.0 * M * (x - a) / L
        nearest = round(val)
        # testing if val is an integer
        if not np.isclose(val, nearest, atol=tol, rtol=0.0):
            return False
        # must be an odd integer
        if nearest % 2 == 0:
            return False
    return True


def find_valid_N(a: float, b: float, interfaces: list[float], N_min: int = 3, N_max: int = 1000, tol: float = 1e-12):
    valid_N = []
    for N in range(N_min, N_max + 1):
        if interfaces_are_face_aligned(a=a, b=b, interfaces=interfaces, N=N, tol=tol):
            valid_N.append(N)
    return valid_N


if __name__ == '__main__':
    valid_N = find_valid_N(a=0.0, b=1.0, interfaces=[0.5])

    print(valid_N)