import numpy as np
from src.lab02.linalg_interp import (
    gauss_iter_solve,
    spline_function,
)

w_d = np.loadtxt("../data/water_density_vs_temp_usgs.txt", "float")
a_d = np.loadtxt("../data/air_density_vs_temp_eng_toolbox.txt", "float")

xd1, yd1 = w_d[:, 0], w_d[:, 1]
xd2, yd2 = a_d[:, 0], a_d[:, 1]


def main():
    A = np.array([[10, -2, 0, 0],
            [-10, 4, -1, 0],
            [0, -1, 9, -1],
            [0, 0, -1, 9]])
    b = np.array([15, 10, 20, 10])
    x0 = np.array([0, 0, 0, 0])

    print(f'A: {A}')
    print(f'b: {b}')
    print(f'x0: {x0}')

    x_s = gauss_iter_solve(A, b, x0, 1e-8,'seidel')
    print(f'X: {x_s}')

    x_j = gauss_iter_solve(A, b, x0, 1e-8,'jacobi')
    print(f'X: {x_j}')

    print(f'Using water density data...')

    x_L_w = spline_function(xd1, yd1, 1)
    print(f'Linear: {x_L_w}')

    x_Q_w = spline_function(xd1, yd1, 2)
    print(f'Quadratic: {x_Q_w}')

    x_C_w = spline_function(xd1, yd1, 3)
    print(f'Cubic: {x_C_w}')

    print(f'Using air density data...')

    x_L_a = spline_function(xd2, yd2, 1)
    print(f'Linear: {x_L_a}')

    x_Q_a = spline_function(xd2, yd2, 2)
    print(f'Quadratic: {x_Q_a}')

    x_C_a = spline_function(xd2, yd2, 3)
    print(f'Cubic: {x_C_a}')



if __name__ == "__main__":
    main()