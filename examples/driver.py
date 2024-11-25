import numpy as np
from src.lab02.linalg_interp import (
    gauss_iter_solve,
)

water_density = np.loadtxt("../data/water_density_vs_temp_usgs.txt", "float")
air_density = np.loadtxt("../data/air_density_vs_temp_eng_toolbox.txt", "float")

#print(water_density)
#print(air_density)

def main():
    # A = np.random.rand(3, 3)
    A = np.array([[4, 0, 0],
            [2, 3, 0],
            [0, 4, 3]])
    # b = np.random.rand(3, 1)
    b = np.array([1, 4, 3])
    # x0 = np.random.rand(3, 1)
    x0 = np.array([1, 5, 6])

    print(f'A: {A}')
    print(f'b: {b}')
    print(f'x0: {x0}')

    x = gauss_iter_solve(A, b, x0, 1e-8, None)

    print(f'X: {x}')


if __name__ == "__main__":
    main()