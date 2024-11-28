import numpy as np

from src.lab02.linalg_interp import (
    gauss_iter_solve,
    spline_function
)

w_d = np.loadtxt("../data/water_density_vs_temp_usgs.txt", "float")
a_d = np.loadtxt("../data/air_density_vs_temp_eng_toolbox.txt", "float")

xd1, yd1 = w_d[:, 0], w_d[:, 1]
xd2, yd2 = a_d[:, 0], a_d[:, 1]

def main():
    print(f'Using water density data...')

    x_L_w = spline_function(xd1, yd1, 1)
    x_L_w_rounded = [round(val, 4) for val in x_L_w]
    print(f'Linear: {x_L_w_rounded}')

    x_Q_w = spline_function(xd1, yd1, 2)
    x_Q_w_rounded = [round(val, 4) for val in x_Q_w]
    print(f'Quadratic: {x_Q_w_rounded}')

    print(f'Using air density data...')

    x_L_a = spline_function(xd2, yd2, 1)
    x_L_a_rounded = [round(val, 4) for val in x_L_a]
    print(f'Linear: {x_L_a_rounded}')

    x_Q_a = spline_function(xd2, yd2, 2)
    x_Q_a_rounded = [round(val, 4) for val in x_Q_a]
    print(f'Quadratic: {x_Q_a_rounded}')


if __name__ == "__main__":
    main()