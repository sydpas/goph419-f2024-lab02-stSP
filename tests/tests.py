import numpy as np

from src.lab02.linalg_interp import (
    gauss_iter_solve,
    spline_function
)

w_d = np.loadtxt("../data/water_density_vs_temp_usgs.txt", "float")
a_d = np.loadtxt("../data/air_density_vs_temp_eng_toolbox.txt", "float")

xd = np.array([1,2,3,4])
yd = np.array([1,4,5,7])

def main():

    x = spline_function(xd, yd, 1)
    print(f'Linear splines: {x}')


if __name__ == "__main__":
    main()