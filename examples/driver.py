import numpy as np

from src.lab02.linalg_interp import (
    gauss_iter_solve,
)

water_density = np.loadtxt("../data/water_density_vs_temp_usgs.txt", "float")
air_density = np.loadtxt("../data/air_density_vs_temp_eng_toolbox.txt", "float")

print(water_density)
print(air_density)

def main():
    gauss_iter_solve(1, 2, 3,4,5)



if __name__ == "__main__":
    main()