import numpy as np

water_density = np.loadtxt("../data/water_density_vs_temp_usgs.txt", "float")
air_density = np.loadtxt("../data/air_density_vs_temp_eng_toolbox.txt", "float")

print(water_density)
print(air_density)
