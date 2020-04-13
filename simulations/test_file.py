import simulation_library
import potentials
import integrators
import data_logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import visuals
import time

system_builder = simulation_library.SystemFactory()
system_3 = system_builder.build_system(dim = 2, T = 0.6, rho = 1.6, N = 1200)
system_3.get_integrator("verlet_cell_list", 0.001, r_cell = 1.3)
coords_logger_3 = data_logging.CoordinateLogger(system_3, 50)
energy_logger_3 = data_logging.EnergyLogger(system_3, 500)
pressure_logger_3 = data_logging.PressureLogger(system_3, 500)
system_3.registerObserver(coords_logger_3)
system_3.registerObserver(energy_logger_3)
system_3.registerObserver(pressure_logger_3)

t1 = time.time()
system_3.run(100)
t2 = time.time()
print("Verlet Neighbors:", t2 - t1, "seconds")
