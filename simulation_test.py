import numpy as np
import scipy
import matplotlib.pyplot as plt

import simulation2 as sim
from parameters import G, M, m, L, V_MEAN


def simulation_test():
    planet_simulator = sim.TwoBodySimulation(mass_1=m, mass_2=M, v_mean=V_MEAN, v_0=np.array([0, V_MEAN]), r_0=L)

    t_max = 2 * np.pi * np.sqrt(planet_simulator.a ** 3 / (G * M))  # Umlaufzeit nach Kepler 3

    t = 100*np.linspace(0, t_max, 1000)

    planet_simulator.simulate_and_visualize(t)
    return


def main():
    simulation_test()


if __name__ == "__main__":
    main()
