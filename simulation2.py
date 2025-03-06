import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

from parameters import L, V_MEAN, M, m, G
import auxiliary_functions as aux


class TwoBodySimulation:

    def __init__(self, r_0=None, v_0=None, l=L, v_mean=V_MEAN, unit_std=0.1, mass_1=M, mass_2=m):
        print("SIM 2 CONSTRUCTOR!")
        if r_0 is None:
            self.r0 = l * (unit_std * np.random.randn() + 1)
        else:
            self.r0 = r_0

        if v_0 is None:
            self.v_0 = v_mean * np.random.randn(2)
        else:
            self.v_0 = v_0

        self.center_m = mass_1
        self.trabant_m = mass_2

        self.h, self.specific_energy, self.excentricity, self.p, self.periapsis, self.a = self.obtain_parameters()

    def obtain_parameters(self):
        h = self.r0 * abs(self.v_0[1])
        specific_energy = (self.v_0[0]**2 + self.v_0[1]**2)/2 - G * self.center_m / self.r0
        excentricity = np.sqrt(1 + 2 * specific_energy * h ** 2 / ((G * self.center_m)**2))
        p = h**2 / (G * self.center_m)
        periapsis = np.arccos((p / self.r0 - 1) / excentricity)
        a = p / (1 + excentricity)**2

        print("H: ", h, "\nspec_energy: ", specific_energy, "\nexcentricity: ", excentricity, "\np: ", p,
              "\nperiapsis: ", periapsis, "\na: ", a)

        return h, specific_energy, excentricity, p, periapsis, a

    def simulate(self, t, output_type="plane"):
        """
                Simulate the position of the trabant at the times specified by t.
                :param t: np.ndarray. Contains time values to which we want to simulate the position of the trabant.
                :param output_type: string. Valid options are "polar", "plane" and "euclidean". Specifies the representation
                in which the simulated data are returned.
                :return: data array containing for each time step the coordinates of the trabant.
                """
        n = np.sqrt(G * self.center_m / self.a ** 3)

        M_vals = n * t  # mean anomaly
        E_vals = np.array([aux.solve_kepler(M, self.excentricity) for M in M_vals])  # excentric anomaly

        theta_vals = 2 * np.arctan(np.sqrt((1 + self.excentricity) / (1 - self.excentricity)) * np.tan(E_vals / 2))
        # real anomaly (values for theta)

        r_vals = self.p / (1 + self.excentricity * np.cos(theta_vals - self.periapsis))  # radial values from trajectory
        # description r(theta).

        # return polar coordinate representation if desired
        if output_type == "polar":
            return np.stack(r_vals, theta_vals, axis=1)

        print(r_vals)
        print(theta_vals)

        # back transformation to plane coordinates
        plane_coordinates = np.zeros((t.shape[0], 2))
        #r_vals = np.ones_like(r_vals)
        plane_coordinates[0, 0] = self.r0
        for k in range(1, t.shape[0]):
            plane_coordinates[k, 0] = r_vals[k] * np.cos(theta_vals[k])
            plane_coordinates[k, 1] = r_vals[k] * np.sin(theta_vals[k])

        return plane_coordinates

    def simulate_and_visualize(self, t):
        plane_coordinates = self.simulate(t, output_type="plane")

        # plotting the obtained data
        plt.figure(figsize=(6, 6))
        plt.plot(plane_coordinates[:, 1], plane_coordinates[:, 0], label='Trabant trajectory', ls=":", marker="+")
        plt.scatter([0], [0], color='orange', label='Central object')
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Simulation of trabant trajectory")
        plt.legend()
        # plt.text("Initial conditions: r0 = " + str(round(self.r0, 2)) + " v0 = " + str(self.trabant.v))
        # plt.axis('equal')
        plt.grid()
        plt.show()
        plt.close()

        return plane_coordinates