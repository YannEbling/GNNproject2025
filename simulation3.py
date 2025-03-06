from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

import auxiliary_functions as aux
import parameters as para


def polar_eom(variables, t):
    """

    :param variables:
    :param t:
    :return:
    """
    r, rho, theta, ceta = variables

    dr_dt = rho
    drho_dt = r*ceta**2 + para.G * para.M/r**2
    dtheta_dt = ceta
    dceta_dt = -2*rho*ceta/r

    return [dr_dt, drho_dt, dtheta_dt, dceta_dt]


def solve_equation_of_motion(initial_conditions, t):
    """

    :param initial_conditions:
    :param t:
    :return:
    """

    sol = odeint(polar_eom, initial_conditions, t)
    return sol


def simulate_and_visualize(initial_conditions, t):
    sol = solve_equation_of_motion(initial_conditions, t)

    r_vals = sol[:, 0]
    rho_vals = sol[:, 1]
    theta_vals = sol[:, 2]
    ceta_vals = sol[:, 3]

    # transform the polar coordinates back to 2d euclidean space for plotting
    x_vals = r_vals * np.cos(theta_vals)
    y_vals = r_vals * np.sin(theta_vals)

    plt.figure(figsize=(6, 6))
    plt.plot(x_vals, y_vals, color="blue", marker="+", ls=":", label="Projected trajectory")
    plt.plot([0], [0], color="orange", marker="o", label="Center mass (sun)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simulated orbit using ODE solver")
    plt.legend()
    plt.grid(alpha=0.6)

    plt.show()
    plt.close()

    return x_vals, y_vals


def test():
    initial_conditions = [para.L, 0, 0, para.V_MEAN]

    t = np.linspace(0, 3600*24*365, 500)

    x, y = simulate_and_visualize(initial_conditions, t)


if __name__ == "__main__":
    test()
