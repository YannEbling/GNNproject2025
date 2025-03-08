from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def Newton_eom(t, variables):
    """
    Newtonian equation of motion in first order representation (in astronomical units, AE and year).
    :param variables: tuple of form (x, vx, y, vy)
    :param t: time parameter, gravitational law is independent of t. Used as integration variable by ODE solver
    :return: derivative of input variables according to equation of motion
    """


    G_M = 4 * np.pi**2

    x, vx, y, vy = variables

    r = np.sqrt(x**2 + y**2)
    ax = - G_M * x / r**3
    ay = - G_M * y / r**3

    return [vx, ax, vy, ay]


def solve_equation_of_motion(initial_conditions, t):
    """
    Function that calls the ODE solver and evaluates the solution at the points contained in t.
    :param initial_conditions: Initial condition for which the initial value problem shall be solved.
    :param t: t values where the solution shall be recorded. Evaluation points
    :return: array of shape (4, len(t)), containing the recorded solutions for x, vx, y and vy
    """

    T = np.max(t)

    sol = solve_ivp(Newton_eom, [0, T], initial_conditions, method="Radau", t_eval=t, dense_output=True)

    sol = sol.y

    return sol[0, :], sol[1, :], sol[2, :], sol[3, :]


def simulate_and_visualize(initial_conditions, t):
    x_vals, vx_vals, y_vals, vy_vals = solve_equation_of_motion(initial_conditions, t)

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
    initial_conditions = [1, 0, 0, 2*np.pi]

    T = 10
    t = np.linspace(0, T, 500)

    x, y = simulate_and_visualize(initial_conditions, t)


if __name__ == "__main__":
    test()
