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


def sample_random_incon(r_mean, r_std, v_mean, v_std, angular_velocity_threshold):
    print("Sampling random initial condition.")
    r = np.random.normal(loc=r_mean, scale=r_std)

    x = np.random.uniform(low=-r, high=r)
    y_sign = np.sign(np.random.uniform(low=-1, high=1))
    if y_sign == 0:
        y_sign = 1
    y = y_sign * np.sqrt(r**2 - x**2)

    v = np.random.normal(loc=v_mean, scale=v_std)

    cos_condition = False
    while not cos_condition:
        v_x = np.random.uniform(low=-v, high=v)
        vy_sign = np.sign(np.random.uniform(low=-1, high=1))
        if vy_sign == 0:
            vy_sign = 1
        v_y = vy_sign * np.sqrt(v**2 - v_x**2)

        r_vector = np.array([x, y])
        v_vector = np.array([v_x, v_y])

        cos_v_r = np.dot(r_vector, v_vector) / (np.linalg.norm(r_vector, ord=2) * np.linalg.norm(v_vector, ord=2))

        if np.abs(cos_v_r) < angular_velocity_threshold:
            cos_condition = True
        else:
            v_y *= (-1)
            v_vector = np.array([v_x, v_y])

            cos_v_r = np.dot(r_vector, v_vector) / (np.linalg.norm(r_vector, ord=2) * np.linalg.norm(v_vector, ord=2))

            if np.abs(cos_v_r) < angular_velocity_threshold:
                cos_condition = True

            else:
                print("Cosine condition still not fullfilled!")

    print("Sampling of initial condition finished.")

    return [x, v_x, y, v_y]



def create_dataset(n_samples, t=None,
                   r_mean=1, r_std=0.1, v_mean=2*np.pi, v_std=0.5, angular_velocity_threshold=0.2,
                   M_G=4*np.pi**2):

    print("Starting dataset creation.")

    if t is None:
        N = 3
        T = np.sqrt(4*np.pi**2 / M_G * r_mean**3) * N  # N complete revolutions of an object in a circular orbit,
                                                       # according to Kepler 3

        t = np.linspace(0, T, 500)

    n_timepoints = t.shape[0]

    data_set = np.zeros(shape=(n_samples, 4, n_timepoints))

    for j in range(n_samples):
        print("Creating instance ", j)
        initial_condition = sample_random_incon(r_mean, r_std, v_mean, v_std, angular_velocity_threshold)

        print("Querying ODE solver for orbit simulation.")
        orbit_j = np.array(solve_equation_of_motion(initial_conditions=initial_condition, t=t))
        if orbit_j.shape[1] != data_set.shape[2]:
            j -= 1
            continue
        print("ODE solver finished.")
        data_set[j, :, :] = orbit_j

    print("Dataset completed.")
    return data_set


def visualize_dataset(data_set):
    for k in range(max(data_set.shape[0], 10)):
        print("Visualizing instance ", k)

        x_vals = data_set[k, 0, :]
        y_vals = data_set[k, 2, :]

        plt.figure(figsize=(6, 6))
        plt.plot(x_vals, y_vals, color="blue", marker="+", ls=":", label="Projected trajectory")
        plt.plot([0], [0], color="orange", marker="o", label="Center mass (sun)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Simulated orbit using ODE solver, sample "+str(k))
        plt.legend()
        plt.grid(alpha=0.6)

        plt.show()
        plt.close()


def test():
    data_set = create_dataset(n_samples=10)
    visualize_dataset(data_set)


if __name__ == "__main__":
    test()
