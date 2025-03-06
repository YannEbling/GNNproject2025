import numpy as np
import scipy
import matplotlib.pyplot as plt


def twod_basis_transform(x: np.array, basis_vector_tuple: tuple[np.ndarray, np.ndarray]) -> np.array:
    """
    Function that computes a basis representation of a vector in a two-d subspace spanned by orthogonal basis vectors.

    :param x: Vector (ndarray), that shall be represented in a 2d basis. Make sure, that x lies in the span of the basis
    vectors.
    :param basis_vector_tuple: The basis (tuple of ndarrays), in which x shall be represented. Has to be orthogonal and
    non-zero, but not normalized. Should span a space which contains x.
    :return x_2d: A two dimensional vector (ndarray), which is the representation of x in the chosen basis.
    """

    basis_1 = basis_vector_tuple[0]
    basis_2 = basis_vector_tuple[1]

    # Check for orthogonality and non_zero:
    assert basis_1 @ basis_2 < 1e-10
    assert np.linalg.norm(basis_1, ord=2) != 0
    assert np.linalg.norm(basis_2, ord=2) != 0

    # normalize the basis vectors
    basis_1 /= np.linalg.norm(basis_1, ord=2)
    basis_2 /= np.linalg.norm(basis_2, ord=2)

    # compute basis decomposition
    x_2d = np.array([x @ basis_1, x @ basis_2])
    return x_2d


def threed_basis_transform(x: np.ndarray, basis_vector_tuple: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Function that transforms a 2d plane representation back to standard euclidean representation, using the two basis
    vectors b1, b2 via the formula x_3d = x_2d_1 * b1 + x_2d_2 * b2.
    :param x: The two dimensional representation of the vector
    :param basis_vector_tuple: Tuple of 3dimensional basis vectors, in which the vector x is expressed
    :return: The three dimensional representation of x in the original basis.
    """
    assert np.shape(x)[0] == len(basis_vector_tuple)

    x_3d = x[0] * basis_vector_tuple[0] + x[1] * basis_vector_tuple[1]

    return x_3d


def kepler_equation(E, M, e):
    """
    Auxiliary function which has a zero at the solution of the Kepler equation  M = E - e * sin(E)
    :param E: excentric anomaly
    :param M: mean anomaly
    :param e: excentricity
    :return: E - e*sin(E) - M
    """
    return E - e * np.sin(E) - M


def solve_kepler(M, e):
    """
    Solve Kepler equation for E using the Newton method.

    :param M: mean anomaly
    :param e: excentricity
    :return: approximate solution of the Kepler equation, using Newton method
    """
    E0 = M  # initial value: M as an appoximation valid for small e
    return scipy.optimize.newton(kepler_equation, E0, args=(M, e))
