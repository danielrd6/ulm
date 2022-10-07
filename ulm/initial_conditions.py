import numpy as np


def generate_volume(n: int = 32, length: float = 10.0):
    """
    Creates a cubic volume of evenly distributed random particles.

    Parameters
    ----------
    n : int
        Number of particles on each axis.
    length : float
        Side of the cube.

    Returns
    -------
    pos : np.ndarray
        Positions of the particles
    """

    pos = np.random.random((n ** 3) * 3).reshape((n ** 3, 3))
    pos = (pos - 0.5) * length
    return pos
