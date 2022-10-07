import numpy as np


def generate_volume(n: int = 32, length: float = 10.0, make_3d: bool = True):
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

    x = np.random.random(n**3)
    y = np.random.random(n**3)

    if make_3d:
        z = np.random.random(n**3)
    else:
        z = np.zeros_like(x)

    pos = (np.column_stack([x, y, z]) - 0.5) * length

    return pos
