import numpy as np


class Cube:
    def __init__(self, center: np.ndarray, length: float, particles: np.ndarray, parent=None):
        self.parent = parent
        self.center = center
        self.length = length
        self.children = []
        self.total_mass = None
        self.center_of_mass = None
        self.branch = False
        self.leaf = False
        self.finished = False

        hl = length / 2.0
        mask = np.array([
            (particles[j] >= (center[i] - hl)) & (particles[j] < (center[i] + hl)) for i, j in enumerate("xyz")])
        mask = mask.all(axis=0)

        n_particles = np.sum(mask)
        if n_particles >= 1:
            self.particles = particles[mask]
            self.mass = self.particles["mass"]
            self.positions = np.column_stack([self.particles["x"], self.particles["y"], self.particles["z"]])
            self.evaluate_center_of_mass()
            if n_particles == 1:
                self.leaf = True
                self.finished = True
            else:
                self.branch = True
        elif n_particles == 0:
            self.particles = None
            self.leaf = True
            self.finished = True
        else:
            raise RuntimeError("Dunno?")

    def evaluate_center_of_mass(self):
        x_cm = []
        total_mass = np.sum(self.mass)
        for i in range(3):
            x_cm.append(np.sum(self.positions[:, i] * self.mass) / total_mass)

        self.center_of_mass = np.array(x_cm)


class Tree:

    def __init__(self, particles: np.ndarray):
        self.particles = particles
        self.positions = np.column_stack([particles["x"], particles["y"], particles["z"]])
        self.cubes = []

    @staticmethod
    def subdivided_centers(center, d):
        """
        Generates 8 new points for the centers of subdivided cubes.

        Parameters
        ----------
        center : np.ndarray
            Coordinates of the center of the current cube.
            np.array([x, y, z])
        d : float
            Length of the current cube.

        Returns
        -------

        """
        b = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                      [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]],
                     dtype=float)
        # np.sqrt(2) / 4 = 0.3535533905932738
        b *= d * 0.3535533905932738
        b += center
        return b

    def make_tree(self, verbose: bool = False):
        max_r = np.max(np.linalg.norm(self.positions, axis=1))
        c = self.subdivided_centers(np.zeros((3,)), 2 * max_r)
        self.cubes = [Cube(center=_, length=max_r, particles=self.particles, parent="trunk") for _ in c]

        def recurse(cube, parent):
            new_c = self.subdivided_centers(center=cube.center, d=cube.length)
            cube.children = [
                Cube(center=_, length=cube.length / 2, particles=self.particles, parent=parent) for _ in new_c]
            if all(_.leaf for _ in cube.children):
                cube.finished = True
            else:
                for i in [j for j in cube.children if j.branch]:
                    recurse(i, cube)

        for a in self.cubes:
            recurse(a, "trunk")
