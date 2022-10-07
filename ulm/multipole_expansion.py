import numpy as np
import tqdm
from scipy.special import legendre

from ulm import initial_conditions


def get_potential(r, ca, charge, order: int = 5, softening_length: float = 0.1):
    v = 0
    for i in range(order):
        a = ((r + softening_length) ** i) * legendre(i)(ca) * charge
        v += np.sum(1 / ((r + softening_length) ** (i + 1)) * a)
    return v


def main(steps=3):
    pos = initial_conditions.generate_volume(5, 10)
    charge = np.random.choice([-1, 1], pos.shape[0])
    potential = np.zeros_like(charge)
    n_particles = pos.shape[0]
    particle_id = np.arange(n_particles)

    for j in range(steps):
        print(f"step: {j:03d}")
        for i in tqdm.tqdm(range(pos.shape[0])):
            # módulo de r.
            r = np.linalg.norm(pos, axis=1)
            # ângulo entre r e r linha.
            cos_alpha = np.dot(pos, pos[i]) / (r * np.linalg.norm(pos[i]))
            mask = particle_id == i
            potential[i] = get_potential(r[~mask], cos_alpha[~mask], charge[~mask], order=10, softening_length=10)

        snap = np.column_stack([particle_id, pos, charge, potential])
        np.save(f"snap_{j:03d}.npy", snap)


if __name__ == '__main__':
    main(steps=1)
