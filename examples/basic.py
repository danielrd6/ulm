import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from ulm import initial_conditions, tree_generator


def draw_rectangle_3d(center, length, axes, **kwargs):
    for i in [-1, 1]:
        b = np.array([[1, 1, i], [-1, 1, i], [-1, -1, i], [1, -1, i], [1, 1, i]], dtype=float)
        v = (b * length) + center
        axes.plot(v[:, 0], v[:, 1], v[:, 2], **kwargs)

    for i in [-1, 1]:
        for j in [-1, 1]:
            v = np.array([[i, j, 1], [i, j, -1]]) * length + center
            axes.plot(v[:, 0], v[:, 1], v[:, 2], **kwargs)


def main(n_particles : int = 4, plot: bool = True, projection="2d"):
    if projection == "3d":
        pos = initial_conditions.generate_volume(n=n_particles, make_3d=True)
    if projection == "2d":
        pos = initial_conditions.generate_volume(n=n_particles, make_3d=False)
    else:
        raise RuntimeError
    # r = []
    # for i in range(pos.shape[0]):
    #     for j in range(pos.shape[0]):
    #         if i != j:
    #             r.append(pos[i] - pos[j])
    # r = np.linalg.norm(r, axis=1)
    # print(f"min distance = {r.min()}")

    mass = np.ones(pos.shape[0])
    particle_id = np.arange(pos.shape[0])

    v = [(particle_id[i], mass[i], pos[i, 0], pos[i, 1], pos[i, 2]) for i in range(pos.shape[0])]
    particles = np.array(v, dtype=[("id", int), ("mass", float), ("x", float), ("y", float), ("z", float)])
    t = tree_generator.Tree(particles=particles)
    t.make_tree(verbose=True)

    fig = plt.figure()
    if projection == "3d":
        ax = fig.add_subplot(projection=projection)
    elif projection == "2d":
        ax = fig.add_subplot()
    else:
        raise RuntimeError

    def draw_boundaries(d, axes, proj):
        if d.branch:
            for e in d.children:
                if projection == "2d":
                    rect = patches.Rectangle((e.center[0] - (e.length / 2), e.center[1] - (e.length / 2)),
                                             width=e.length, height=e.length, alpha=0.1, facecolor="none",
                                             edgecolor="black")
                    axes.add_patch(rect)
                elif projection == "3d":
                    draw_rectangle_3d(d.center, d.length, axes, color="black", lw=1, alpha=0.1)
                else:
                    raise RuntimeError
                draw_boundaries(e, axes, proj)

    for a in t.cubes:
        draw_boundaries(a, ax, proj=projection)

    if projection == "2d":
        ax.scatter(particles["x"], particles["y"], marker=".")
    elif projection == "3d":
        ax.scatter(particles["x"], particles["y"], particles["z"], marker=".")

    plt.show()

    return t


if __name__ == '__main__':
    x = main(n_particles=3, projection="2d")
