import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from ulm import initial_conditions, tree_generator


def main():
    pos = initial_conditions.generate_volume(n=4)
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
    ax = fig.add_subplot()

    def recurse(d, axes):
        if d.branch:
            for e in d.children:
                rect = patches.Rectangle((d.center[0] - (d.length / 2), d.center[1] - (d.length / 2)),
                                         width=d.length, height=d.length, alpha=0.1)
                axes.add_patch(rect)
                recurse(e, axes)

    for a in t.cubes:
        recurse(a, ax)

    ax.scatter(particles["x"], particles["y"])
    ax.set_aspect("equal")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    plt.show()

    return t


if __name__ == '__main__':
    x = main()
