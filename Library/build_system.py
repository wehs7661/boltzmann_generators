import numpy as np
import itertools


def build_dimer_coords(N = 38, box = 6, noise = 0.1, bond = [17, 24]):
    r_min = -box /2
    r_max = -r_min
    d = box / N ** (1/2)
    pos = np.linspace(r_min + 0.5 * d, r_max - 0.5 * d, np.ceil(N**(1/2)))
    coords = list(itertools.product(pos, repeat=2))
    coords = coords[:N]
    coords.insert(0, coords.pop(bond[0]))
    coords.insert(0, coords.pop(bond[1]))
    coords = np.array(coords)
    if noise:
        noise = noise * np.random.randn(*coords.shape)
        coords += noise
    return coords