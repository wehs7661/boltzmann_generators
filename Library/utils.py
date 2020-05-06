import numpy as np
from scipy.optimize import linear_sum_assignment

def apply_hungarian_alg(ref_coord, coords):
    new_coords = []
    for coord in coords:
        cost_matrix = np.subtract.outer(ref_coord[:2], coord[:2].T)
        cost_matrix = np.array([cost_matrix[:,0,0,:], cost_matrix[:,1,1,:]])
        cost_matrix = np.linalg.norm(cost_matrix, axis = 0)
        old_index, new_index = linear_sum_assignment(cost_matrix)
        coord[new_index + 2] = coord[old_index + 2]
        new_coords.append(coord)
    
    return np.array(new_coords)