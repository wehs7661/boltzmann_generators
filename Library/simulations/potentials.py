import numpy as np
import torch

# Particle Potentials
class WCAPotential:
    def __init__(self, sigma, epsilon):
        self.r_c = np.power(2, 1/6) * sigma 
        self.sigma = sigma
        self.epsilon = epsilon
        self.E_c = self.epsilon * 4 * ( (self.r_c / self.sigma) ** -12 - ( self.r_c / self.sigma) ** -6 )

    def __call__(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2:
            return(self.epsilon * 4 * ( (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) - self.E_c )
        else:
            return(0.0)
    def derivative(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2:
            return( 24 * self.epsilon * r_ij / np.dot(r_ij, r_ij) * (2 * (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) )
        else:
            return(np.zeros(r_ij.shape))

class LJpotential:
    def __init__(self, sigma, epsilon, r_c = None):
        self.r_c = r_c
        self.sigma = sigma
        self.epsilon = epsilon
        self.E_c = self.epsilon * 4 * ( (self.r_c / self.sigma) ** -12 - ( self.r_c / self.sigma) ** -6 )

    def __call__(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return(self.epsilon * 4 * ( (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ))
        else:
            return(0.0)
    def derivative(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return( 24 * self.epsilon * r_ij / np.dot(r_ij, r_ij) * (2 * (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) )
        else:
            return(np.zeros(r_ij.shape))

class LJRepulsion:
    def __init__(self, sigma, epsilon, r_c = None):
        self.r_c = np.power(2, 1/6) * sigma 
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return(self.epsilon * 4 * ((self.sigma**2 / np.dot(r_ij, r_ij)**6) ))
        else:
            return(0.0)
    def derivative(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return( 24 * self.epsilon * r_ij / np.dot(r_ij, r_ij) * (2 * (self.sigma**2 / np.dot(r_ij, r_ij)**6) ))
        else:
            return(np.zeros(r_ij.shape))

# 1D Potentials (for Internal Coordinates)
class DoubleWellPotential1D:
    def __init__(self, d0, a, b, c):
        self.d0 = d0
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, d):
        return 1/4 * self.a * (d - self.d0) ** 4 - 1/2 * self.b * (d - self.d0) ** 2 + \
            self.c * (d - self.d0)

    def derivative(self, d):
        return self.a * (d - self.d0) ** 3 - self.b * (d - self.d0) + self.c


# Central Potentials
class DoubleWellPotential:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def __call__(self, r):
        return 1/4 * self.a * r[0] ** 4 - 1/2 * self.b * r[0] ** 2 + \
            self.c * r[0] + 1/2 * self.d * r[1] ** 2

    def derivative(self, r):
        dx = self.a * r[0] ** 3 - self.b * r[0] + self.c
        dy = self.d * r[1]
        return np.array([dx, dy])

class HarmonicPotential:
    def __init__(self, k, x_o = None):
        self.k = k
        if x_o is None:
            self.x_o = 0
        else:
            self.x_o = x_o
    
    def __call__(self, r):
        return 1/2 * self.k * np.dot((r - self.x_o), (r - self.x_o))

    def derivative(self, r):
        return self.k * (r - self.x_o)

class MuellerPotential:
    def __init__(self, alpha, A, a, b, c, xj, yj):
        self.alpha = alpha
        self.A = A
        self.a = a
        self.b = b
        self.c = c
        self.xj = xj
        self.yj = yj
    
    def __call__(self, r):
        E = 0
        i = 0
        if r.dtype == np.float:
            for A, a, b, c, xj, yj in zip(self.A, self.a, self.b, self.c, self.xj, self.yj):
                i += 1
                E += A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2)
            #     print("Index", i, ":", A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2))
            # print("E =", self.alpha * E)
            # print("r =", r)
            return(self.alpha * E)
        elif r.dtype == torch.float:
            for A, a, b, c, xj, yj in zip(self.A, self.a, self.b, self.c, self.xj, self.yj):
                i += 1
                E += A * torch.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2)
            #     print("Index", i, ":", A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2))
            # print("E =", self.alpha * E)
            # print("r =", r)
            return(self.alpha * E)


    def derivative(self, r):
        dx = 0
        dy = 0
        for A, a, b, c, xj, yj in zip(self.A, self.a, self.b, self.c, self.xj, self.yj):
            dx += A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2) * (2 * a * (r[0] - xj) + b * (r[1] - yj))
            dy += A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2) * (b * (r[0] - xj) + 2 * c * (r[1] - yj))
        # print("dE =", self.alpha * np.array([dx, dy]))
        # print("r = ", r)
        return(self.alpha * np.array([dx, dy]))

class HarmonicBox:
    def __init__(self, l_box, k_box):
        self.l_box = l_box
        self.k_box = k_box

    def __call__(self, r_ij):
        u_upper = np.sum(np.heaviside(r_ij - self.l_box/2, 0) * self.k_box * (r_ij - self.l_box/2) ** 2)
        u_lower = np.sum(np.heaviside(-r_ij - self.l_box/2, 0) * self.k_box * (-r_ij - self.l_box/2) ** 2)
        return u_upper + u_lower

    def derivative(self, r_ij):
        du_upper = np.heaviside(r_ij - self.l_box/2, 0) * 2 * self.k_box * (r_ij - self.l_box/2)
        du_lower = np.heaviside(-r_ij - self.l_box/2, 0) * -2 * self.k_box * (-r_ij - self.l_box/2)
        return du_upper + du_lower


