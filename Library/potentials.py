import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 10
})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')


class DoubleWellPotential:
    def __init__(self, **kwargs):
        """
        Initializes a double-well potential model.

        Attributes
        ----------
        All the coefficients are assigned as attrbiutes.
        x_roots         (np.array) The x values of the roots of the equation that the 
                        first derivative of the potential is 0.
        y_roots         (np.array) The y values of the roots of the equation that the 
                        first derivative of the potential is 0.
        stationary_pts  (np.array) The coordinates of the stationary points.
        min_left        (np.array) The coordinates of the left minimum.
        min_right       (np.array) The coordinates of the right minimum.
        saddle          (np.array) The coordiantes of the saddle point.
        e_diff          (float) The potential energy difference between the minima.
        barrier         (float) The height of the energy barrier.
        """
        defaults = {"a": 1, "b": 6, "c": 1, "d": 1}
        if not len(kwargs):
            self.params = defaults
        else:
            self.params = kwargs
        for key in self.params:
            setattr(self, key, self.params[key])

        # stationary pts: solve 4ax^{3} - 2bx + c=0 (y = 0)
        self.x_roots = np.roots([4 * self.a, 0, -2 * self.b, self.c])
        self.y_roots = np.zeros(3)
        self.stationary_pts = np.array(
            [[self.x_roots[i], self.y_roots[i]] for i in range(3)])

        # classify the stationary pts into minima and saddle point
        # solve 12ax^{2} - 2b = 0
        if self.a * self.b >= 0:  # ensure that there are two inflection pts
            for i in range(len(self.x_roots)):
                if self.x_roots[i] > np.sqrt(self.b / 6 * self.a):
                    self.min_right = np.array([self.x_roots[i], 0])
                if self.x_roots[i] < - np.sqrt(self.b / 6 * self.a):
                    self.min_left = np.array([self.x_roots[i], 0])
                else:
                    self.saddle = np.array([self.x_roots[i], 0])

        # energy barrier and energy difference
        e_right = self.get_energy(self.min_right)
        e_left = self.get_energy(self.min_left)
        e_saddle = self.get_energy(self.saddle)
        self.e_diff = e_right - e_left
        self.barrier = e_saddle - e_left

    def get_energy(self, coords):
        """
        Calculates the potential energy for given coordinates.

        Parameters
        ----------
        coords : np.array
            The coordinates of interest.

        Returns
        -------
        E : float or np.array
            The potential energy coorespnoding to the viven coordiantes.
        """
        x = coords[0]
        y = coords[1]
        E = self.a * x ** 4 - self.b * x ** 2 + self.c * x + self.d * y ** 2
        return E

    def plot_section(self, **kwargs):
        """
        Plot the cross section given a fixed value of x or y passed by **kwargs.
        """
        if len(kwargs) != 1:
            print('Error! x or y should fixed.')

        plt.grid()
        if 'x' in kwargs:  # x1 is fixed
            x = kwargs['x']
            y = np.linspace(-7, 7, 100)
            E = self.get_energy([x, y])
            plt.plot(y, E)
            plt.xlabel('$ x_{2} $')
            plt.ylabel('$ Energy $')
            plt.title(
                'Section of the double-well potential at $ x_{1} $ = %s' % x)

        elif 'y' in kwargs:  # x2 is fixed
            y = kwargs['y']
            x = np.linspace(-3, 3, 100)
            E = self.get_energy([x, y])
            plt.plot(x, E)
            plt.xlabel('$ x_{1} $')
            plt.ylabel('Potential energy $u({x_{1}})$')
            plt.title(
                'Cross section of the double-well potential at $ x_{2} $ = %s' % y)

    def plot_FES(self):
        """
        Plots the free energy surface (a contour plot) of the double-well potential.
        """
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-8, 8, 100)
        X, Y = np.meshgrid(x, y)
        E = self.get_energy([X, Y])

        plt.contourf(X, Y, E, 20)
        plt.xlabel('$ x_{1} $')
        plt.ylabel('$ x_{2} $')
        plt.title('Contour plot of the double-well potential')
        clb = plt.colorbar()
        clb.ax.set_title(r'$H(x)$')

    def plot_samples(self, samples=None,  fig=None, color='green'):
        """
        Highlights the positions of the samples on the contour plot of double-well potential.

        Parameters
        ----------
        samples : 
            The samples to be plotted.
        fig : object
            The figure where the samples will be added to.
        color : str
            The color of the samples.

        Returns
        -------
        fig : object
            A contour plot of the double-well potential with samples highlighted.
        """

        nofig = False
        if fig is None:
            nofig = True
            fig = plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        if nofig is True:
            self.plot_FES()
        if samples is not None:
            plt.scatter(samples[:, 0], samples[:, 1], color=color, s=0.5)

        plt.subplot(1, 2, 2)
        if nofig is True:
            self.plot_section(y=0)
        if samples is not None:
            plt.scatter(samples[:, 0], self.get_energy(
                samples.T), color=color, s=0.5)

        return fig

class MullerBrownPotential:
    def __init__(self, alpha = 0.1, A = [-200, -100, -170, 15],
                                    a = [-1, -1, -6.5, 0.7],
                                    b = [0, 0, 11, 0.6],
                                    c = [-10, -10, -6.5, 0.7],
                                    x_j = [1, 0, -0.5, -1],
                                    y_j = [0, 0.5, 1.5, 1]):
        self.alpha = alpha
        self.A = A
        self.a = a
        self.b = b
        self.c = c
        self.x_j = x_j
        self.y_j = y_j

    def get_energy(self, coords):
        x = coords[0]
        y = coords[1]
        if x.dtype == np.float:
            E = np.zeros(x.shape)
            for A, a, b, c, xj, yj in zip(self.A, self.a, self.b, self.c, self.x_j, self.y_j):
                E += A * np.exp(a * (x - xj)**2 + b * (x - xj) * (y - yj) + c * (y - yj)**2)
            return self.alpha * E
        elif x.dtype == torch.float:
            E = torch.zeros(x.shape)
            for A, a, b, c, xj, yj in zip(self.A, self.a, self.b, self.c, self.x_j, self.y_j):
                E += A * torch.exp(a * (x - xj)**2 + b * (x - xj) * (y - yj) + c * (y - yj)**2)
            return self.alpha * E
        
    def plot_section(self, **kwargs):
        
        if len(kwargs) != 1:
            print('Error! x or y should fixed.')
        
        plt.grid()
        if 'x' in kwargs:  # x1 is fixed
            x = kwargs['x']
            y  = np.linspace(-1.2, 0.8, 200)
            E = self.get_energy([x, y])
            plt.plot(y, E)
            plt.xlabel('$ x_{2} $')
            plt.ylabel('$ Energy $')
            plt.title('Section of the muller-brown potential at $ x_{1} $ = %s' % x)

        elif 'y' in kwargs:  # x2 is fixed
            y = kwargs['y']
            x = np.linspace(-1, 1, 100)
            E = self.get_energy([x, y])
            plt.plot(x, E)
            plt.xlabel('$ x_{1} $')
            plt.ylabel('Potential energy $u({x_{1}})$')
            plt.title('Cross section of the muller-brown potential at $ x_{2} $ = %s' % y)

        elif 'offset_diag' in kwargs:
            off_set = kwargs['offset_diag']
            x = np.linspace(-1.2, 0.8, 200)
            y = -x + off_set
            xy_data = np.array([x, y]).T
            x_proj = np.dot(xy_data, np.array([1,-1]))/np.dot(np.array([1,-1]),np.array([1,-1]))
            E = self.get_energy([x,y])
            plt.plot(x_proj, E)
            plt.xlabel('$ x_{1} $')
            plt.ylabel('Potential energy $u({x_{1}})$')
            plt.title('Cross section of the muller-brown potential on line x = -y + %s' % off_set)

    def plot_samples(self, samples=None,  fig=None, color='green'):
        nofig = False
        if fig is None:
            nofig = True
            fig = plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        if nofig is True:
            self.plot_FES()
        if samples is not None:
            plt.scatter(samples[:, 0], samples[:, 1], color=color, s=0.5)
        
        plt.subplot(1, 2, 2)
        if nofig is True:
            self.plot_section(offset_diag=0.5)
        if samples is not None:
            x = samples[:, 0]
            y = samples[:, 1]
            xy_data = np.array([x, y]).T
            x_proj = np.dot(xy_data, np.array([1,-1]))/np.dot(np.array([1,-1]),np.array([1,-1]))
            plt.scatter(x_proj, self.get_energy(samples.T), color=color, s=0.5)

        return fig

    def plot_FES(self):
        x = np.linspace(-1.5, 1.25, 100)
        y = np.linspace(-0.5, 2, 100)
        X, Y = np.meshgrid(x, y)
        E = self.get_energy([X, Y])

        plt.contourf(X, Y, E, 500, cmap = "jet", vmin = -10, vmax = -3)
        plt.xlabel('$ x_{1} $')
        plt.ylabel('$ x_{2} $')
        plt.title('Contour plot of the muller-brown potential')
        clb = plt.colorbar()
        clb.ax.set_title(r'$H(x)$')

class DimerSimulation:
    def __init__(self, epsilon = 1.0, sigma = 1.1, k_d = 20, d0 = 1.5, a = 25, \
                 b = 10, c = -0.5, l_box = 3, k_box = 100, harmonic_centering = True):
        self.epsilon = epsilon
        self.sigma = sigma
        self.k_d = k_d
        self.d0 = d0
        self.a = a
        self.b = b
        self.c = c
        self.l_box = l_box
        self.k_box = k_box
        self.harmonic_centering = harmonic_centering

    def bond_energy(self, d):
        return 1/4*self.a*(d - self.d0)**4 - 1/2*self.b*(d - self.d0)**2 + self.c*(d - self.d0)


    def get_energy(self, coords):
        U = 0
        d = np.linalg.norm(coords[0, :] - coords[1, :])
        
        # Bond Potential
        U += self.bond_energy(d)
        if self.harmonic_centering:
            U += self.k_d*(coords[0, 0] + coords[1, 0]) ** 2
            U += self.k_d*(coords[0, 1] ** 2 + coords[1, 1] ** 2)            
        for i in range(len(coords)):
            
            # Boundary Conditions
            if abs(coords[i, 0]) >= self.l_box:
                U += self.k_box*(abs(coords[i,0]) - self.l_box)**2
            if abs(coords[i, 1]) >= self.l_box:
                U += self.k_box*(abs(coords[i, 1]) - self.l_box)**2

            # LJ Interactions
            for j in range(2, len(coords)):
                if i == j:
                    continue
                d = coords[i, :] - coords[j, :]
                U += self.epsilon * (self.sigma**2 / np.dot(d,d))**6
        return U




