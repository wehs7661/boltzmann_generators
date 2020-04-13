import numpy as np 
import copy

class MetropolisSampler:

    def __init__(self, model, x0, temp=1.0, sigma=0.1, stride=1):

        """ 
        A Metropolis sampler that draws samples from a given potential model by running
        a Metropolis Monte-Carlo simulation.

        Parameters
        ----------
        model : potentials object
            The energy model applied.
        x0 : list or numpy.array
            Initial configuration
        sigma : float
            Standard deviation of Gaussian proposal step
        temp : float
            Temperature. By default (1.0) the energy is interpreted 
        stride : int
            The stride for saving the trajectory data.
        """
        self.model = model
        self.sigma = sigma
        self.temp = temp
        self.stride = stride

        # time evolution of the position and the energy
        # self.xtraj, self.etraj = [], []

    def run(self, x, nsteps=1):
        # quantities at t = 0
        x0 = x
        E0 = self.model.get_energy(x[0], x[1])
        self.xtraj, self.etraj = [copy.deepcopy(x0)], [E0]

        for i in range(nsteps):
            E_current = self.model.get_energy(x[0], x[1])
            delta_x = self.sigma * np.random.randn(2,)  # same dimension as self.x
            x_proposed = x + delta_x
            E_proposed = self.model.get_energy(x_proposed[0], x_proposed[1])
            delta_E = E_proposed - E_current
            beta = 1 / self.temp
            
            # metropolis-hasting algorithm
            if delta_E < 0:
                E_current += delta_E
                x += delta_x
            else:
                p_acc = np.exp(-beta * delta_E)
                if np.random.rand() < p_acc:
                    E_current += delta_E
                    x += delta_x
                else:
                    pass   # the coordinates and energy remain the same
            
            if i % self.stride == self.stride - 1:
                self.xtraj.append(copy.deepcopy(x))
                self.etraj.append(E_current)

        self.xtraj = np.array(self.xtraj)
        self.etraj = np.array(self.etraj)

