import numpy as np
import copy
from tqdm.auto import tqdm


class MetropolisSampler:

    def __init__(self, model, temp=1.0, sigma=0.1, stride=1):
        """ 
        A Metropolis sampler that draws samples from a given potential model by running
        a Metropolis Monte-Carlo simulation.

        Parameters
        ----------
        model : potentials object
            The energy model applied.
        temp : float
            Temperature. By default (1.0) the energy is interpreted.
        sigma : float
            Standard deviation of Gaussian proposal step.
        stride : int
            The stride for saving the trajectory data.
        """
        self.model = model
        self.sigma = sigma
        self.temp = temp
        self.stride = stride

    def run(self, x, nsteps, diff=False):
        """
        Run an Monte Carlo simulation. 

        Parameters
        ----------
        x : list or numpy.array
            The coordinates of the initial configuration.
        nsteps : int
            The number of Monte Carlo steps.
        diff : bool
            Whether to remove the duplicated configurations. Spcefically, 
            in a Monte Carlo simulation, the coordinates might or might not
            change between trials. Note that diff=True will remove the same
            configuration but the resulting samples will not be Boltzmann-weighted.

        Attributes
        ----------
        xtraj       (np.array) The coordinate as a function of Monte Carlo step.
        etraj       (np.array) The energy value as a function of Monte Carlo step.
        """
        # quantities at t = 0
        x0 = x
        E0 = self.model.get_energy(x)
        self.xtraj, self.etraj = [copy.deepcopy(x0)], [E0]

        for i in tqdm(range(nsteps)):
            E_current = self.model.get_energy(x)
            delta_x = self.sigma * np.random.randn(*x.shape)  # same dimension as self.x
            x_proposed = x + delta_x
            E_proposed = self.model.get_energy(x_proposed)
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
                if diff is True:
                    if x[0] != self.xtraj[-1][0]:
                        self.xtraj.append(copy.deepcopy(x))
                        self.etraj.append(E_current)
                else:
                    self.xtraj.append(copy.deepcopy(x))
                    self.etraj.append(E_current)

        self.xtraj = np.array(self.xtraj)
        self.etraj = np.array(self.etraj)
