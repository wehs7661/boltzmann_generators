import sys
import copy
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class RealNVP(nn.Module):  # inherit from nn.Module
    def __init__(self, s_net, t_net, mask, prior, system, sys_dim):
        """
        This is the init function for RealNVP class.

        Parameters
        ----------
        s_net : lambda function
            The scaling function as a neural network in the anonymous form. 
        t_net : lambda function
            The translation functione as a neural network in the anonymous form.
        mask : torch.Tensor
            The masking scheme for affine coupling layers. 
        prior : torch.distributions object
            The prior probability distribution in the latent space (generally a normal distribution)
        system : object
            The object of the system of interest. (For example, DoubleWellPotential)
        sys_dim : tuple
            The dimensionality of the system

        Attributes
        ----------
        prior       The prior probability distribution in the latent space (generally a normal distribution)
        mask        The masking scheme for affine coupling layers.
        s           The scaling function in the affine coupling layers.
        t           The translation function in the affine coupling layers.
        sys_dim     The dimensionality of the system
        """
        
        super(RealNVP, self).__init__()  # nn.Module.__init__()
        self.prior = prior 
        self.mask = nn.Parameter(mask, requires_grad=False)  # could try requires_grad=False
        self.t = torch.nn.ModuleList([t_net() for _ in range(len(mask))]) 
        self.s = torch.nn.ModuleList([s_net() for _ in range(len(mask))]) 
        # nn.ModuleList is basically just like a Python list, used to store a desired number of nn.Module’s.
        # Also note that t[i] and s[i] are entire sequences of operations
        self.system = system # class of what molecular system are we considering. E.g. Ising.
        self.sys_dim = sys_dim # tuple describing original dim. of system. e.g. Ising Model with N = 8 would be (8,8)

    def inverse_generator(self, x, process=False):
        """
        Inverse Boltzmann generator which transforms the samples drawn from the probability distribution
        in the configuration space to the real space. (Fxz in the Boltzmann generator paper)

        Parameters
        ---------
        x : torch.Tensor
            Samples drawn from the probability distribution in the configuration space

        Returns
        -------
        z : torch.Tensor
            Samples generated in the latent space
        log_R_xz : torch.Tensor
            Log of the determinant of the Jacobian of Fxz (invese generator)
        """
        z = x   # just for initialization
        log_R_xz = x.new_zeros(x.shape[0])
        # new_zeros(size) returns a tensor of size "size" filled with 0s
        z_list = []
        z_list.append(copy.deepcopy(x.detach().numpy()))

        for i in reversed(range(len(self.t))):   # move backwards through the layers
            # here we split the dataset into two channels (with 1:d and d+1:D dimensions)
            # See equation (9) in the original RealNVP papaer
            z_ = self.mask[i] * z    # b * x in equation (9)
            s = self.s[i](z_)        # s(b * x) in equation (9)
            t = self.t[i](z_)        # t(b * x) in equation (9)
            z = z_ + (1 - self.mask[i]) * (z - t) * torch.exp(-s)  # equation (9)
            log_R_xz -= torch.sum(s, -1)  # negative sign: since we are moving backward!
            if process is True: 
                z_list.append(copy.deepcopy(z.detach().numpy()))

        if process is False:
            return z, log_R_xz
        else:
            return z, log_R_xz, z_list

    def generator(self, z, process=False):
        """ 
        Boltzmann generator which transforms the samples drawn from the probability distribution 
        in the latent space to the configuration space. (Fzx in the Boltzmann generator paper)
        
        Parameters
        ----------
        z : torch.Tensor
            Samples drawn from the the probability distribution in the latent space

        Returns
        -------
        x : torch.Tensor
            Samples generated in the configuration space
        log_R_zx : torch.Tensore
            Log of the determinant of the Jacobian of Fzx (generator)
        """
        x = z   # just for initialization
        log_R_zx = z.new_zeros(z.shape[0])
        # new_zeros(size) returns a tensor of size "size" filled with 0s
        x_list = []
        x_list.append(copy.deepcopy(z.detach().numpy()))

        for  i in range(len(self.t)):
            # here we split the dataset into two channels (with 1:d and d+1:D dimensions)
            # See equation (9) in the original RealNVP papaer
            x_ = self.mask[i] * x           # b * x in equation (9)
            s = self.s[i](x_)               # s(b * x) in equation (9)
            t = self.t[i](x_)               # t(b * x) in equation (9)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)   # equation (9)
            log_R_zx += torch.sum(s, -1)    # equation (6) 
            if process is True:
                x_list.append(copy.deepcopy(x.detach().numpy()))
        
        if process is True:
            return x, log_R_zx, x_list
        else:
            return x, log_R_zx

    def loss_total(self, batch, w_ml=1.0, w_kl=1.0, w_rc=1.0):
        """
        Calculate the total loss function.

        Parameters
        ----------
        batch : 

        w_ml : 

        w_kl : 

        w_rc

        Returns
        -------
        loss : 

        """        
        loss = w_ml * self.loss_ML(batch) + w_kl * self.loss_ML(batch) + w_rc * self.loss_RC(batch)

        return loss
        
    def loss_ML(self, batch_x):
        """
        Calculate the loss function when training by example (samples from the configuration space)
        J_ML = E[u_z(z) - log Rxz(x)], where u_z(z) = 0.5 * /(sigma^{2}) * z^{2} (sigma = 1)

        Parameters
        ----------
        batch_x : torch.Tensor
            A batch of samples in the configuration space.
        
        Returns
        -------
        J_ml : torch.Tensor
            The loss function J_ML
        """
        z, log_R_xz = self.inverse_generator(batch_x)
        # we don't need u_z in the calculation of J_ml
        # but we do need u_x in the calculation of J_kl, see the method 'loss_KL'
        # we need u_x to caluculate the weighs for calculation expecatation in the confiugration space
        u_x = self.calculate_energy(batch_x, space='configuration')
        weights_x = torch.exp(-u_x)
        J_ml = self.expectation(0.5 * torch.norm(z, dim=1) ** 2 - log_R_xz, weights=weights_x)

        return J_ml

    def loss_KL(self, batch_z):
        """
        Calculate the loss function when training by energy (samples from the latent space)
        J_KL = E[u_x(x) - log Rzx(z)]

        Parameters
        ----------
        batch_z : torch.Tensor
            A batch of samples in the latent space

        Returns
        -------
        J_kl : torch.Tensor
            The loss function J_KL
        """
        x, log_R_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x, space='configuration')   # we need this to calculate J_kl
        # we need u_z to caluculate the weighs for calculation expecatation in the latant space
        u_z = self.calculate_energy(batch_z, space='latent')
        weights_z = torch.exp(-u_z)
        J_kl = self.expectation(u_x - log_R_zx, weights=weights_z)

        return J_kl

    def loss_RC(self):
        pass

    def calculate_energy(self, batch, space):
        """
        Calculate the energy of each each configuration in a batch of dataset.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of configurations
        
        Returns
        -------
        energy : torch.Tensor
            The energies of the configurations
        space : str
            Whether to calcualte the energy in the real space (x) or the 
            latent space (z). Available options: 'latent' or 'configuration'.
        """

        energy = batch.new_zeros(batch.shape[0])  # like np.zeros, same length as batch_data
        if space == 'configuration':
            for i in range(batch.shape[0]):  # for each data point in the dataset
                config = batch[i, :].reshape(self.sys_dim)  # ensure correct dimensionality
                energy[i] = self.regularize_energy(self.system.get_energy(config.detach().numpy()))
                # import pdb
                # pdb.set_trace()
                # regularize the energy (see page 3 in the SI)

        elif space == 'latent':
            for i in range(batch.shape[0]):  # for each data point in the dataset
                config = batch[i, :].reshape(self.sys_dim)  # ensure correct dimensionality
                # for 2D Gaussian distribution, u(z) = (1 / (2*sigma **2)) * z ** 2
                # in our case, sigma =1 and z ** 2 = z[0] ** 2 + z[1] ** 2
                energy[i] = self.regularize_energy(0.5 * (config[0] ** 2 + config[1] ** 2))
                # regularize the energy (see page 3 in the SI)

        else:
            print("Error! Unavailable option of parameter 'space' specificed.")
            sys.exit()

        return energy

    def regularize_energy(self, energy, e_high = 10 ** 4, e_max = 10 ** 20):
        if energy.item() > e_high:
            energy = e_high + np.log10(energy - e_high + 1)
        elif energy.item() > e_max:
            energy= e_high + np.log10(e_max - e_high + 1)
        return energy


    def expectation(self, observable, weights):
        """ 
        Calculate the expectation value of an observable
        
        Parameters
        ----------
        observable : torch.Tensor
            Observable of interest.

        Returns
        -------
        e : torch.Tensor
            Expectation value as a one-element tensor
        """
        # e = torch.dot(observable, self.weights) / torch.sum(self.weights) the same as below
        e = torch.sum(observable * weights) / torch.sum(weights)

        return e

        

