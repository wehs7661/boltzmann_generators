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
        logp        The log likelihood
        sys_dim     The dimensionality of the system
        """
        
        super(RealNVP, self).__init__()  # nn.Module.__init__()
        self.prior = prior 
        self.mask = nn.Parameter(mask)  # could try requires_grad=False

    def generator(self, z):
        # Fzx
        pass

    def inverse_generator(self, x):
        # Fxz
        pass

    def forward(self):
        pass

    def log_prob(self):
        pass

    def latent_sample(self):
        # sample
        pass

    def loss(self):
        # total loss
        pass

    def loss_ML(self):
        pass

    def loss_KL(self):
        pass

    def calculate_energy(self):
        pass

    def expectation(self):
        pass

        

