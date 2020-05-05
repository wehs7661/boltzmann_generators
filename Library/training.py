import os
import sys
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils import data
from torch import distributions
from generator import RealNVP
from collections import OrderedDict
from density_estimator import density_estimator

# the following two functions and setup_yaml() are for preserving the
# order of the dictionary to be printed to the parameter yaml file


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())


def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


setup_yaml()


class BoltzmannGenerator:
    def __init__(self, model_params=None):
        """
        Initialize a Boltzmann generator. Note that this method assumes identical number 
        of nodes for all layers in the network (except for the input/output nodes). The
        activation functions are all ReLU function in all layers, except that the the last
        activation function of the scaling network is a hyperbolic tangent function. 

        Parameters
        ----------
        model_params : dict
            A dictionary of the model parameters, including the n_blocks, dimension, 
            n_nodes, n_layers, n_epochs, batch_size, LR and prior_sigma.

        Attributes
        ----------
        All the keys in model_param will be assigned as attributes.
        params      (dict) The model parameters.
        """
        defaults = {'n_blocks': 3, 'dimension': 2, 'n_nodes': 100, 'n_layers': 3,
                    'n_epochs': 200, 'batch_size': 2048, 'LR': 0.001, 'prior_sigma': 1}

        if model_params is None:
            self.params = defaults
        else:
            # check if all the parameters are specified
            for key in defaults:
                if key not in model_params:
                    model_params[key] = defaults[key]
            else:
                self.params = model_params

        # Assign attributes
        for key in self.params:
            setattr(self, key, self.params[key])

    def affine_layers(self):
        """
        This method defines the common architecture of the networks in 
        an affine coupling layers for each NVP block. All the activation 
        functions applied here are ReLU functions.

        Returns
        -------
        layers : list
            A list of affine coupling layers along with the corresponding
            activation functions(nn.Linear() and nn.ReLU()).
        """
        layers = []
        for i in range(self.n_layers):
            if i == 0:  # first layer
                layers.append(nn.Linear(self.dimension, self.n_nodes))
            elif i == self.n_layers - 1:  # last layer
                layers.append(nn.Linear(self.n_nodes, self.dimension))
            else:  # hidden layers
                layers.append(nn.Linear(self.n_nodes, self.n_nodes))

            if i != self.n_layers - 1:
                layers.append(nn.ReLU())

        return layers

    def build_networks(self):
        """
        Build the networks (s_net and t_net) for scaling and translating in 
        an affine coupling layer and assign them as attributes.

        Attributes
        ----------
        s_net       (lambda function) The scaling network in an afffine coupling layer for one NVP block.
        t_net       (labmda function) The translation network in an affine coupling layer for one NVP block.
        """
        self.s_net = lambda: nn.Sequential(*self.affine_layers(), nn.Tanh())
        self.t_net = lambda: nn.Sequential(*self.affine_layers())

    def build(self, system, mask = None):
        """
        This method builds a Boltzmann generator given the system of interest.

        Parameters
        ----------
        system : object
            The object of the system of interest. (For example, DoubleWellPotential)

        Attributes
        ----------
        mask        (torch.Tensor) The masking shceme in the form of a tensor.
        prior       (torch.distributions) The prior probability distribution in the latent space.

        Returns
        -------
        model : object
            An RealNVP object, which is an untrained Boltzmann generator.
        """
        self.system = system
        self.build_networks()   # build the affine coupling layers
        affine = np.concatenate((np.ones(int(self.dimension/2)), np.zeros(int(self.dimension/2))))
        affine = np.array([affine, np.flip(affine)] * self.n_blocks)
        self.mask = torch.from_numpy(affine.astype(np.float32))
        self.prior = distributions.MultivariateNormal(torch.zeros(
            self.dimension), torch.eye(self.dimension) * self.prior_sigma)
        
        model = RealNVP(self.s_net, self.t_net, self.mask,
                        self.prior, self.system, self.reshape)
        for key in self.params:
            setattr(model, key, self.params[key])

        return model

    def preprocess_data(self, samples):
        """
        This method preprocess the samples and return a batch of samples.

        Attributes
        ----------
        n_pts       (int) The number of samples in total.

        Returns
        -------
        batch : torch.Tensor
            A batch of samples serving as the input for inverse generators or generators.
        """
        self.n_pts = len(samples)   # number of data points
        training_set = samples.astype('float32')
        subdata = data.DataLoader(
            dataset=training_set, batch_size=self.batch_size)

        return subdata

    def train(self, model, w_loss, x_samples=None, z_samples=None, optimizer=None, KDE_optimize=False, rxn_coordinate = None):
        """
        Trains a Boltzmann generator. This method does not return anything, but the 
        parameters in the input model will be adjusted based on the training.

        Parameters
        ----------
        model : objet
            The object of the model to be trained that is built by Boltzmann.build.
        w_loss : list or np.array
            The weighting coefficients of the loss functions. w_loss = [w_ML, w_KL, w_RC]
        x_samples : np.array
            The training data set in the configuration sapce for training the inverse generator.
        z_samples : np.array
            The training data set in the latent space for training the generator. 
        optimizer : object
            The object of the optimizer for gradient descent method.
        KDE_optimize : bool
            Whether to optimize the bandwitch when using KDE to estimate the probability
            distributions for RC loss calculation.
        rxn_coordinate : lambda function
            Function for getting rxn coordinate of interest out of system
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad == True], lr=self.LR)
        
        if rxn_coordinate is None:
            rxn = lambda x: x[:,0]

        # preprocess the training datasets
        if w_loss[0] != 0 or w_loss[2] != 0:    # calculations of J_ML and J_RC requires x_samples
            if x_samples is None:
                print(
                    'Error! w_ML is specfied but no samples in the configuration space are given.')
                sys.exit()
            else:
                subdata_x = self.preprocess_data(x_samples)

            if w_loss[1] == 0 and z_samples is None:
                # when only training on J_ML, we don't need z_samples but we need to make 
                # up subdata_z to enable zip function in the training loop. subdata_z could 
                # be any iterable, since w_loss[1] == 0 and w_loss[2] == 0, subdata_z won't be trained anyway
                subdata_z = np.zeros(len(subdata_x))

        if w_loss[1] != 0:   # Calculation of J_KL reqruies z_samples
            if z_samples is None:
                print(
                    'Error! w_KL or w_RC is specified but no samples in the latent space are given.')
                sys.exit()
            else:
                subdata_z = self.preprocess_data(z_samples)

            if (w_loss[0] == 0 and w_loss[2] == 0) and x_samples is None:
                # Similarly, we don't need x_samples if w_ML and w_RC are both 0, we don't need x_samples
                # but we still need to make up subdata_x
                subdata_x = np.zeros(len(subdata_z))

        # for the ease of coding, we set loss_X as 0 if loss_X is 0
        loss_ML, loss_KL, loss_RC = w_loss[0], w_loss[1], w_loss[2]

        if w_loss[2] != 0:
            estimator = density_estimator(rxn(x_samples), optimize=KDE_optimize)
        
        # start training!
        self.loss_iteration = []   # the loss of each iteration

        for i in tqdm(range(self.n_epochs)):
            for batch_x, batch_z in zip(subdata_x, subdata_z):  # iterations
                if w_loss[0] != 0:
                    loss_ML = model.loss_ML(batch_x)
                if w_loss[1] != 0:
                    loss_KL = model.loss_KL(batch_z)
                if w_loss[2] != 0:
                    loss_RC = model.loss_RC(batch_x, estimator)

                loss = w_loss[0] * loss_ML + w_loss[1] * \
                    loss_KL + w_loss[2] * loss_RC   # the loss of a iteration

                # convert from 1-element tensor to scalar
                self.loss_iteration.append(loss.item())

                # backpropagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()   # check https://tinyurl.com/y8o2y5e7 for more info
                print("Total loss: %s" % loss.item(), end='\r')

    def save(self, model, save_path, previous_loss=None):
        """
        Save the trained Boltzmann generator and the parameters used for training.

        model : object
            The Boltzmann generator model to be saved. 
        save_path : str
            The directory where the model will be saved (along with the filename).
        previous_loss : list
            A list of previus loss function values.
        """
        # save the trained model
        torch.save(model.state_dict(), save_path)

        # save the parameters for training the model
        for file in os.listdir('.'):
            if file == save_path + '.yml':
                # make sure that the output file is newly made
                os.remove(save_path + '.yml')

        outfile = open(save_path + '.yml', 'a+', newline='')
        outfile.write('# Pamaeters for training the model\n')
        model_params = copy.deepcopy(vars(self))
        del model_params['params']
        del model_params['s_net']
        del model_params['t_net']
        del model_params['mask']
        del model_params['prior']
        del model_params['loss_iteration']  # use outfile.write instead
        yaml.dump(model_params, outfile, default_flow_style=False)
        outfile.write('\n# Training result\n')

        if previous_loss is not None:
            self.loss_iteration = previous_loss + self.loss_iteration
        outfile.write('loss: ' + str(self.loss_iteration))

    def load(self, model, load_path):
        """
        Loads a trained Boltzmann generator and the training result

        model : objet
            The object of the model to be trained that is built by build method.
            Note that this model must have the same architecture as the trained model
            to be loaded.
        load_path : str
            The directory where the model was saved (along with the filename).
        """
        # load the trained model
        model.load_state_dict(torch.load(load_path))

        # load the training result (loss_iteration)
        f = open(load_path + '.yml', 'r')
        lines = f.readlines()
        f.close()

        loss_found = False
        for l in lines:
            if 'loss: ' in l:
                loss_found = True
                loss_str = l.split('[')[1].split(']')[
                    0].split(',')  # a list of string
                loss = [float(i) for i in loss_str]

        if loss_found is False:
            print("Error! Incomplete results stored in %s" %
                  (load_path + '.yml'))
            sys.exit()

        print('Total loss: %s' % loss[-1])

        return model, loss
