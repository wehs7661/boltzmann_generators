import os 
import sys
import copy
import yaml
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils import data
from generator import RealNVP

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
            n_hidden, l_hidden, n_iteration, batch_size, LR and prior_sigma
        """
        defaults = {'n_blocks': 3, 'dimension': 2, 'n_nodes': 100, 'n_layers': 3, 
                    'n_iterations': 200, 'batch_size': 1000, 'LR': 0.001, 'prior_sigma': 1}

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
        layers = []
        for i in range(self.l_hidden):
            if i == 0:  # first layer
                layers.append(nn.Linear(self.dimension, self.n_nodes))
            elif i == self.n_layers - 1:  # last layer
                laers.append(nn.Linear(self.n_nodes, self.dimension))
            else:  # hidden layers
                layers.append(nn.Linear(self.dimension, self.n_nodes))
            layers.append(nn.ReLU())
        
        t_layers = copy.deepcopy(layers)
        s_layers = copy.deepcopy(layers)
        s_layers.append(nn.Tanh())

        s_net = lambda: nn.Sequential(*s_layers)
        t_net = lambda: nn.Sequential(*t_layers)

        return s_net, t_net

    def build(self, system):
        """

        Parameters
        ----------
        system : object
            The object of the system of interest. (For example, DoubleWellPotential)
        """
        s_net, t_net = self.affine_layers()   # build the affine coupling layers
        mask = torch.from_numpy(np.array([[0, 1], [1, 0]] * self.n_blocks).astype(np.float32))
        prior = distributions.MultivariateNormal(torch.zeros(self.dimension), torch.eye(self.dimension) * self.prior_sigma) 
        model = RealNVP(s_net, t_net, mask, prior, system, (self.dimension,))

        return model

    def preprocess_data(samples):
        training_set = samples.astype('float32')
        subdata = data.DataLoader(dataset=training_set, batch_size=self.batch_size)
        batch = torch.from_numpy(subdata.dataset)   # note that subdata.dataset is a numpy array

        return batch


    def train(self, model, w_loss, x_samples=None, z_samples = None, optimizer=None):
        """
        Trains a Boltzmann generator.

        Parameters
        ----------
        model : objet
            The object of the model to be trained that is built by Boltzmann.build.
        w_loss : list or np.array
            The weighting coefficients of the loss functions. w_loss = [w_ML, w_KL, w_RC]
        samples : np.array
            The training data set for training the model. 
        optimizer : object
            The object of the optimizer for gradient descent method.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=self.LR)
        
        # preprocess the tradining datasets
        if w_loss[0] != 0:                     # w_ML
            if z_samples is None:
                print('Error! w_ML is specfied but no samples in the latent space are given.')
                sys.exit()
            else:
                batch_z = self.preprocess_data(z_samples)

        if w_loss[1] != 0 or w_loss[2] != 0:   # w_KL and w_RC
            if x_samples is None:
                print('Error! w_KL or w_RC is specified but no samples in the configuration space are given.')
                sys.exit()
            else:
                batch_x = self.preprocess_data(x_samples)
            
        # start training!
        self.loss_list = []
        for i in tqdm(range(self.n_iteration)):
            # for the ease of coding, we set loss_X as 0 if loss_X is 0
            loss_ML, loss_KL, loss_RC = w_loss[0], w_loss[1], w_loss[2]

            loss_ML = model.loss_ML(batch_z)
            loss_KL = model.loss_KL(batch_x)
            # loss_KL = model.loss_RC(batch_x)

            loss = w_loss[0] * loss_ML + w_loss[1] * loss_KL + w_loss[2] * loss_RC
            self.loss_list.append(loss.item())  # convert from 1-element tensor to scalar

            # backpropagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()   # check https://tinyurl.com/y8o2y5e7 for more info
            print("Total loss: %s" % loss.item(), end='\r')

    return model

    def save(self, model, save_path):
        """
        Save the trained Boltzmann generator and the parameters used for training.


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
        yaml.dump(model_params, outfile, default_flow_style=False)


