import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from torch import distributions
from scipy.stats import multivariate_normal

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 10
})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')

class BoltzmannPlotter:
    def __init__(self, system, model, x_samples, n_z=5000, prior=None):
        """
        Initialize the class for visualizing the results of a Boltzmann generator.

        Parameters
        ----------
        system : object
            The object of the system of interest. (For example, DoubleWellPotential)
        model : object
            A trained Boltzmann generator
        x_samples : list
            x_samples in a list. There could be mutiple datasets (np.array) in the 
            list which correspond to different metastable states.
        """
        self.system = system
        self.model = model
        self.x_samples = x_samples
        self.n_z = n_z
        self.prior = prior
        if self.prior is None:
            self.prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)) 

        # below we set up a list of 12 colors that look distinct with the colors
        # of the contour plot
        ncolors = 10  # number of distinct colors
        first_rgb , last_rgb = 50, 225
        cmap = plt.cm.jet  
        colors = ['red', 'silver']  # first two colors
        colors.append([cmap(i) for i in np.linspace(first_rgb, last_rgb, ncolors).astype(int)])
        self.colors = np.array(colors)


    def binormal_contour(self):
        x, y = np.mgrid[-3 : 3 : 0.01, -3 : 3 : 0.01]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
        
        plt.contourf(x, y, rv.pdf(pos), 20)
        plt.xlabel('$ z_{1} $')
        plt.ylabel('$ z_{2} $')
        plt.title('Bivariate normal distribution')
        clb = plt.colorbar()
        clb.ax.set_title(r'$P(z)$')
        
        ax = plt.gca()
        ax.set_aspect('equal')

    def transform_x_samples(self, x_samples):
        z, _, z_list = self.model.inverse_generator(torch.from_numpy(x_samples.astype('float32')), process=True)
        z = z.detach().numpy()

        return z, z_list

    def generator_result(self, save_path):

        fig = plt.subplots(1, 4, figsize=(25, 5))

        # First subplot: samples drawn from the configuration space
        plt.subplot(1, 4, 1)
        self.system.plot_FES()
        for i in range(len(self.x_samples)):
            plt.scatter(self.x_samples[i][:, 0], self.x_samples[i][:, 1], color=self.colors[i], s=0.5)
        if type(self.system).__name__ == 'DoubleWellPotential':
            plt.annotate('(configuration space, $ x \sim p(x) $)', xy=(0, 0), xytext=(-4.2, 6.5), color='white', size='12')
        ax = plt.gca()
        ax.set_aspect(0.58)

        # Second subplot: transform configuration samples to latent samples using a trained inversed generator
        z_generated = []
        self.z_generated_list = []   # for plotting "map_to_latent"
        for i in range(len(self.x_samples)):
            z, z_list = self.transform_x_samples(self.x_samples[i])
            z_generated.append(z)
            self.z_generated_list.append(z_list)  # will not be used in this method but useful for other methods
        
        plt.subplot(1, 4, 2)
        self.binormal_contour()
        for i in range(len(z_generated)):
            plt.scatter(z_generated[i][:, 0], z_generated[i][:, 1], color=self.colors[i], s=0.5)
        if type(self.system).__name__ == 'DoubleWellPotential':
            plt.annotate('(latent space, $ z = F_{xz}(x) $)', xy=(0, 0), xytext=(-2.625, 2.4375), color='white', size='12')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])

        # Third subplot: draw samples from the prior Gaussian distribution
        z_samples = self.prior.sample_n(self.n_z)
        z_samples = z_samples.detach().numpy()  

        plt.subplot(1, 4, 3)
        self.binormal_contour()
        plt.scatter(z_samples[:, 0], z_samples[:, 1], color='white', s=0.5)
        if type(self.system).__name__ == 'DoubleWellPotential':
            plt.annotate('(latent space, $ z \sim p(z) $)', xy=(0, 0), xytext=(-2.625, 2.4375), color='white', size='12')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])

        # fourth subplot: transform the latent samples back to the configuration space using a generator
        # self.x_generated_list will not be used here, but is useful in other methods
        x_generated, _, self.x_generated_list = self.model.generator(torch.from_numpy(z_samples), process=True)
        x_generated = x_generated.detach().numpy()

        plt.subplot(1, 4, 4)
        self.system.plot_FES()
        plt.scatter(x_generated[:, 0], x_generated[:, 1], color='white', s=0.5)
        if type(self.system).__name__ == 'DoubleWellPotential':
            plt.annotate('(configuration space, $ x = F_{zx}(z) $)', xy=(0, 0), xytext=(-4.2, 6.5), color='white', size='12')
            plt.xlim([-5, 5])
            plt.ylim([-8, 8])
        ax = plt.gca()
        ax.set_aspect(0.58)

        plt.savefig(save_path, dpi=600)

    def map_titles(self):
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
        title_list = ['Input distribution']
        for i in range(self.model.n_blocks):
            if i == self.model.n_blocks - 1:
                title_list.append('%s NVP block, generated samples' % ordinal(i + 1))
            else:
                title_list.append('%s NVP block' % ordinal(i + 1))

        return title_list


    def map_to_latent(self, save_path, z_generated_list=None):
        # note that self.z_generated_list used here might have multiple datasets of the tranformation process
        # corresponding to different metastable states mapped from configuration space

        if hasattr(self, 'x_generated_list') is False:
            # Note: if both are provided, we priortize self.x_generated_list
            self.x_generated_list = x_generated_list
            if self.x_generated_list is None:
                print("Error! There is no attribute 'x_generated_list' and no datasets are provided!")
                sys.exit()

        fig = plt.subplots(1, self.model.n_blocks + 1, figsize=(25, 4))
        title_list = self.map_titles()

        for i in range(self.model.n_blocks + 1):  # plus "input distribution"
            plt.subplot(1, self.model.n_blocks + 1, i + 1)
            self.binormal_contour()
            for j in range(len(self.z_generated_list)):
                plt.scatter(self.z_generated_list[j][i * 2][:, 0], self.z_generated_list[j][i * 2][:, 1], color=self.colors[j], s=0.5)
            
            plt.title(title_list[i])
            if type(self.system).__name__ == 'DoubleWellPotential':
                plt.annotate('(latent space, $ z = F_{xz}(x) $)', xy=(0, 0), xytext=(-2.625, 2.4375), color='white', size='12')
            
            plt.xlim([-3, 3])
            plt.ylim([-3, 3])
    
        plt.savefig(save_path, dpi=600)

    def map_to_configuraiton(self, save_path, x_generated_list=None):
        # Note that unlinke z_generated_list, x_generated_list should only have one
        # dataset of transformation process in the configuration space
        
        if hasattr(self, 'x_generated_list') is False:
            # Note: if both are provided, we priortize self.x_generated_list
            self.x_generated_list = x_generated_list
            if self.x_generated_list is None:
                print("Error! There is no attribute 'x_generated_list' and no datasets are provided!")
                sys.exit()

        fig = plt.subplots(1, self.model.n_blocks + 1, figsize=(25, 4))
        title_list = self.map_titles()

        for i in range(self.model.n_blocks + 1):
            plt.subplot(1, self.model.n_blocks + 1, i + 1)
            self.system.plot_FES()
            plt.scatter(self.x_generated_list[i * 2][:, 0], self.x_generated_list[i * 2][:, 1], color='white', s=0.5)
            plt.title(title_list[i])
            if type(self.system).__name__ == 'DoubleWellPotential':
                plt.annotate('(configuration space, $ x = F_{zx}(z) $)', xy=(0, 0), xytext=(-4.2, 6.5), color='white', size='12')
                plt.xlim([-5, 5])
                plt.ylim([-8, 8])
            
            ax = plt.gca()
            ax.set_aspect(0.58)
    
        plt.savefig(save_path, dpi=600)


