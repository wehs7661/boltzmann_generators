# Visualization Library
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from matplotlib import animation
from torch import distributions
from scipy.stats import multivariate_normal
from density_estimator import density_estimator

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 10
})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')


def make_2D_traj_potential(x_traj, potential, xlim, ylim, min = -10, max = -3, fps = 30, markersize = 8, color = 'red', cmap = "viridis"):
    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1],50), np.linspace(ylim[0],ylim[1],50))
    Z = []
    for i in range(len(X)):
        z = []
        for j in range(len(X)):
            v = potential(np.array([X[i,j], Y[i,j]]))
            z.append(v)
        Z.append(z)
    Z = np.array(Z)

    fps = fps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(X,Y,Z, levels = 500, cmap=cmap)
    sct, = ax.plot([], [], "o", markersize=markersize, color = color, vmin = min, vmax = max)
    cbar = fig.colorbar(cs)
    plt.clim(min,max)
    # pct, = ax.plot([], [], "o", markersize=8)

    def update_graph(i, xa, ya): # , vx, vy):
        sct.set_data(xa[i], ya[i])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    video_traj = x_traj
    ani = animation.FuncAnimation(fig, update_graph, video_traj.shape[0], fargs=(video_traj[:,:,0], video_traj[:,:,1]), interval=1000/fps)
    plt.rcParams['animation.html'] = 'html5'
    return(ani)


def make_2D_traj(x_traj, box, fps = 30, markersize = 8, color = 'red'):
    fps = fps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [], "o", markersize=markersize, color = color)
    # pct, = ax.plot([], [], "o", markersize=8)

    def update_graph(i, xa, ya): # , vx, vy):
        sct.set_data(xa[i], ya[i])

    x_lim = box[0]/2 
    y_lim = box[1]/2
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-y_lim, y_lim])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    video_traj = x_traj
    ani = animation.FuncAnimation(fig, update_graph, video_traj.shape[0], fargs=(video_traj[:,:,0], video_traj[:,:,1]), interval=1000/fps)
    plt.rcParams['animation.html'] = 'html5'
    return(ani)

def plot_2D_potential(potential, xlim, ylim, min = -10, max = -3, fps = 30, markersize = 8, color = 'red', cmap = "viridis"):
    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1],50), np.linspace(ylim[0],ylim[1],50))
    Z = []
    for i in range(len(X)):
        z = []
        for j in range(len(X)):
            v = potential(np.array([X[i,j], Y[i,j]]))
            z.append(v)
        Z.append(z)
    Z = np.array(Z)
    cs = plt.contourf(X,Y,Z, levels = 500, cmap=cmap, vmin=min, vmax=max)
    cbar = plt.colorbar(cs)
    plt.clim(min,max)
    plt.ylabel("$x_2$")
    plt.xlabel("$x_1$")

def make_2D_traj_bond(x_traj, box, bonds, fps = 30, markersize = 8, color = 'red'):
    fps = fps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [], "o", markersize=markersize, color = color)
    bct, = ax.plot([], [], color = color, linewidth=markersize/3)
    # pct, = ax.plot([], [], "o", markersize=8)

    def update_graph(i, xa, ya): # , vx, vy):
        sct.set_data(xa[i], ya[i])
        for bond in bonds:
            bct.set_data([xa[i][bond[0]], xa[i][bond[1]]], [ya[i][bond[0]], ya[i][bond[1]]])



    x_lim = box[0]/2 
    y_lim = box[1]/2
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-y_lim, y_lim])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    video_traj = x_traj
    ani = animation.FuncAnimation(fig, update_graph, video_traj.shape[0], fargs=(video_traj[:,:,0], video_traj[:,:,1]), interval=1000/fps)
    plt.rcParams['animation.html'] = 'html5'
    return(ani)


def plot_bond_sampling(x_traj, dimer, bins = 20):
    # Simulation Sampling
    bond_dist = [np.linalg.norm(x_traj[i][0] - x_traj[i][1]) for i in range(x_traj.shape[0])]

    # Analytical energy
    distance = np.linspace(0.5, 2.5, 30)
    bond_energy = dimer.bond_energy(distance)

    # Plotting
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Dimer Distance")
    ax1.set_ylabel("Counts", color = "red")
    ax1.hist(bond_dist, bins = bins, color = "red")

    ax2.plot(distance, bond_energy, color = "blue")
    ax2.set_ylabel("Bond Energy", color="blue")
    ax2.tick_params(axis='y')


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

        # below we set up a list of 102 colors that look distinct with the colors
        # of the contour plot
        ncolors = 10  # number of distinct colors
        first_rgb , last_rgb = 50, 225
        cmap = plt.cm.jet  
        colors = ['red', 'silver']  # first two colors
        colors += [cmap(i) for i in np.linspace(first_rgb, last_rgb, ncolors).astype(int)]
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

    def transform_x_samples(self, x_samples, process=False):
        """
        x_samples : np.array
        """
        z, _, z_list = self.model.inverse_generator(torch.from_numpy(x_samples.astype('float32')), process=process)
        z = z.detach().numpy()

        return z, z_list

    def generator_result(self, save_path):
        """
        save_path : str
            The directory where the model will be saved (along with the filename)
        """

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
            z, z_list = self.transform_x_samples(self.x_samples[i], True)
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
    
    def latent_interpolation(self, save_path, n_path=10, x1=None, x2=None):
        """

        Parameters
        ----------
        n_path : int
            Number of reaction paths to plot. (Also the number of linear interpolations.)

        """
        # Part 1-1: randomly choose points near the minima (the first point of x1 and x2 are minima)
        if type(self.system).__name__ == 'DoubleWellPotential':
            x1 = np.ones([n_path,2]) * self.system.min_left 
            x2 = np.ones([n_path,2]) * self.system.min_right
        else:
            if x1 is None or x2 is None:
                print("Error! No data points or type of system provided!")
            else:
                x1 = np.ones([n_path,2]) * x1 
                x2 = np.ones([n_path,2]) * x2 

        x1 += np.vstack((np.array([0, 0]), np.random.random([n_path - 1, 2]) - 0.5))
        x2 += np.vstack((np.array([0, 0]), np.random.random([n_path - 1, 2]) - 0.5))
        
        # Part 1-2: plot the samples x1, x2
        fig = plt.subplots(1, 3, figsize=(18, 4.5))
        plt.subplot(1, 3, 1)
        self.system.plot_FES()
        plt.scatter(x1[:, 0], x1[:, 1], color=self.colors[0], s=1)
        plt.scatter(x2[:, 0], x2[:, 1], color=self.colors[1], s=1)
        ax = plt.gca()
        ax.set_aspect(0.58)

        # Part 2-1: map the samples from the configuration space to latent space
        z1, _ = self.model.inverse_generator(torch.from_numpy(x1.astype('float32')))
        z2, _ = self.model.inverse_generator(torch.from_numpy(x2.astype('float32')))
        z1 = z1.detach().numpy()
        z2 = z2.detach().numpy()
        
        # Part 2-2: peform linear interoplation and plot the samples in the latent space
        # z_samples contains 100 points on each line connected by each pair of points in z1 and z2
        z_samples = torch.Tensor(np.linspace(z1, z2, 100, axis=1))
        z = z_samples.detach().numpy()  # for plotting
        plt.subplot(1, 3, 2)
        self.binormal_contour()
        for i in range(len(z)):
            if len(z) > len(self.colors):
                plt.scatter(z[i][:, 0], z[i][:, 1], s=1)
            else:
                plt.scatter(z[i][:, 0], z[i][:, 1], color=self.colors[i], s=1)
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])

        # map the samples back to the configuration space (use z_samples) and plot 
        plt.subplot(1, 3, 3)
        self.system.plot_FES()
        for i in range(len(z_samples)):
            x_gen, _ = self.model.generator(z_samples[i])
            x_gen = x_gen.detach().numpy()
            if len(z_samples) > len(self.colors):
                plt.scatter(x_gen[:, 0], x_gen[:, 1], s=1)
            else:
                plt.scatter(x_gen[:, 0], x_gen[:, 1], color=self.colors[i], s=1)
        ax = plt.gca()
        ax.set_aspect(0.58)

        fig[0].tight_layout()
        plt.savefig(save_path, dpi=600)

    def config_histogram(self, samples, n_bins=None, weights=None):
        """
        n_bins : int
            Number of bins for the histogram methods. The default is 1/500 of the
            number of samples.
        """
        if n_bins is None:
            n_bins = int(len(samples) / 500)

        counts, bins =np.histogram(samples, bins=n_bins, weights=weights)
        p = counts / np.sum(counts)   # the approximation of probability, px(x)
        centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])  # x values
        
        return p, centers

    def statistical_weights(self, x_samples, z_samples, log_R_zx):
        u_x = self.model.calculate_energy(torch.from_numpy(x_samples), space='configuration')
        u_z = self.model.calculate_energy(z_samples, space='latent')
        w = torch.exp(-u_x + u_z + log_R_zx)
        w = w.detach().numpy()

        return w

    def build_estimator(self, n_z, save_path, optimize=False):
        """
        Builds a kernel density estimator based on the samples generated by a Boltzmann generator.

        """
        z = self.model.prior.sample_n(n_z)
        x, log_R_zx = self.model.generator(z)
        x = x.detach().numpy()
        w = self.statistical_weights(x, z, log_R_zx)

        estimator = density_estimator(x[:, 0], weights=w, optimize=optimize)
        with open(save_path, 'wb') as handle:
            pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return estimator


    def free_energy_profile(self, save_path, n_z1, estimator=None):
        """
        This methods plots the free energy profile based on three ways, including 
        standard histograms, weighted histograms, and kernel density estimation (KDE).

        Parameters
        ----------
        n_z1 : int
            Number of samples drawn from the prior Gaussian distribution in the latent
            space for the histogram methods.
        estimator : sklearn.
        
        """        
        # analytical solution
        if type(self.system).__name__ == 'DoubleWellPotential':
            x_analytical  = np.linspace(-3, 3, 100)
            f_analytical = self.system.get_energy([x_analytical, 0]) - np.log(np.sqrt(np.pi))
            f_analytical -= np.min(f_analytical)

        # Part 1: histogram methods
        # Part 1-1: standard histograms
        z1 = self.model.prior.sample_n(n_z1)
        x1, log_R_zx1 = self.model.generator(z1)
        x1 = x1.detach().numpy()
        
        p1, centers1 = self.config_histogram(x1[:, 0])
        f1 = -np.log(p1)
        f1 -= np.min(f1)
        
        fig = plt.subplots(1, 3, figsize=(18, 4.5))
        plt.subplot(1, 3, 1)
        plt.scatter(centers1, f1, label='Estimated', s=1.5, color='green')
        plt.plot(x_analytical, f_analytical, label='Exact', color='blue')
        plt.xlabel("$x_{1}$")
        plt.ylabel("Free energy $f$")
        plt.title("Free energy based on the standard histogram")
        plt.legend()
        plt.grid()

        # Part 1-2: weighted histogram
        w1 = self.statistical_weights(x1, z1, log_R_zx1)   # first calculate the statistical weights
        p2, centers2 = self.config_histogram(x1[:, 0], weights=w1)
        f2 = -np.log(p2)
        f2 -= np.min(f2)

        plt.subplot(1, 3, 2)
        plt.scatter(centers2, f2, label='Estimated', s=1.5, color='green')
        plt.plot(x_analytical, f_analytical, label='Exact', color='blue')
        plt.xlabel("$x_{1}$")
        plt.ylabel("Free energy $f$")
        plt.title("Free energy based on the weighted histogram")
        plt.legend()
        plt.grid()

        # Part 2: kernel density estimation (KDE)
        if type(self.system).__name__ == 'DoubleWellPotential':
            x_range = np.linspace(-3, 3, 1000)

        log_p3 = estimator.score_samples(x_range[:, None])

        f3 = -log_p3
        f3 -= np.min(f3)

        plt.subplot(1, 3, 3)
        plt.plot(x_range, f3, label='Estimated', color='green')
        plt.plot(x_analytical, f_analytical, label='Exact', color='blue')
        plt.xlabel("$x_{1}$")
        plt.ylabel("Free energy $f$")
        plt.title("Free energy as a function of $x_{1}$")
        plt.legend()
        plt.grid()

        plt.savefig(save_path, dpi=600)
