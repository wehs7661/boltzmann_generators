import numpy as np
import integrators
import thermostats
import itertools
import potentials
import torch
import data_logging
import boundary_conditions
class Particle:
    def __init__(self, potential, loc, vel = None, mass = 1):
        self.potential = potential
        self.loc = loc
        self.dim = len(loc)
        self.mass = mass
        self.vel = vel
        self.force = None
        self.neighbors = []
        self.prev_loc = None # <---- maybe implement as a particle decorator
        self.cell = None # <---- same thing here
        self.index = None

class Bond:
    def __init__(self, potential, particle_1, particle_2, particle_interactions = False, bc = None):
        self.particle_1 = particle_1
        self.particle_2 = particle_2
        self.potential = potential
        self.particle_interactions = particle_interactions
        self.bc = bc
        if bc is None:
            self.bc = boundary_conditions.NoBoundaryConditions(None)


    def get_distance(self):
        r_ij = self.bc(self.particle_1.loc - self.particle_2.loc)
        return(np.sqrt(np.dot(r_ij,r_ij)))

    def get_rij(self):
        return self.bc(self.particle_1.loc - self.particle_2.loc)
    
    def get_energy(self):
        d_ij = self.get_distance()
        return self.potential(d_ij)

    def get_force(self):
        d_ij = self.get_distance()
        f = -self.potential.derivative(d_ij)
        r_ij_hat = self.bc(self.particle_1.loc - self.particle_2.loc) / d_ij
        f_ij = f * r_ij_hat
        return f_ij


class SystemFactory:
    def __init__(self):
        self.placement = {
                          "lattice" : self.lattice_placement,
                          "random" : self.random_placement,
                         }
        
        self.potentials = {
                           "WCA" : potentials.WCAPotential,
                           "LJ" : potentials.LJpotential,
                           "LJ-rep" : potentials.LJRepulsion
                          }

        self.central_potentials = {
                                   "harmonic" : potentials.HarmonicPotential,
                                   "double_well" : potentials.DoubleWellPotential,
                                   "mueller" : potentials.MuellerPotential,
                                   "harmonic_box" : potentials.HarmonicBox
                                  }

        self.boundary_conditions = {
                                    "periodic" : boundary_conditions.PeriodicBoundaryConditions,
                                    "none" : boundary_conditions.NoBoundaryConditions
                                   }


    def build_system(self, 
                    dim = 2,
                    T = 1,
                    rho = 0.6,
                    N = 20,
                    mass = 1,
                    placement_method = "lattice",
                    boundary_conditions = "periodic",
                    central_potential = None,
                    potential = "WCA",
                    **args):
        if len(args) == 0:
            args = {"sigma" : 1, "epsilon" : 1}   
        box = (N / rho) ** (1/dim) * np.ones(dim)
        coords = self.placement[placement_method](N, box)
        system = System(box=box)
        system.bc = self.boundary_conditions[boundary_conditions](system)
        vels = np.random.normal(scale = np.sqrt(T/mass), size = (N, dim))
        for coord, vel in zip(coords, vels):
            system.add_particle(Particle(self.potentials[potential](**args), coord, vel = vel, mass = mass))
        self.get_dofs(system, placement_method)
        return system

    def get_dofs(self, system, placement_method):
        system.dof = system.dim * len(system.particles)
        if placement_method == "periodic":
            system.dof -= system.dim
        
    def add_central_potential(self, system, central_potential, **kwargs):
        system.central_potential = self.central_potentials[central_potential](**kwargs)
    
    def lattice_placement(self, n, box):
        r_min = -box / 2
        r_max = - r_min
        dim = len(box)
        d = box / (n)**(1/dim)
        pos = np.linspace(r_min + 0.5 * d, r_max - 0.5 * d, np.ceil(n**(1/dim)))
        coords = np.array(list(itertools.product(pos, repeat=dim)))
        coords = np.array([np.diagonal(coord) for coord in coords])
        return coords

    def random_placement(self, n, box, d_min = None, center = None):
        dim = len(box)
        if d_min is None:
            d_min = 0.1 * np.average(box)
            print(d_min)
        if center is None:
            center = np.zeros(dim)
        coords = np.zeros((n, dim))
        coords[0] = (np.random.random(dim) - 0.5)*box + center
        i = 0
        while i <= n - 1:
            prop = (np.random.random(dim) - 0.5) * box + center
            if np.any(np.linalg.norm(prop - coords, axis=1) > d_min):
                coords[i] = prop
                i += 1
        print(coords)
        return coords


class System(data_logging.Subject):
    def __init__(self, box = np.array([1, 1]), dim = 2):
        super().__init__()
        self.particles = []
        self.bonds = []
        self.box = box
        self.int_fact = integrators.IntegratorFactory(self)
        self.integrator = None
        self.central_potential = None
        self.bc = boundary_conditions.NoBoundaryConditions(self)
        self.dim = dim
        self.dof = None

    # Observer functions

    def registerObserver(self, Observer):
        self.observers.append(Observer)

    def removeObserver(self, Observer):
        self.observers.remove(Observer)

    def notifyObservers(self, steps):
        for obs in self.observers:
            obs.update(steps)

    # Get type of integrator
    def get_integrator(self, integrator_name, **kwargs):
        self.integrator = self.int_fact.get_integrator(integrator_name, **kwargs)

    def get_thermostat(self, thermostat_name, T, **kwargs):
        self.integrator.get_thermostat(thermostat_name, T, **kwargs)

    # Run Simulation
    def run(self, steps):
        for step in range(steps):
            self.integrator.integrate()
            self.notifyObservers(step + 1)
        

    def add_particle(self, particle):
        if particle.force is None:
            particle.force = 0
        if particle.vel is None:
            particle.vel = 0
        self.particles.append(particle)


    # def __init__(self, potential, particle_1, particle_2, particle_interactions = False, bc = None):

    def add_bond(self, potential, particle_i, particle_j):
        self.bonds.append(Bond(potential, particle_i, particle_j, bc = self.bc))



    # Boundary Conditions
    def apply_bc(self, coords):
        return self.bc(coords)

    # Get/Set system states
    def get_velocities(self):
        vels = np.zeros((len(self.particles), self.dim))
        for i in range(len(self.particles)):
            vels[i, :] = self.particles[i].vel
        return vels

    def set_velocities(self, vels, indices = []):
        if  len(indices) == 0:
            indices = list(range(len(self.particles)))
        
        for i in range(len(indices)):
            # print("Updating Particle", i, "velocity")
            # print(vels[i,:])
            self.particles[indices[i]].vel = vels[i, :]

    def get_coordinates(self):
        # print(self.particles[0].loc.dtype)
        if self.particles[0].loc.dtype == np.float:
            coords = np.zeros((len(self.particles), self.dim))
        elif self.particles[0].loc.dtype == torch.float:
            coords = torch.zeros((len(self.particles), self.dim))
        else:
            print("Non-compatible data type! Please use np.ndarrays or torch.tensors")
        for i in range(len(self.particles)):
            coords[i ,:] = self.particles[i].loc
        return coords

    def set_coordinates(self, coords, indices = []):
        if  len(indices) == 0:
            indices = range(len(self.particles))
        
        for i in indices:
            self.particles[i].loc = coords[i]

    def get_forces(self):
        forces = np.zeros((len(self.particles), self.dim))
        for i in range(len(self.particles)):
            forces[i ,:] = self.particles[i].force
            # print(forces)
        return forces

    def set_forces(self, forces, indices = []):
        if  len(indices) == 0:
            indices = range(len(self.particles))
        
        for i in indices:
            # print(forces[i, :])
            self.particles[i].force = forces[i, :]

    def get_masses(self):
        masses = np.zeros(len(self.particles))
        for i in range(len(self.particles)):
            masses[i] = self.particles[i].mass
        return masses

    def set_masses(self, masses, indices = []):
        if  len(indices) == 0:
            indices = range(len(self.particles))
        
        for i in indices:
            self.particles[i].masses = masses[i]

    def get_energy(self, coords = None, energy_type = "potential"):
        e_type = {
                  "total" : 0,
                  "potential" : 1,
                  "kinetic" : 2
                 }
        if coords is None:
            H, U, K = self.integrator.calculate_energy()
            return(H, U, K)
        else:
            old_coords = self.get_coordinates()
            self.set_coordinates(coords.reshape(old_coords.shape))
            energy = self.integrator.calculate_energy()[e_type[energy_type]]
            self.set_coordinates(old_coords)
            return(energy)

            
        

