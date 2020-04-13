from abc import ABC, abstractmethod
import numpy as np

class Thermostat(ABC):
    def __init__(self, integrator, T):
        self.integrator = integrator
        self.T = T
        self.count = 0
    
    @abstractmethod
    def apply(self):
        pass


class ThermostatFactory():
    def __init__(self, integrator):
        self.integrator = integrator
        self.thermostats = {
                            "anderson" : AndersonThermostat,
                           }

    def get_thermostat(self, thermostat_name, T, **kwargs):
        if thermostat_name in self.thermostats.keys():
            return(self.thermostats[thermostat_name](self.integrator, T, **kwargs))
        else:
            raise NotImplementedError(thermostat_name + "is not a valid thermostat!")


class AndersonThermostat(Thermostat):
    def __init__(self, integrator, T, colisions =  0.01, freq = 50):
        super().__init__(integrator, T)
        self.freq = freq
        self.colisions = colisions

    def apply(self):
        self.count += 1
        if self.count % self.freq == 0:
            self.count = 0
            for particle in self.integrator.system.particles:
                if self.integrator.dt * self.freq * self.colisions > \
                     np.random.uniform():
                    particle.vel = np.random.normal(scale = np.sqrt(self.T/particle.mass),
                                                    size = particle.vel.shape)



            
    