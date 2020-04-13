from abc import ABC, abstractmethod

class Subject(ABC):
    def __init__(self):
        self.observers = []

    @abstractmethod
    def registerObserver(self, Observer):
        pass

    @abstractmethod
    def removeObserver(self, Observer):
        pass
    
    @abstractmethod
    def notifyObservers(self):
        pass



class Observer(ABC):
    @abstractmethod
    def update(self):
        pass


class CoordinateLogger(Observer):
    def __init__(self, system, freq):
        self.system = system
        self.coordinates = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            self.coordinates.append(self.system.get_coordinates())
    

class EnergyLogger(Observer):
    def __init__(self, system, freq):
        self.system = system
        self.K = []
        self.U = []
        self.H = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            h, u, k = self.system.integrator.calculate_energy()
            self.K.append(k)
            self.U.append(u)
            self.H.append(h)

class PressureLogger(Observer):
    def __init__(self, system, freq):
        self.system = system
        self.P = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            p = self.system.integrator.calculate_pressure()
            self.P.append(p)



class VelocityLogger(Observer):
    def __init__(self, system, freq):
        self.system = system
        self.vels = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            v = self.system.get_velocities()
            self.vels.append(v)

class TemperatureLogger(Observer):
    def __init__(self, system, freq):
        self.system = system
        self.T = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            T = self.system.integrator.calculate_temperature()
            self.T.append(T)