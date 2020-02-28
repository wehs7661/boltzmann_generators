import numpy as np


class MuellerPotential:
    def __init__(self, *args, alpha = 0.1):
        self._alpha = alpha
        if len(args) == 1:
            # Matrix implementation
            param_matrix = np.array(args[0])
            assert param_matrix.shape[0] == 6
            self._a = []
            self._b = []
            self._c = []
            self._A = []
            self._x = []
            self._y = []
            for i in range(param_matrix.shape[1]):
                self._a.append(param_matrix[0,i])
                self._b.append(param_matrix[1,i])
                self._c.append(param_matrix[2,i])
                self._A.append(param_matrix[3,i])
                self._x.append(param_matrix[4,i])
                self._y.append(param_matrix[5,i])
            self._a = np.array(self._a)
            self._b = np.array(self._b)
            self._c = np.array(self._c)
            self._A = np.array(self._A)
            self._x = np.array(self._x)
            self._y = np.array(self._y)
            



        if len(args) == 6:
            # Vector implementation
            self._a = np.array(args[0])
            self._b = np.array(args[1])
            self._c = np.array(args[2])
            self._A = np.array(args[3])
            self._x = np.array(args[4])
            self._y = np.array(args[5])

        else:
            raise NotImplementedError("MullerPotential objects can only take 1 or 6 arguments!")

    def __call__(self, x, y):
        v_energy = np.vectorize(self._get_energy)
        return(v_energy(x, y))
    
    def _get_energy(self, x, y):
        # print(self._A * np.exp(self._a * np.power(x - self._x, 2) + self._b * (x - self._x) * (y - self._y)  + self._c * np.power(y - self._y, 2)))
        E = self._alpha * np.sum( self._A * np.exp(self._a * np.power(x - self._x, 2) + self._b * (x - self._x) * (y - self._y)  + self._c * np.power(y - self._y, 2)))
        return(E)

