import numpy as np
from scipy.optimize import fmin
#from nptyping import NDArray
from typing import Any, Callable
from dataclasses import dataclass
from obj_func import Obj

@dataclass
class SteepestDescent:
    ndim: int
    nu: float
    sigma: float
    eps: float

    def grad(self, f, x, h=1e-4):
        g = np.zeros_like(x)
        for i in range(self.ndim):
            tmp = x[i]
            x[i] = tmp + h
            yr = f(x)
            x[i] = tmp - h
            yl = f(x)
            g[i] = (yr - yl) / (2 * h)
            x[i] = tmp
            #print("the value of yr is", yr)
            #print("the shape of yr is ", yr.shape)
        #print("shape of g:", g.shape)

        return g

        

    def nabla_F(self, x):
        obj = Obj()
        F = obj.Fss()    # why is Fss here and not Fs
        nabla_F = np.zeros((len(F), self.ndim)) # (m, n) dimensional matrix
        #print("shape if jacobian initial:", nabla_F.shape)
        for i, f in enumerate(F):
            nabla_F[i] = self.grad(F[i], x)
        return nabla_F

    def phi(self, d, x):
        nabla_F = self.nabla_F(x)
        return max(np.dot(nabla_F, d)) + 0.5 * np.linalg.norm(d) ** 2

    def theta(self, d, x):
        return self.phi(d, x) + 0.5 * np.linalg.norm(d) ** 2

    def armijo(self, d, x):
        power = 0
        obj = Obj()
        t = pow(self.nu, power)
        Fl = np.array(obj.Fs(x + t * d))
        Fr = np.array(obj.Fs(x))
        Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        while np.all(Fl > Fr + Re):
            t *= self.nu
            Fl = np.array(obj.Fs(x + t * d))
            Fr = np.array(obj.Fs(x))
            Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        return t
    
    def steepest(self, x):
        d = np.array(fmin(self.phi, x, args=(x, )))
        th = self.theta(d, x)
        #print("th", th)
        print(abs(th) > self.eps)
        i = 0
        while abs(th) > self.eps:
            print("i:", i)
    
            i = i + 1

            t = self.armijo(d, x)

            #print("t:", t)

            x = x + t * d
            d = np.array(fmin(self.phi, x, args=(x, )))
            th = self.theta(d, x)
        return x
