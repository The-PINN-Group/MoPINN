#@title Here we define the objective functions that we want to optimize. 

import numpy as np
#from nptyping import NDArray   # I am not using it as it was giving some syntax errors. I still have to figure out how ot use it  properly 
from typing import Any, Callable

class Obj:

    def f(self, x):
        return x[0]**2 + 3 * (x[1] - 1)**2

    def g(self, x):
        return 2 * (x[0] - 1)**2 + x[1]**2

    def Fs(self, x):
        return np.array([self.f(x), self.g(x)])

    def Fss(self):
        return np.array([self.f, self.g])