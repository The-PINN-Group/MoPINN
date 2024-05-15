import numpy as np
from steepd_mc import SteepestDescent
from obj_func import Obj

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

obj = Obj()

x_init = np.array([1, 2])
f_opt = sd.steepest(x_init)
print(obj.Fs(f_opt)) # Pareto optimal