from pinntorch._dependencies import * 
from pinntorch._sampling import slice_dim

def boundary_location_factory(L = None, R = None, sharpness = 1.0, dim = 0):
    """
    Creates a boundary location function that picks out the boundary location for left and right of the domain

    Parameters
    ----------
    L : float, optional\\
        The left boundary location. If `R` is None, this creates a boundary location function that applies 
        to the left boundary only (default: None).
    R : float, optional\\
        The right boundary location. If `L` is None, this creates a boundary location function that applies 
        to the right boundary only (default: None).
    sharpness : float, optional\\
        The sharpness of the boundary condition. A higher value creates a steeper gradient at the boundary 
        location (default: 1.0).
    dim : int, optional\\
        The input dimension along which to apply the boundary location function (default: 0).

    Returns
    -------
    function\\
        A boundary location function that can be used to apply boundary conditions to the PDE solution.
    """
    if L is None and R is None:
        raise ValueError("At least one boundary location must be set")
    def _bpf_both(data):
        x = slice_dim(data, dim)
        return (1-torch.exp(-sharpness*(x-L)))*(1-torch.exp(+sharpness*(x-R)))
    def _bpf_left(data):
        x = slice_dim(data, dim)
        return (1-torch.exp(-sharpness*(x-L)))
    def _bpf_right(data):
        x = slice_dim(data, dim)
        return (1-torch.exp(+sharpness*(x-R)))
    if L is not None and R is not None:
        if L >= R:
            raise ValueError('left boundary should be left of the right boundary')
        return _bpf_both
    if L is not None:
        return _bpf_left
    return _bpf_right

def LR_dirichlet_factory(L, L_value, R, R_value, dim = 0):
    """
    Creates a dirichlet boundary value function for two different values at the left and right of a specified output dimension.

    Parameters
    ----------
    L : float\\
        The left boundary location.
    L_value : float\\
        The value of the function at the left boundary location.
    R : float\\
        The right boundary location.
    R_value : float\\
        The value of the function at the right boundary location.
    dim : int, optional\\
        The dimension along which to apply the boundary value function (default: 0).

    Returns
    -------
    function
        A boundary value function that can be used to apply dirichlet boundary conditions to the PDE solution.
    """
    if L >= R:
        raise ValueError('left boundary should be left of the right boundary')
    a = (R_value - L_value)/(R - L)
    b = (R*L_value - L*R_value)/(R - L)
    def linear(data : torch.Tensor):
        return a*slice_dim(data, dim)+b
    return linear
