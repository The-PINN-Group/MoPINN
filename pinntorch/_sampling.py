from pinntorch._dependencies import * 

def generate_sample(point_counts, span, requires_grad=True):
    """
    Generates a tensor of random samples within a given range.
    
    
    Parameters
    ----------
    point_counts : int\\
        The number of random samples to generate.
    span : Tuple[float, float]\\
        The range to sample from represented by a tuple of (start, end).
    requires_grad : bool, optional\\
        Flag indicating whether to track the tensor's gradient (default: True).

    Returns
    -------
    torch.Tensor\\
        A tensor of random samples within the given range.\\
    """
    return torch.tensor((span[1]-span[0])*np.random.rand(point_counts,1) + span[0], dtype = torch.float32, requires_grad=requires_grad)
       
def generate_grid(point_counts, domain, domain_span=None, requires_grad=True):
    """Generates a grid of points in the specified domain with the specified number of points for each dimension.

    Parameters
    ----------
    point_counts : tuple or int\\
        the number of points for each dimension. If int is provided, it is assumed that all dimensions have the same number of points.
    domain : tuple or string\\
        The domain of the grid. If tuple is provided, it must contain a tuple of (min, max) values for each dimension. If string 'auto' is provided, the domain span for each dimension is set to the value specified in domain_span.
    domain_span : tuple, optional\\
        If domain is 'auto', this specifies the min and max values of all dimensions.
    requires_grad : bool, optional\\
        If True (default), the returned tensors will have gradient computation enabled.

    Returns
    -------
    list of torch.Tensor\\
    A list of 1D tensors, each representing the values of one dimension of the grid. If the grid has only one dimension, a single tensor is returned instead of a list.

    -----

    Examples
    --------
    >>> generate_grid(point_counts=(3, 3), domain=((0.0, 2.0), (0.0, 5.0)))
    [   tensor([[0.], [0.], [0.], [1.], [1.], [1.], [2.], [2.], [2.]]), 
        tensor([[0.0000], [2.5000], [5.0000], [0.0000], [2.5000], [5.0000], [0.0000], [2.5000], [5.0000]])
    ]
    
    >>> generate_grid(point_counts=5, domain='auto', domain_span=(0.0, 1.0))
    tensor([[0.0000],
            [0.2500],
            [0.5000],
            [0.7500],
            [1.0000]])\
    """

    # Check if point_counts and domain are single integers or tuples and convert to lists
    if isinstance(point_counts, int):
        point_counts = [point_counts]
        domain = [domain]

    # Check if domain is set to 'auto' and use domain_span if it is
    if domain == 'auto' or type(domain) is list and 'auto' in domain:
        assert domain_span is not None, "If domain is set to 'auto', domain_span cannot be None"
        domain = [domain_span for _ in point_counts]

    # Check that point_counts and domain have the same length
    assert len(point_counts) == np.prod(np.array(domain).shape)/2, "point_counts and domain must have the same length"

    # Generate raw tensors using torch.linspace()
    raw = [torch.linspace(start=domain[i][0], end=domain[i][1], steps=point_counts[i], requires_grad=requires_grad) for i in range(len(point_counts))]

    # Generate grid
    if len(domain) == 1:
        # If only one dimension, return a flattened tensor
        grid = torch.meshgrid(raw, indexing="ij")[0].flatten().reshape(-1, 1)
    else:
        # If multiple dimensions, return a list of flattened tensors
        grid = [g.flatten().reshape(-1, 1) for g in torch.meshgrid(raw, indexing="ij")]
    
    return grid


#def generate_offset_grid(point_counts, domain, domain_span=None, offset=0.0, requires_grad=True):
#    grid = generate_grid(point_counts, domain, domain_span, requires_grad)
    

def at(value):
    """
    Creates a one-dimensional PyTorch tensor with a single element and sets its value to the specified value.

    Parameters
    ----------
    value : float\\
        The value to set the tensor to.

    Returns
    -------
    torch.Tensor\\
        A PyTorch tensor with a single element and the specified value, with the `requires_grad` attribute set to `True`.
    """
    return torch.tensor([[value]], requires_grad=True)


def slice_dim(data: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Slice a tensor along a specified dimension to extract a particular column or dimension of the data.

    Parameters
    ----------
    data : torch.Tensor\\
        The input tensor to slice.
    dim : int\\
        The index of the dimension along which to slice the tensor.

    Returns
    -------
    torch.Tensor\\
        The sliced tensor, with shape (N, 1), where N is the number of rows in the original tensor.

    Examples
    --------
    >>> data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> slice_dim(data, 1)
    tensor([[2],
            [5],
            [8]])
    """
    return data[:,dim].reshape(-1, 1)


def unique_excluding(data, exclude_value=torch.nan):
    """
    Returns a PyTorch tensor containing the unique values in the input data, after excluding a specific value if provided.

    Parameters
    ----------
    data : torch.Tensor\\
        A PyTorch tensor containing the input data.
    exclude_value : float, optional\\
        The value to exclude from the input data. Defaults to `torch.nan`, which doesn't exclude any value.

    Returns
    -------
    torch.Tensor\\
        A PyTorch tensor containing the unique values in the input data, after excluding the specified value (if provided).
        The returned tensor has the `requires_grad` attribute set to `True`.
    """
    data = data[data != exclude_value]
    raw = torch.unique(data).reshape(-1, 1).detach().cpu().numpy()
    raw = torch.Tensor(raw)
    raw.requires_grad = True
    return raw


def fill_like(data, value):
    """
    Returns a PyTorch tensor with the same shape as the input data, where all elements have the specified value.

    Parameters
    ----------
    data : torch.Tensor\\
        A PyTorch tensor containing the input data, whose shape will be copied.
    value : float\\
        The value to set all elements in the returned tensor to.

    Returns
    -------
    torch.Tensor\\
        A PyTorch tensor with the same shape as the input data, where all elements have the specified value.
        The returned tensor has the `requires_grad` attribute set to `True`.
    """
    return torch.ones_like(data, requires_grad=True) * value

def normalize_tensor(tensor, domain):
    """
    Normalizes the input tensor(s) by scaling its values to the range [0,1] based on the specified domain.

    Parameters
    ----------
    tensor : torch.Tensor or list of torch.Tensor\\
        The input tensor(s) to normalize.
    domain : tuple or list of tuple\\
        A tuple (min, max) specifying the minimum and maximum values of the domain to normalize the tensor to.
        If a list is provided, it must have the same length as the input tensor list.

    Returns
    -------
    torch.Tensor or list of torch.Tensor\\
        A normalized PyTorch tensor or list of PyTorch tensors with values in the range [0,1].
        The returned tensor(s) have the same shape and data type as the input tensor(s).

    Raises
    ------
    ValueError\\
        If the input tensor(s) or domain are not compatible.
    """
    if type(tensor) is list:
        if len(tensor) != np.prod(np.array(domain).shape)/2:
            raise ValueError('length of tensor input and length of domain must match.')
        return [_normalize_tensor(actual_tensor, domain[i]) for i,actual_tensor in enumerate(tensor)]
    return _normalize_tensor(tensor, domain)

def _normalize_tensor(tensor: torch.Tensor, domain: tuple) -> torch.Tensor:
    dim_min, dim_max = domain 
    tensor_normalized = (tensor - dim_min) / (dim_max - dim_min)
    return tensor_normalized
