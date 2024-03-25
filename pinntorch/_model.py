from pinntorch._dependencies import * 
from pinntorch._sampling import slice_dim
import torch.nn.init as init
import warnings

class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super(SimpleWrapper, self).__init__()
        self.base_model = model
    
    def initialize_weights(self, seed=None):
        if hasattr(self.base_model, 'initialize_weights'):
            self.base_model.initialize_weights(seed)
        elif hasattr(self.base_model, 'reset_parameters'):
            self.base_model.reset_parameters()

    def forward(self, *input_data):
        x = self.base_model(_stack_multi_input(*input_data))
        output = [tens.reshape(-1, 1) for tens in torch.unbind(x, 1)]
        return output

class DirichletWrapper(SimpleWrapper):
    def __init__(self, model):
        SimpleWrapper.__init__(self, model)
        self.fixed_dirichlet = []
    
    def add_fixed_dirichlet(self, output_index, location_function, value_function):
        """
        Adds a fixed Dirichlet boundary condition to the set of boundary conditions for the PDE solution.

        Parameters
        ----------
        output_index : int\\
            The index of the output that the boundary condition applies to.
        location_function : function\\
            A function that defines the location of the boundary condition along the given output index.
        value_function : function\\
            A function that defines the value of the boundary condition at the given location.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a boundary condition already exists for the given output index.
        """
        for bc in self.fixed_dirichlet:
            if bc[0] == output_index:
                raise ValueError(f"A boundary condition already exists for output index {output_index}")
        self.fixed_dirichlet.append((output_index, location_function, value_function))

    def forward(self, *input_data):
        x = self.base_model(_stack_multi_input(*input_data))
        output = [tens.reshape(-1, 1) for tens in torch.unbind(x, 1)]
        for dirichlet in self.fixed_dirichlet:
            if callable(dirichlet[2]):
                boundary_value = dirichlet[2](input_data)
            else:
                boundary_value = dirichlet[2]
            output[dirichlet[0]] = dirichlet[1](input_data)*output[dirichlet[0]]+boundary_value

        return output
    
class RescaleWrapper(SimpleWrapper):
    def __init__(self, model, original_domains):
        SimpleWrapper.__init__(self, model)
        self.original_domains = original_domains
        
    def rescale_input(self, input_data):
        rescaled_data = input_data.clone()
        for i, (x_min, x_max) in enumerate(self.original_domains):
            a, b = self.domain_ranges[i]
            rescaled_data[:, i] = (input_data[:, i] - x_min) / (x_max - x_min)
        return rescaled_data

    def forward(self, *input_data):
        # Rescale the input data for each dimension to the desired domain.
        rescaled_input_data = self.rescale_input(_stack_multi_input(*input_data))
        
        x = self.base_model(rescaled_input_data)
        output = [tens.reshape(-1, 1) for tens in torch.unbind(x, 1)]
        return output


def get_activation_function(activation_function):
    if callable(activation_function):
        return activation_function
    elif activation_function == 'tanh':
        return nn.Tanh()
    elif activation_function == 'relu':
        return nn.ReLU()
    elif activation_function == 'elu':
        return nn.ELU()
    elif activation_function == 'sin':
        return Sin()
    elif activation_function in ['none', 'identity', '']:
        return None
    else:
        raise ValueError(f"Activation function is not supported.")

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def _stack_multi_input(*input_tensors):
    if len(input_tensors) > 1:
        return torch.cat(input_tensors, dim=1)
    else:
        return input_tensors[0]

class MLP(nn.Module):
    def __init__(self, layer_sizes, layer_activations):
        super(MLP, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("The number of layers should be at least 2.")
        
        if len(layer_sizes) - 1 != len(layer_activations):
            raise ValueError("The number of layers (except the first) and activation functions must match.")
        
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if i < len(layer_sizes):
                activation = get_activation_function(layer_activations[i - 1])
                if activation is not None:
                    layers.append(activation)

        self.model = nn.Sequential(*layers)

        self.initialize_weights()
    
    def initialize_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, input_tensors):
        return self.model(input_tensors)


class PIMLP(nn.Module):
    """
    A simple neural network that approximates the solution of a differential equation.

    This neural network is used as a universal function approximator in the context of physics-informed neural networks (PINNs).

    Parameters
    ----------
    dim_input : int\\
        The dimensionality of the input to the neural network.
    num_hidden : int\\
        The number of hidden layers in the neural network.
    dim_hidden : int\\
        The dimensionality of the hidden layers.
    dim_output : int\\
        The dimensionality of the output layer.
    activation : str, optional\\
        The activation function to use in the neural network (default: 'tanh'). Options: 'tanh', 'periodic'\\
    """
    def __init__(self, layer_sizes, layer_activations):
        super(PIMLP, self).__init__()
        self.model = SimpleWrapper(MLP(layer_sizes, layer_activations))

    def initialize_weights(self, seed=None):
        self.model.initialize_weights(seed)

    def forward(self, *input_data):
        return self.model(*input_data)

class PINN(nn.Module):
    """
    A simple neural network that approximates the solution of a differential equation.

    This neural network is used as a universal function approximator in the context of physics-informed neural networks (PINNs).

    Parameters
    ----------
    dim_input : int\\
        The dimensionality of the input to the neural network.
    num_hidden : int\\
        The number of hidden layers in the neural network.
    dim_hidden : int\\
        The dimensionality of the hidden layers.
    dim_output : int\\
        The dimensionality of the output layer.
    activation : str, optional\\
        The activation function to use in the neural network (default: 'tanh'). Options: 'tanh', 'periodic'\\
    """
    def __init__(self, dim_input:int, num_hidden: int, dim_hidden: int, dim_output : int, activation='tanh'):
        super().__init__()
        warnings.warn("The PINN class is deprecated and only exists for backwards compatibility, please use other classes like \"PIMLP\"", DeprecationWarning)
        self.layer_in = nn.Linear(dim_input, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, dim_output)
        self.middle_layers = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden - 1)])
        self._set_activation_functions(num_hidden, dim_hidden, activation) 
        self.fixed_dirichlet = []
        self.initialize_weights()
    
    def _set_activation_functions(self, num_hidden, dim_hidden, activation):
        if activation == 'tanh':
            self.activations = nn.ModuleList([nn.Tanh() for _ in range(num_hidden)])
        elif activation == 'sin':
            self.activations = [torch.sin for _ in range(num_hidden)]
        elif activation == 'fourier':
            self.activations = nn.ModuleList([FourierActivation(dim_hidden) for _ in range(num_hidden)])
        else:
            assert False, 'invalid activation specification: '+str(activation)
    
    def initialize_weights(self, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)

        #for activation in self.activations:
        #    if isinstance(activation, PeriodicActivation):
        #        init.uniform_(activation.frequencies, 0.5, 1.5)#normal_(activation.freq, 0, 1.0)
                
        #for activation in self.activations:
        #    if isinstance(activation, FourierActivation):
        #        init.normal_(activation.frequencies, 0, 1.0)

    
    def to_dict(self):
        dict_repr = {
            '__class__': self.__class__.__name__,
            'dim_input': self.layer_in.in_features,
            'num_hidden': len(self.middle_layers) + 1,
            'dim_hidden': self.middle_layers[0].out_features,
            'dim_output': self.layer_out.out_features,
            'activation': self.activations[0].__class__.__name__,
            'fixed_dirichlet': self.fixed_dirichlet,
            'weights_dict': self.state_dict()
        }
        return dict_repr
    
    def add_fixed_dirichlet(self, output_index, location_function, value_function):
        """
        Adds a fixed Dirichlet boundary condition to the set of boundary conditions for the PDE solution.

        Parameters
        ----------
        output_index : int\\
            The index of the output that the boundary condition applies to.
        location_function : function\\
            A function that defines the location of the boundary condition along the given output index.
        value_function : function\\
            A function that defines the value of the boundary condition at the given location.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a boundary condition already exists for the given output index.
        """
        for bc in self.fixed_dirichlet:
            if bc[0] == output_index:
                raise ValueError(f"A boundary condition already exists for output index {output_index}")
        self.fixed_dirichlet.append((output_index, location_function, value_function))

    def forward(self, *input_data):
        input_data = _stack_multi_input(*input_data)
        # Calculate the nn output
        out = self.activations[0](self.layer_in(input_data))
        for i, layer in enumerate(self.middle_layers):
            out = self.activations[i+1](layer(out))
        output = self.layer_out(out)
        
        # split the output along the dimensions of the output
        output = [tens.reshape(-1, 1) for tens in torch.unbind(output, 1)]
        
        #print('output', output)
        #print('input_data', input_data)
        #print('dirichlet', self.fixed_dirichlet)
        # apply fixed dirichlet boundaries
        for dirichlet in self.fixed_dirichlet:
            if callable(dirichlet[2]):
                boundary_value = dirichlet[2](input_data)
            else:
                boundary_value = dirichlet[2]
            output[dirichlet[0]] = dirichlet[1](input_data)*output[dirichlet[0]]+boundary_value
        return output



#class PeriodicActivation(nn.Module):
#    def __init__(self, input_size):
#        super().__init__()
#        self.input_size = input_size
#        self.initialize_weights()
#    
#    def initialize_weights(self, seed=None):
#        if seed is not None:
#            torch.manual_seed(seed)
#        self.frequencies = nn.Parameter(torch.rand(self.input_size)+0.5)
#    
#    def forward(self, x):
#        frequencies = self.frequencies.view(1, -1)
#        return torch.sin(x * frequencies)
    
    
class FourierActivation(nn.Module):
    def __init__(self, input_size, scale = 1.0):
        super().__init__()
        self.scale = scale
        self.input_size = input_size
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.frequencies = nn.Parameter(torch.randn(self.input_size) * self.scale)

    def forward(self, x):
        frequencies = self.frequencies.view(1, -1)
        return torch.sin(x * frequencies)

     
def f(model: nn.Module, *input_data : torch.Tensor, of : int = 'all'):
    """
    Evaluate the output of a neural network for a given set of inputs.

    Parameters
    ----------
    model : PINN\\
        The neural network model to evaluate.
    input_data : torch.Tensor\\
        The input data for the neural network.
    of : int or str, optional\\
        The index of the output to return or 'all' for all outputs (default: 'all').

    Returns
    -------
    torch.Tensor\\
        The output of the neural network for the given input data.\\
    """
    output = model(*input_data)
    if type(output) is tuple:
        return output
    if len(output) == 1:
        return output[0]
    if of == 'all':
        return output
    return output[of]

def df(model: nn.Module, *input_data : torch.Tensor, of : int = 0, wrt = 0, order: int = 1) -> torch.Tensor:
    """
    Compute the partial derivative of a function represented by a neural network with respect to a specified input.

    Parameters
    ----------
    model : PINN\\
        The neural network model representing the function.
    input_data : torch.Tensor\\
        The input data for the neural network.
    of : int\\
        The index of the output to take the derivative of (default: 0).
    wrt : int or torch.Tensor\\
        'with respect to' => The index of the input to take the derivative with respect to, or the tensor itself (default: 0).
    order : int\\
        The order of the derivative to compute (default: 1).

    Returns
    -------
    torch.Tensor\\
        The partial derivative of the function with respect to the specified input.\\
    """
    df_of = f(model, *input_data, of=of)
 
    if type(wrt) == int:
        respect_to = input_data[wrt]
    else:
        respect_to = wrt

    for _ in range(order):
        df_of = torch.autograd.grad(
            df_of,
            respect_to,
            grad_outputs=torch.ones_like(respect_to),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_of

