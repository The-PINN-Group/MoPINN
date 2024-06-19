from pinntorch._dependencies import * 
from pinntorch._sampling import slice_dim
import torch.nn.init as init
import warnings

def _stack_multi_input(*input_tensors):
    if len(input_tensors) > 1:
        return torch.cat(input_tensors, dim=1)
    else:
        return input_tensors[0]

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
        self.initialize_weights()
    
    def _set_activation_functions(self, num_hidden, dim_hidden, activation):
        if activation == 'tanh':
            self.activations = nn.ModuleList([nn.Tanh() for _ in range(num_hidden)])
        elif activation == 'sin':
            self.activations = [torch.sin for _ in range(num_hidden)]
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

    def forward(self, *input_data):
        input_data = _stack_multi_input(*input_data)
        # Calculate the nn output
        out = self.activations[0](self.layer_in(input_data))
        for i, layer in enumerate(self.middle_layers):
            out = self.activations[i+1](layer(out))
        output = self.layer_out(out)
        
        # split the output along the dimensions of the output
        output = [tens.reshape(-1, 1) for tens in torch.unbind(output, 1)]
        return output

     
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

