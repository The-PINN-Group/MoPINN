from pinntorch import *
from functools import partial
from tqdm import trange
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from time import time
import argparse
torch.manual_seed(0)

#Passing the argument for the number of neurons. 
#parser = argparse.ArgumentParser(description='Pareto front with NSGA2.')
#parser.add_argument('--n_value', type=float, required=True, help='n_neurons: the # of neurons')
#args = parser.parse_args()
n_neurons = 5#int(args.n_value)

K = 5.0
def exact_solution_log(x):
    """Returns a torch tensor given an input tensor x"""
    return 1/(1+torch.exp(-torch.Tensor(K*x)))


# exact solution in NumPy: This one is needed for the loss function becasue somehow the tensor form does not work as of now.
def exact_solution_log_np(x):
    return 1/(1+np.exp(-K*x))


# TODO specify the types of input. here: x and t. May deviate for other problems
def physics_loss(nn: PINN, x: torch.Tensor = None) -> torch.float:
    """Compute the full physics loss function as interior loss + boundary loss

    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """
    # TODO: define PDE loss
    pde_loss_pre = df(nn, x) - K*f(nn, x)*(1 - f(nn, x))
    pde_loss = pde_loss_pre.pow(2).mean()
    
    # TODO: define conditional losses (initial + boundary)
    boundary_loss_right_pre = (f(nn, at(+1.0)) - exact_solution_log_np(+1)) 
    boundary_loss_right = boundary_loss_right_pre.pow(2).mean()

    # TODO: combine all losses
    final_loss = pde_loss + boundary_loss_right
    
    return final_loss

def create_noisy_data(x, mean, std_dev, seed_ = 42, func = 'log'):

    """adds gaussian noise to the data"""

    if func == 'log':
        exact_soln = exact_solution_log(x)

    torch.manual_seed(seed_)

    return exact_soln + torch.randn(exact_soln.size())*std_dev + mean 

def data_loss(nn: PINN, data: torch.Tensor = None, x: torch.Tensor = None) -> torch.float:

    """Compute the data loss"""

    u_n = f(nn, x) # evaluating the model

    # MSE loss 
    diff = u_n - data    # data = u_exact + gaussian noise     
    loss = diff.pow(2).mean()
    return loss

N = 20  # number of collocation points
training_points = generate_grid((N), (-1.0,1.0))
test_points = generate_grid((50), (-1.0,1.0))
plot_points = generate_grid((100), domain=(-1.0,1.0))



learnable_func = 'log'   #$ function ('log'/'cos') that we want to learn with data
alpha = 1.0 # deciding the weight of the data loss. 

mean = 0.0
std = 0.1
data_noisy = create_noisy_data(training_points, mean, std, func=learnable_func)
#n_neurons = 11  # n_neurons = 7, 8, 9, 11 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.layer3 = nn.Linear(n_neurons, n_neurons)
        self.layer4 = nn.Linear(n_neurons, 1)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = self.layer4(x)
        return x
    
    def get_learnable_params(self,):
        model_l_params = self.state_dict()
        dict_params = {}
        for name, l_param in model_l_params.items():
            dict_params[name] = l_param
            
        return dict_params

    # Function to flatten the parameters
def get_num_elmts(dict_params):
    
    N = 0
    for name in list(dict_params.keys()):
        N +=  dict_params[name].numel()    
    # Flatten the list of tensors into a single list of elements
    flattened_list = [element.item() for tensor in list(dict_params.values()) for element in tensor.view(-1)]

    return N, flattened_list

#model = PINN(1, 3, 9, 1)
model = NeuralNetwork()
dict_params = model.get_learnable_params()
N, initial_params = get_num_elmts(dict_params)
s = 50
xl_is, xu_is = s*min(initial_params), s*max(initial_params)  # 50 by 50 works together with 9 Neurons. 
#list(dict_params.keys())

obj1 = physics_loss(model, training_points)

print("type of obj1:", type(obj1))

def objective1(X_params):
    X_params = torch.tensor(X_params)
    model_params = model.state_dict()
    start_k, end_k = 0, 0
    for name, param in model_params.items():
        end_k = start_k + dict_params[name].numel()
        
        flat_tensor = X_params[start_k:end_k]
        # print("flat_tensor: ", flat_tensor.shape)
        #print("param: ", param.shape)
        new_param = flat_tensor.view(param.shape)
        param.copy_(new_param)
        
        start_k = end_k
    
    model.train()
    train_loss = 0.0

    obj1 = physics_loss(model, training_points)
    return obj1.cpu().detach()

def objective2(X_params):
    X_params = torch.tensor(X_params)
    model_params = model.state_dict()
    start_k, end_k = 0, 0
    for name, param in model_params.items():
        end_k = start_k + dict_params[name].numel()
        
        flat_tensor = X_params[start_k:end_k]
        # print("flat_tensor: ", flat_tensor.shape)
        #print("param: ", param.shape)
        new_param = flat_tensor.view(param.shape)
        param.copy_(new_param)
        
        start_k = end_k
    
    model.train()
    train_loss = 0.0

    obj1 = data_loss(model, data_noisy, training_points)
    return obj1.cpu().detach()





objs = [objective2, objective1]

n_var = N

problem = FunctionalProblem(n_var,
                            objs,
                            xl= xl_is,
                            xu= xu_is,
                            )


ref_dirs = get_reference_directions("uniform", 2, n_partitions=45)

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)
# Start timer
print(datetime.now())
t0 = time()

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

T_norm_1 = time()-t0
# Print computation time
print('\nComputation time: {} secs'.format(T_norm_1))
print(datetime.now())

# Decision Space
X_pareto_front = res.X
#Objective Space
F_pareto_front = res.F

print("len F_pareto_front):", len(F_pareto_front))

# Specify the folder name
folder_name = f'NSGA2_n_{n_neurons}'

# Create the new folder if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# Specify the path to the new file
file_path_1 = os.path.join(folder_name, 'Logistic_X_pareto_front.txt')
file_path_2 = os.path.join(folder_name, 'Logistic_F_pareto_front.txt')

# Save the NumPy array as a .txt file in the new folder
np.savetxt(file_path_1, X_pareto_front)
np.savetxt(file_path_2, F_pareto_front)

#Plot objective (train)
#Scatter().add(res.F).show()
F_pareto_front = np.loadtxt(f'NSGA2_n_{n_neurons}/Logistic_F_pareto_front.txt')
#plt.plot(F_pareto_front[:, 0], F_pareto_front[:, 1],"-o",  label = "train (EA)" , linewidth = 4, markersize= 8)
plt.scatter(F_pareto_front[:, 0], F_pareto_front[:, 1], s=30, label="train (EA)")  # `s` sets the size of the markers.
plt.xlabel(r"$L_\mathrm{DATA}$")
plt.ylabel(r"$L_\mathrm{PHYSICS}$")
plt.show()
plt.savefig(f"NSGA_n_{n_neurons}/NSGA_n{n_neurons}.png")