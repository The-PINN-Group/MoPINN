
import os
from random import randint

import numpy as np

#from DataLoader import MultiMNISTData
#from MultiTaskLeNet import MultiTaskLeNet, MultiTaskLeNet_Linear
#from MultiTaskOptimizer import build_multi_task_optimizer


from os import path
import torch
import torch.optim
from sklearn.model_selection import train_test_split
from torch import device, nn, Tensor
from typing import Iterable, Union
import matplotlib.pyplot as plt
from datetime import datetime
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
_params_t = Union[Iterable[Tensor], Iterable[dict]]

import torch.optim
from torch import Tensor


def build_multi_task_optimizer(optimizer_type: torch.optim.Optimizer):
    """
    Builds a MultiTaskOptimizer Class derived from optimizer_type, e.g Adam or SGD.

    :param optimizer_type: class of the optimizer to use, e.g torch.optim.Adam
    :return: MultiTaskOptimizer: class
    """
    class MultiTaskOptimizer(optimizer_type):
        underlying_optimizer = optimizer_type

        @staticmethod
        def frank_wolfe_solver(gradients: list,
                               termination_threshold: float = 1e-4,
                               max_iterations: int = 10,
                               device: str = "cpu") -> Tensor:
            """
            Applies Frank-Wolfe-Solver to a list of (shared) gradients on t-many tasks
            :param gradients: list of (shared) gradients
            :param termination_threshold: termination condition
            :param max_iterations: #iterations before algorithm termination
            :return: Tensor of shape [t]
            """

            # Amount of tasks
            T = len(gradients)
            # Amount of layers
            L = len(gradients[0])

            # Initialize alpha
            alpha = torch.tensor([1 / T for _ in range(T)], device=device)

            M = torch.zeros(size=(T, T), dtype=torch.float32, device=device)


            for i in range(T):
                flat_gradient_i = torch.concat([torch.flatten(gradients[i][layer]) for layer in range(L)])
                for j in range(T):
                    flat_gradient_j = torch.concat([torch.flatten(gradients[j][layer]) for layer in range(L)])
                    if M[j][i] != 0:
                        M[i][j] = M[j][i]
                    else:
                        M[i][j] = torch.dot(flat_gradient_i, flat_gradient_j)

            # Initialize gamma
            gamma = float('inf')
            iteration = 0

            #return alpha([0.5, 0.5])

            while gamma > termination_threshold and iteration <= max_iterations:
                alpha_m_sum = torch.matmul(alpha, M)
                t_hat = torch.argmin(alpha_m_sum)

                g_1 = torch.zeros_like(alpha, device=device)
                g_2 = alpha

                g_1[t_hat] = 1

                g1_Mg1 = torch.matmul((g_1), torch.matmul(M, g_1))
                g2_Mg2 = torch.matmul((g_2), torch.matmul(M, g_2))
                g1_Mg2 = torch.matmul((g_1), torch.matmul(M, g_2))

                if g1_Mg1 <= g1_Mg2:
                    gamma = 1
                elif g1_Mg2 >= g2_Mg2:
                    gamma = 0
                else:
                    dir_a = g2_Mg2 - g1_Mg2
                    dir_b = g1_Mg1 - 2*g1_Mg2 + g2_Mg2
                    gamma = dir_a / dir_b

                alpha = (1 - gamma) * alpha + gamma * g_1
                iteration += 1

                if T <= 2:
                    break
            return alpha

    return MultiTaskOptimizer


# Example usage
if __name__ == "__main__":
    # invoking class factory function
    MTLOptimizerClass = build_multi_task_optimizer(torch.optim.Adam)
    example_parameters = [torch.tensor([0]), torch.tensor([1])]
    mtl_optim = MTLOptimizerClass(example_parameters, lr=0.0001)
    print(f"Initialized Multi-Task Optimizer with underlying optimizer: {mtl_optim.underlying_optimizer}")

""" def train_multi(X, y, model: nn.Module, optimizer: MultiTaskOptimizer, loss_fn):
    # X is a torch Variable
    
    #TODO: Adapt this function for PINNs. It should take same arguments as the training function in our PINNs template.
    # Ideally in our template we do training in the main script. So maybe it better to do it there.  
    
    permutation = torch.randperm(X.size()[0])
    losses = [[] for _ in range(model.num_of_tasks)]
    task_accuracies = [[] for _ in range(model.num_of_tasks)]
    total_accuracies = []

    for i in range(0, X.size()[0], batch_size):
        #TODO: Look into if we need batches or not. I think that we do not need it. 

        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X[indices].to(device), y[indices].to(device)
        # Only accept full-size batches
        if batch_x.shape[0] != batch_size:
            continue
        batch_x = batch_x.reshape([batch_size, 1, 32, 32, 3])

        shared_gradients = [[] for _ in range(model.num_of_tasks)]
        for t in range(model.num_of_tasks):

            #TODO: This loop runs over the number of loss functions that I will have

            model.zero_grad()

            #TODO: The following 6 lines of code are not needed as I already have the losses in PINNs
            # Full forward pass
            output = model.forward(batch_x)
            # Compute t-specific loss
            t_output = output[:, t * model.task_out_n:(t + 1) * model.task_out_n]
            t_label = batch_y[:, t]
            t_loss = loss_fn(t_output, t_label)
            #losses[t].append(t_loss.item())

            # Compute classification accuracy per task
            # task_accuracies[t].append(compute_accuracy(t_output, batch_y[:, t]))

            # Backward pass (i.e., gradients are calculated in the backward pass)
            t_loss.backward()

            #storing the gradients from each layer in nexted lists. 
            for param in model.shared.parameters():
                if param.grad is not None:
                    _grad = param.grad.data.detach().clone()
                    shared_gradients[t].append(_grad)

        # Compute total accuracy (both digits correct)
        # total_accuracies.append(compute_total_accuracy(output, batch_y))

        alphas = optimizer.frank_wolfe_solver(shared_gradients, device=device)

        # Collect task specific gradients regarding task specific loss
        z = model.shared_forward(batch_x)

        # aggregate loss
        #loss = torch.zeros(1, device=device)
        #for t in range(model.num_of_tasks):
        #    loss += loss_fn(model.subnetworks[f"task_{t}"].forward(z), batch_y[:, t]) * alphas[t]


        loss = # dot product of alphas and loss vector/list 
        loss.backward()

        optimizer.step()
    return losses, task_accuracies, total_accuracies
 """