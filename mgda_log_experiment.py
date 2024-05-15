# %%

from pinntorch import *
from functools import partial
from typing import List
import wandb
import argparse
from types import SimpleNamespace


hyperparameters = SimpleNamespace(
    number_hidden_layers=5,
    number_hidden_dimension=5,
    learning_rate=0.003,
    epochs=100,
    torch_seed=70,
    numpy_seed=70,
    pde="logistic equation",
    pde_parameter_k=5.0,
    data_noice_variance=0.1,
    number_data_points=20,
    number_training_points=20,
    number_test_points=39,  # 2 x training points - 1 for points between training points
    number_plot_points=100,
    moo_method="mgda",
    moo_ls_alpha_weight=0.5,
    moo_normalization="norm",  # "norm", "loss", "loss+", "none"
    base_optimizer="Adam",
    wiggle_weights_variance=0.1,
    wiggle_weights_seed=0,
    init_weights_seed=0,
    torch_device="cpu",
)

# dict hyperparameters: default values
# args argpasre: overwrite default values
# wandb.config: only for wandb


def setup_device(config=hyperparameters):
    device = torch.device(config.torch_device)  # Use 'cuda' for GPU or 'cpu' for CPU
    print(f"Using device: {device}")

    # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.manual_seed(config.torch_seed)
    np.random.seed(config.numpy_seed)

    return device


def wiggle_weights(model, variance=hyperparameters.wiggle_weights_variance):
    for param in model.parameters():
        noise = variance * torch.randn_like(param.data)
        param.data.add_(noise)


def generate_noisy_data(
    data_points, fn, variance=hyperparameters.data_noice_variance
) -> List[torch.Tensor]:
    f_data = fn(data_points) + variance * torch.randn_like(data_points)

    return f_data


def compute_data_loss(
    nn: PINN, x_data: List[torch.Tensor] = None, y_data: List[torch.Tensor] = None
) -> torch.float:
    data_loss = 0

    y_real = f(nn, x_data)
    mse_loss = torch.nn.MSELoss()
    data_loss = mse_loss(y_real, y_data)

    return data_loss


def compute_pde_loss(
    nn: PINN,
    x: List[torch.Tensor] = None,
    pde_parameter_k=hyperparameters.pde_parameter_k,
) -> torch.float:
    pde_loss = 0

    pde_loss_pre = df(nn, x) - pde_parameter_k * f(nn, x) * (1 - f(nn, x))
    pde_loss = pde_loss_pre.pow(2).mean()

    return pde_loss


def compute_boundary_loss(nn: PINN, x: List[torch.Tensor] = None) -> torch.float:
    boundary_loss = 0

    boundary_loss = (f(nn, at(+1.0)) - logistic_fn(+1)).pow(2).mean()

    return boundary_loss


# analytic solution
def logistic_fn(x, pde_parameter_k=hyperparameters.pde_parameter_k):
    return 1 / (1 + np.exp(-pde_parameter_k * x))


def load_data(config=hyperparameters):
    # collocation points for phyiscs
    training_points = generate_grid(
        (config.number_training_points), (-1.0, 1.0)
    )  # equidistant grid
    test_points = generate_grid((config.number_test_points), (-1.0, 1.0))
    plot_points = generate_grid((config.number_plot_points), domain=(-1.0, 1.0))

    x_data_points = generate_grid(
        config.number_data_points, (-1.0, 1.0), requires_grad=False
    )  # grid sample
    y_data_points = generate_noisy_data(
        x_data_points, logistic_fn, config.data_noice_variance
    )  # solution with noise

    return training_points, test_points, plot_points, x_data_points, y_data_points


def visualize(
    trained_model,
    callbacks,
    plot_points,
    training_points,
    x_data_points,
    y_data_points,
    solution_fn=logistic_fn,
    config=hyperparameters,
):
    # plotting
    #train_loss_evolution = callbacks[0].loss_history
    #true_error_evolution = callbacks[1].mae_history

    #train_weight_evolution_moo = callbacks[0].extra_logs
    #alpha = train_weight_evolution_moo[-1]["mgda_weight_0"]

    # plot_evolution(
    #     [
    #         train_loss_evolution,
    #         true_error_evolution,
    #     ],
    #     [
    #         "train loss",
    #         "true error",
    #     ],
    #     linestyle=["-", "-"],
    # ) # is included in the wandb callback

    plot_solution_1D(
        trained_model,
        plot_points,
        training_points,  # Wait for new plotly version
        solution_fn=logistic_fn,
        labels=[str(config.moo_method.name)],
    )
    plt.plot(x_data_points, y_data_points, "o", label="data points")
    # plt.title(f"α={alpha:.2f}") # only for ls
    plt.legend()

    figs = list(map(plt.figure, plt.get_fignums()))
    for fig in figs:
        wandb.log({"solution": wandb.Image(fig)})
        # wandb.log({"plot": fig})

    # plt.show()


def train(config=hyperparameters, device=torch.device(hyperparameters.torch_device)):
    """Training Loop

    Args:
        config (_type_, optional): _description_. Defaults to hyperparameters.

    Returns:
        _type_: Returns model, callbacks and inputs
    """
    with wandb.init(project=config.wandb_project, config=config, save_code=True):
        # config = wandb.config  # use wandb config to get the hyperparameters

        # load data
        (
            training_points,
            test_points,
            plot_points,
            x_data_points,
            y_data_points,
        ) = load_data(config)

        # model
        model = PINN(
            1, config.number_hidden_layers, config.number_hidden_dimension, 1
        )  # input_dim, num_hidden, hidden_dim, output_dim

        # initialize weights
        model.initialize_weights(seed=config.init_weights_seed)

        # # wiggle weights
        # torch.manual_seed(config.wiggle_weights_seed)
        # wiggle_weights(model, variance=config.wiggle_weights_variance)
        # # print weights of model
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.data}")

        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {parameter_count}")

        callbacks = [
            TrainLossMonitor(),
            TrueErrorMonitor(test_points, logistic_fn),
            # SolutionMonitor(plot_points, training_points, store_every=20),
            CheckpointMonitor(log_period=100),
            WandBMonitor(test_points, logistic_fn),
        ]

        # losses
        def combined_loss(*args, **kwargs):
            pde_loss = compute_pde_loss(*args, **kwargs)
            boundary_loss = compute_boundary_loss(*args, **kwargs)
            return pde_loss + boundary_loss

        # loss_fn_pde = partial(compute_pde_loss, x=training_points) # only pde
        # loss_fn_boundary = partial(
        #     compute_boundary_loss, x=training_points
        # )  # is included in the data
        loss_fn_pde = partial(
            combined_loss, x=training_points
        )  # combined pde and boundary
        loss_fn_data = partial(
            compute_data_loss, x_data=x_data_points, y_data=y_data_points
        )
        loss_fn = [loss_fn_pde, loss_fn_data]  # always pde first, data second

        print('method_config:', config.moo_method)
        method = WeightMethods(
            method=config.moo_method,
            n_tasks=2,
            # normalization=config.moo_normalization, # mgda
            device=device,
        )  # wandb config converts to string, therefore use the dict

        if config.moo_method == Moo_method.ls:
            method.method.task_weights = torch.tensor(
                [
                    config.moo_ls_alpha_weight,
                    1 - config.moo_ls_alpha_weight,
                ]  # always pde first, data second
            )

        # training
        print('method\n', method)
        trained_model = train_model(
            model=model,
            loss_fn=loss_fn,
            mo_method=method,
            max_epochs=config.epochs,
            optimizer_fn=partial(config.base_optimizer, lr=config.learning_rate),  # wandb config converts to string, therefore use the dict
            epoch_callbacks=callbacks,
        )

        # plotting, optional
        visualize(
            trained_model,
            callbacks,
            plot_points,
            training_points,
            x_data_points,
            y_data_points,
            config=config,
        )

        return (
            trained_model,
            callbacks,
            plot_points,
            training_points,
            x_data_points,
            y_data_points,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_hidden_layers",
        type=int,
        default=hyperparameters.number_hidden_layers,
        help="number of hidden layers",
    )
    parser.add_argument(
        "--number_hidden_dimension",
        type=int,
        default=hyperparameters.number_hidden_dimension,
        help="number of hidden dimension",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=hyperparameters.learning_rate,
        help="learning rate",
    )
    parser.add_argument(
        "--epochs", type=int, default=hyperparameters.epochs, help="number of epochs"
    )
    parser.add_argument(
        "--moo_method",
        type=str,
        default=hyperparameters.moo_method,
        help="multi-objective optimization method",
    )
    parser.add_argument(
        "--moo_normalization",
        type=str,
        default=hyperparameters.moo_normalization,
        help="multi-objective optimization normalization",
    )
    parser.add_argument(
        "--base_optimizer",
        type=str,
        default=hyperparameters.base_optimizer,
        help="base optimizer",
    )
    parser.add_argument(
        "--number_training_points",
        type=int,
        default=hyperparameters.number_training_points,
        help="number of training points",
    )
    parser.add_argument(
        "--number_test_points",
        type=int,
        default=hyperparameters.number_test_points,
        help="number of test points",
    )
    parser.add_argument(
        "--number_plot_points",
        type=int,
        default=hyperparameters.number_plot_points,
        help="number of plot points",
    )
    parser.add_argument(
        "--number_data_points",
        type=int,
        default=hyperparameters.number_data_points,
        help="number of data points",
    )
    parser.add_argument(
        "--data_noice_variance",
        type=float,
        default=hyperparameters.data_noice_variance,
        help="data noise variance",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=hyperparameters.torch_seed,
        help="torch seed",
    )
    parser.add_argument(
        "--pde_parameter_k",
        type=float,
        default=hyperparameters.pde_parameter_k,
        help="pde parameter k",
    )
    parser.add_argument(
        "--torch_device",
        type=str,
        default=hyperparameters.torch_device,
        help="torch device",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="logistic_equation_hyperparameter_search",
        help="wandb project name",
    )
    parser.add_argument(
        "--wiggle_weights_variance",
        type=float,
        default=hyperparameters.wiggle_weights_variance,
        help="wiggle weights variance",
    )
    parser.add_argument(
        "--wiggle_weights_seed",
        type=int,
        default=hyperparameters.wiggle_weights_seed,
        help="wiggle weights seed",
    )
    parser.add_argument(
        "--init_weights_seed",
        type=int,
        default=hyperparameters.init_weights_seed,
        help="init weights seed",
    )
    parser.add_argument(
        "--numpy_seed",
        type=int,
        default=hyperparameters.numpy_seed,
        help="numpy seed",
    )
    parser.add_argument(
        "--pde",
        type=str,
        default=hyperparameters.pde,
        help="pde",
    )
    parser.add_argument(
        "--moo_ls_alpha_weight",
        type=float,
        default=hyperparameters.moo_ls_alpha_weight,
        help="moo ls alpha weight",
    )

    args = parser.parse_args()

    if hasattr(torch.optim, args.base_optimizer):
        OptimClass = getattr(torch.optim, args.base_optimizer)
        args.base_optimizer = OptimClass
        print(f"Using optimizer: {args.base_optimizer}")

    if hasattr(Moo_method, args.moo_method):
        args.moo_method = getattr(Moo_method, args.moo_method)
        print(f"Using moo method: {args.moo_method}")

    return args


if __name__ == "__main__":
    args = parse_args()
    device = setup_device(config=args)
    train(config=args, device=device)


# %%
