from pinntorch._dependencies import *
from pinntorch._model import f
from pinntorch._training import SolutionPlotMonitor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib import animation


def plot_evolution(
    value_evolution,
    evolution_label,
    title="Training loss evolution",
    xlabel="# epochs",
    ylabel="loss",
    linestyle=['solid', 'solid'],
):
    """
    Plots the evolution of a value over time.

    Parameters
    ----------
    value_evolution : list[torch.Tensor] or torch.Tensor or numpy.ndarray\\
        The evolution of the value as a list of numpy arrays or a PyTorch tensor of shape (num_epochs,).
    evolution_label : list[str]] or str\\
        The label or labels to use for the value evolution on the plot. If `value_evolution` contains more than one loss, `evolution_label` should be a list of labels of the same length.
    title : str, optional\\
        The title of the plot, by default "Training loss evolution".
    xlabel : str, optional\\
        The label of the x-axis, by default "# epochs".
    ylabel : str, optional\\
        The label of the y-axis, by default "loss".

    Raises
    ------
    ValueError\\
        If the length of `value_evolution` and `evolution_label` do not match when `value_evolution` contains more than one loss.
    """
    fig, ax = plt.subplots()
    if isinstance(evolution_label, list):
        if len(value_evolution) != len(evolution_label):
            raise ValueError("The number of value evolutions and labels must match.")
        for i, evo in enumerate(value_evolution):
            ax.semilogy(evo, label=evolution_label[i], linestyle=linestyle[i])
    else:
        ax.semilogy(value_evolution, label=evolution_label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))
    ax.legend()


def plot_solution_1D(
    model, plot_points, train_points=None, labels=None, colors=None, solution_fn=None
):
    """
    Plot the one-dimensional outputs separately of a neural network model for a given set of points.

    Parameters
    ----------
    model : torch.nn.Module\\
        The trained neural network model.
    plot_points : torch.Tensor\\
        The points at which to evaluate the model's solution.
    train_points : torch.Tensor, optional\\
        The points used during the training of the model, by default None.
    labels : list of str, optional\\
        Labels for the different output dimensions of the model, by default None.
    colors : list of str, optional\\
        Colors to use for the plotted lines, by default None.
    solution_fn : callable, optional\\
        The analytic solution function to compare with, by default None.
    """
    fig, ax = plt.subplots()
    if train_points is not None:
        f_final_training = f(model, train_points, of="all")
        if type(f_final_training) is not list:
            f_final_training = [f_final_training]
        train_points_np = train_points.detach().cpu().numpy()
    f_final = f(model, plot_points, of="all")
    if type(f_final) is not list:
        f_final = [f_final]
    plot_points_np = plot_points.detach().cpu().numpy()

    if labels == None:
        labels = ["output " + str(i) for i in range(len(f_final))]
    if colors == None:
        colors = "bgrcykw"
    for i in range(len(f_final)):
        ax.plot(
            plot_points_np,
            f_final[i].detach().cpu().numpy(),
            label=labels[i] + " model solution",
            color=colors[i % len(colors)],
        )

    ### analytic solution ###
    if solution_fn != None:
        ax.plot(
            plot_points_np,
            solution_fn(plot_points_np),
            linestyle="--",
            label=f"analytic solution",
            color="magenta",
            alpha=0.75,
        )
        if train_points != None:
            ax.scatter(
                train_points_np,
                solution_fn(train_points_np),
                label=labels[i] + " training points",
                color="magenta",
                marker="x",
            )
    ax.set(title="Equation solved with the NN", xlabel="t", ylabel="f(t)")
    ax.legend()


def add_plot_solution_1D(
    model, plot_points, train_points=None, labels=None, colors=None, solution_fn=None
):
    """
    Plot the one-dimensional outputs separately of a neural network model for a given set of points.

    Parameters
    ----------
    model : torch.nn.Module\\
        The trained neural network model.
    plot_points : torch.Tensor\\
        The points at which to evaluate the model's solution.
    train_points : torch.Tensor, optional\\
        The points used during the training of the model, by default None.
    labels : list of str, optional\\
        Labels for the different output dimensions of the model, by default None.
    colors : list of str, optional\\
        Colors to use for the plotted lines, by default None.
    solution_fn : callable, optional\\
        The analytic solution function to compare with, by default None.
    """
    ax = plt.gcf().add_subplot()
    if train_points is not None:
        f_final_training = f(model, train_points, of="all")
        if type(f_final_training) is not list:
            f_final_training = [f_final_training]
        train_points_np = train_points.detach().cpu().numpy()
    f_final = f(model, plot_points, of="all")
    if type(f_final) is not list:
        f_final = [f_final]
    plot_points_np = plot_points.detach().cpu().numpy()

    if labels == None:
        labels = ["output " + str(i) for i in range(len(f_final))]
    if colors == None:
        colors = "bgrcykw"
    for i in range(len(f_final)):
        if train_points != None:
            ax.scatter(
                train_points_np,
                f_final_training[i].detach().cpu().numpy(),
                label=labels[i] + " train points",
                color=colors[i % len(colors)],
            )
        ax.plot(
            plot_points_np,
            f_final[i].detach().cpu().numpy(),
            label=labels[i] + " model solution",
            color=colors[i % len(colors)],
        )

    ### analytic solution ###
    if solution_fn != None:
        ax.plot(
            plot_points_np,
            solution_fn(plot_points_np),
            linestyle="--",
            label=f"analytic solution",
            color="magenta",
            alpha=0.75,
        )
    ax.set(title="Equation solved with the NN", xlabel="t", ylabel="f(t)")
    ax.legend()


#! UNTESTED
def plot_solution_xt_animation(model: nn.Module, x: torch.Tensor, t: torch.Tensor):
    fig, ax = plt.subplots()
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)

    def animate(i):
        t_partial = torch.ones_like(x_raw) * t_raw[i]
        f_final = f(model, x_raw, t_partial)
        ax.clear()
        ax.plot(
            x_raw.detach().cpu().numpy(),
            f_final.detach().cpu().numpy(),
            label=f"Time {float(t[i])}",
        )
        ax.set_ylim([0, 1])
        ax.legend()

    n_frames = t_raw.shape[0]
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=30, repeat=False)

    return anim


def animate_solution_1D_history(
    solution_monitor: SolutionPlotMonitor, true_solution_fn=None, ylim=(0, 1), interval=10
):
    solution_plot_history = solution_monitor.solution_plot
    solution_train_history = solution_monitor.solution_train
    train_points = solution_monitor.train_points
    plot_points = solution_monitor.plot_points
    plot_interval = solution_monitor.plot_interval

    num_outputs = len(solution_plot_history[0])
    print(len(solution_plot_history))

    fig, ax = plt.subplots()
    colors = "bgrcykw"

    def update(epoch):
        ax.clear()
        for i in range(num_outputs):
            if train_points is not None:
                ax.scatter(
                    train_points,
                    solution_train_history[epoch][i],
                    label=f"Epoch {(epoch)*plot_interval} - train points [{i}]",
                    color=colors[i],
                )
            ax.plot(
                plot_points,
                solution_plot_history[epoch][i],
                label=f"Epoch {(epoch)*plot_interval} - model solution [{i}]",
                color=colors[i],
            )
            ax.set_ylim(ylim)

        if true_solution_fn is not None:
            ax.plot(
                plot_points,
                true_solution_fn(plot_points),
                linestyle="--",
                label=f"analytic solution",
                color="magenta",
                alpha=0.75,
            )
        ax.legend()

    anim = FuncAnimation(
        fig, update, frames=len(solution_plot_history), interval=interval, repeat=False
    )
    return anim


def plot_solution_xt_3d(
    nn_trained: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    xt_shape: tuple,
    color_map=cm.hsv,
):
    z = f(nn_trained, x, t)
    x = x.cpu().detach().numpy().reshape(xt_shape)
    t = t.cpu().detach().numpy().reshape(xt_shape)
    z = z.cpu().detach().numpy().reshape(xt_shape)

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.set(xlabel="x (location)", ylabel="t (time)", zlabel="function value")

    ls = LightSource(270, 45)

    surf = ax.plot_surface(
        x, t, z, cmap=color_map, linewidth=0, antialiased=False, shade=False
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    
def plot_solution_heatmap(nn_trained, x, t, xt_shape, color_map=cm.hsv):
    z = f(nn_trained, x, t)
    x = x.cpu().detach().numpy().reshape(xt_shape)
    t = t.cpu().detach().numpy().reshape(xt_shape)
    z = z.cpu().detach().numpy().reshape(xt_shape)

    fig, ax = plt.subplots()
    cax = ax.imshow(z, cmap=color_map, origin='lower', aspect='auto', extent=(x.min(), x.max(), t.min(), t.max()))
    plt.colorbar(cax, label='Function Value')
    ax.set(xlabel='x (location)', ylabel='t (time)', title='Heat Map of Function')

    plt.show()