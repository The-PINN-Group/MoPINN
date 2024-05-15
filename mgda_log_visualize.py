import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker


def identify_pareto(points):
    """
    Identify Pareto front indices in a set of points.
    :param points: An array of points.
    :return: Indices of points in the Pareto front.
    """
    population_size = points.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(points[j] <= points[i]) and any(points[j] < points[i]):
                pareto_front[i] = 0
                break
    return pareto_front


def exp_formatter(x, pos):
    """Custom formatter to display labels as 10 to the minus exponent"""
    return r"$10^{{-{:.0f}}}$".format(-np.log10(x))


def plot_pareto_front(
    L_D,
    L_P,
    pareto_indices,
    xtick_rotation=0,
    file_name="pareto_plot",
    bbox_bounds=(0.1, -0.1, 5.2, 3.8),
    left_margin=0.15,
    bottom_margin=0.15,
    x_lim=None,
    y_lim=None,
):
    plt.figure(figsize=(5, 4))

    # Plot all points
    plt.scatter(L_D, L_P, color="gray", label="Dominated")  # Non-Pareto points in gray
    # Plot Pareto points in a different color
    plt.scatter(
        np.array(L_D)[pareto_indices],
        np.array(L_P)[pareto_indices],
        color="blue",
        label="Non-Dominated",
    )  # Pareto points in blue

    plt.ylabel(r"$\mathcal{L}_\mathrm{PHYSICS}$", loc="center", fontsize=13)
    plt.xlabel(r"$\mathcal{L}_\mathrm{DATA}$", loc="center", fontsize=13)
    plt.grid()

    # Set axis to use scientific notation
    plt.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
    plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))

    plt.legend()

    # Optionally set x and y limits to zoom in
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.subplots_adjust(left=left_margin, bottom=bottom_margin)

    bbox_instance = Bbox.from_bounds(*bbox_bounds)
    plt.xticks(rotation=xtick_rotation)

    plt.savefig("plots/" + file_name + ".png", dpi=600)
    plt.show()


# Load the data
with pd.HDFStore(
    "plots/history_logistic_equation_mgda_nonorm_pareto (pn9t23h4).h5", "r"
) as store:
    runs_last_epoch = []
    for key in store.keys():
        df = store[key]
        last_epoch_data = df.iloc[-1]  # Assuming the data is sorted by epoch
        runs_last_epoch.append(last_epoch_data)

# Extract L_D (loss_data), L_P (loss_pde) from the last epoch of each run
L_D = [run["loss_data"] for run in runs_last_epoch]
L_P = [run["loss_pde"] for run in runs_last_epoch]

points = np.array(list(zip(L_D, L_P)))
pareto_indices = identify_pareto(points)

plot_pareto_front(
    L_D,
    L_P,
    pareto_indices,
    file_name="pareto_front_visualization",
    # bbox_bounds=(0.125, 0.1, 0.9, 0.9),
    x_lim=(0.007, 0.011),
    y_lim=(-0.01, 0.07),
)
