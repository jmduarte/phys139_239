########################################
# CS/CNS/EE 155 2018
# Problem Set 1
#
# Author:       Andrew Kang
# Description:  Set 1 SGD helper
########################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# All functions operate under the assumption that:
#   - We are dealing with SGD in the 2-dimensional case
#   - The dataset is drawn from points in the domain [-1, 1] X [-1, 1]


####################
# DATASET FUNCTIONS
####################


def generate_dataset(N, f, noise):
    # Generates an approximately linearly separable dataset:
    #   - X is drawn from [-1, 1] X [-1, 1].
    #   - Y is the dot product of X with some f plus some noise.
    X = np.random.uniform(-1, 1, (N, 2))
    Y = np.dot(f, X.T) + noise * np.random.rand(N)
    return X, Y


def generate_dataset1():
    # A specific instance of generate_dataset().
    np.random.seed(155)
    return generate_dataset(500, np.array([0.5, -0.1]).T, 0.1)


def generate_dataset2():
    # A specific instance of generate_dataset().
    np.random.seed(155)
    return generate_dataset(500, np.array([-0.2, -0.3]).T, 0.1)


####################
# PLOTTING FUNCTIONS
####################


def plot_dataset(X, Y, show=True):
    # Create a new figure and get its axes.
    plt.close("all")
    fig = plt.figure()
    ax = fig.gca()

    # Plot X and Y with the 'bwr' colormap centered at zero.
    plt.set_cmap("bwr")
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=Y,
        edgecolor="black",
        linewidth=0.5,
        vmin=min(np.min(Y), -np.max(Y)),
        vmax=max(np.max(Y), -np.min(Y)),
    )
    plt.colorbar()

    # Label the axes.
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    if show:
        plt.show()

    return fig, ax


def get_loss_grid(x_params, y_params, X, Y, loss):
    # Get 2D meshgrid.
    dx = np.linspace(*x_params)
    dy = np.linspace(*y_params)
    w_grid = np.meshgrid(dx, dy)

    # Evaluate loss on each point of the meshgrid.
    loss_grid = np.zeros_like(w_grid[0])
    for i in range(len(loss_grid)):
        for j in range(len(loss_grid[0])):
            w = np.array([w_grid[0][i, j], w_grid[1][i, j]])
            loss_grid[i, j] = loss(X, Y, w)

    return w_grid, loss_grid


def plot_loss_function(X_grid, Y_grid, loss_grid):
    # Create a new figure and get its axes.
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot the loss function in 3D.
    ax.plot_surface(X_grid, Y_grid, loss_grid)

    return fig, ax


####################
# SGD ANIMATION FUNCTIONS
####################


def multiSGD(SGD, X, Y, params, N_epochs):
    # Arrays to store the results of SGD.
    losses_lst = np.zeros((len(params), N_epochs))
    W_lst = np.zeros((len(params), N_epochs, 2))

    for i, param in enumerate(params):
        print("Performing SGD with parameters", param, "...")

        # Run SGD on the current set of parameters and store the results.
        W, losses = SGD(X, Y, param["w_start"], param["eta"], N_epochs)
        W_lst[i] = W
        losses_lst[i] = losses

    # some abysmal variable naming here... lol whoops
    return W_lst, losses_lst


# NOTE: 'FR' is not exactly the frame rate. Again, abysmal variable naming.
def animate_sgd_suite(SGD, loss, X, Y, params, N_epochs, FR, ms=1):
    delay = 5

    # Run SGD on each set of parameters.
    W_lst, losses_lst = multiSGD(SGD, X, Y, params, N_epochs)

    # Get the loss grid and plot it.
    w_grid, loss_grid = get_loss_grid((-1, 1, 100), (-1, 1, 100), X, Y, loss)
    fig, ax = plot_loss_function(w_grid[0], w_grid[1], loss_grid)

    # Label the axes:
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")

    # Plot w_start values.
    (_,) = ax.plot(W_lst[:, 0, 0], W_lst[:, 0, 1], losses_lst[:, 0], "+", mew=2, ms=10, c="orange")

    # Initialize graph to animate on.
    (graph,) = ax.plot([], [], [], "o", ms=ms, c="orange")
    graph.set_markeredgecolor("orange")
    graph.set_markeredgewidth(1)

    # Define frame animation function.
    def animate(i):
        if i > delay:
            i -= delay

            graph.set_data(W_lst[:, : FR * (i + 1), 0].flatten(), W_lst[:, : FR * (i + 1), 1].flatten())
            graph.set_3d_properties(losses_lst[:, : FR * (i + 1)].flatten())

            return graph

    # Animate!
    print("\nAnimating...")
    anim = FuncAnimation(fig, animate, frames=int(N_epochs / FR) + delay, interval=50)

    return anim


def animate_convergence(X, Y, W, FR):
    delay = 5

    # Plot w_start values.
    fig, ax = plot_dataset(X, Y, show=False)

    # Initialize graph to animate on.
    (graph,) = ax.plot([], [])

    # Define frame animation function.
    def animate(i):
        if i > delay:
            i -= delay

            w = W[i]
            x_ax = np.linspace(-1, 1, 100)
            graph.set_data(x_ax, -(w[0] / w[1]) * x_ax)
            return graph

    # Animate!
    print("\nAnimating...")
    anim = FuncAnimation(fig, animate, frames=int(len(W) / FR) + delay, interval=50)

    return anim


# Hey there! Hope you're having fun with the set :^)
