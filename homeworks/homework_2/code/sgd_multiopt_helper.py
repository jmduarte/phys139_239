########################################
# CS/CNS/EE 155 2018
# Problem Set 1
#
# Author:       Andrew Kang
# Description:  Set 1 visualization helper
#               (Loss function with multiple optima)
########################################

import numpy as np

# Technically this isn't SGD, but whatever - this is for
# demonstration purposes only...


def valley(w, x_center, y_center, depth, girth):
    a = depth
    b = 1 / girth
    return 1 / (1 + a * np.exp(-b * ((w[0] - x_center) ** 2 + (w[1] - y_center) ** 2)))


def GD_loss(w):
    return valley(w, 0.4, 0.5, 3, 0.25) * valley(w, -0.3, -0.6, 1, 0.25)


def GD_gradient(w):
    curr = GD_loss(w)
    diff = 0

    for theta in np.linspace(0, 2 * np.pi, 100, endpoint=False):
        dw_curr = np.array([np.cos(theta), np.sin(theta)])
        diff_new = GD_loss(w + dw_curr) - curr

        if diff_new > diff:
            diff = diff_new
            dw = dw_curr

    return dw


def GD(w_start, eta, N_iters):
    losses = np.zeros(N_iters)
    W = np.zeros((N_iters, len(w_start)))
    w = w_start

    # Perform GD for each epoch.
    for i in range(N_iters):
        W[i] = w
        losses[i] = GD_loss(w)
        w -= eta * GD_gradient(w)

    return W, losses


def SGD(X, Y, w_start, eta, N_iters):
    return GD(w_start, eta, N_iters)


def loss(X, Y, w):
    return GD_loss(w)
