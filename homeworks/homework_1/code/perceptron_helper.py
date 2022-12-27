# UCSD PHYS 139/239 2021
# Homework 1
# Original Author:       Joey Hong
# Description:  Set 1 Perceptron helper

import numpy as np


def predict(x, w, b):
    """
    The method takes the weight vector and bias of a perceptron model, and
    predicts the label for a single point x.

    Inputs:
        x: A (D, ) shaped numpy array containing a single point.
        w: A (D, ) shaped numpy array containing the weight vector.
        b: A float containing the bias term.

    Output:
       The label (1 or -1) for the point x.
    """
    prod = np.dot(w, x) + b
    return 1 if prod >= 0 else -1


def plot_data(X, Y, ax):
    # This method plots a labeled (with -1 or 1) 2D dataset.
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c="green", marker="+")
    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], c="red")


def boundary(x_1, w, b):
    # Gets the corresponding x_2 value given x_1 and the decision boundary of a
    # perceptron model. Note this only works for a 2D perceptron.
    if w[1] == 0.0:
        denom = 1e-6
    else:
        denom = w[1]

    return (-w[0] * x_1 - b) / denom


def plot_perceptron(w, b, ax):
    # This method plots a perceptron decision boundary line. Note this only works for
    # 2D perceptron.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_2s = [boundary(x_1, w, b) for x_1 in xlim]
    ax.plot(xlim, x_2s)
    if predict([xlim[0], ylim[0]], w, b) == -1:
        ax.fill_between(xlim, ylim[0], x_2s, facecolor="red", alpha=0.5)
    else:
        ax.fill_between(xlim, x_2s, ylim[-1], facecolor="red", alpha=0.5)
