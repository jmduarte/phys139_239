import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, roc_curve


def plot_roc(labels, predictions):
    """Plots the ROC curve given a set of labels and predictions

    Args:
        labels (numpy.array)
        predictions (numpy.array)

    Returns:
        fig (matplotlib.figure.Figure)
    """

    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_score = auc(fpr, tpr)
    acc_score = accuracy_score(labels, predictions > 0.5)

    # plot TPR vs. FPR (ROC curve)
    fig = plt.figure()
    plt.plot(fpr, tpr, color="blueviolet", label=f"AUC = {auc_score*100:.2f}%, acc. = {acc_score*100:.2f}%")

    # make the plot readable
    plt.xlabel("False positive rate", fontsize=12)
    plt.ylabel("True positive rate", fontsize=12)
    plt.legend(frameon=False)
    plt.show()
    return fig
