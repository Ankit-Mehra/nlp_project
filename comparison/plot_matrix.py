"""
Script to plot confusion matrix of textblob, vader, svc, logistic
in a 2x2 grid
"""
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(ax, true_labels, predicted_labels, model_name, labels):
    """
    Plot confusion matrix for a given model.

    Parameters:
    ax: Matplotlib axis object where the plot will be drawn.
    true_labels: Actual labels.
    predicted_labels: Labels predicted by the model.
    model_name: Name of the model.
    labels: Class labels to be displayed on the confusion matrix.
    """
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax)
    ax.set_title(model_name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

def main():
    """
    Main function
    """

if __name__ == "__main__":
    main()
