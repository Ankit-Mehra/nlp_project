# get precision, recall, and f1-score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  ConfusionMatrixDisplay


def get_precision_recall_f1score(y_test, y_pred_test):
    """
    Get precision, recall, and f1-score
    """
    print(classification_report(y_test, y_pred_test))

# confusion matrix
def get_confusion_matrix(y_test, y_pred_test):
    """
    Get confusion matrix
    """
    return confusion_matrix(y_test, y_pred_test)

# plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred_test):
    """
    Plot the confusion matrix
    """
    labels = ['Negative', 'Neutral', 'Positive']

    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
