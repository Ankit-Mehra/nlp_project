"""
Main script for comparison of models
"""
import matplotlib.pyplot as plt
import pandas as pd
from comparison.comparison import make_comparison_data, metrics_comparison_table
from comparison.plot_matrix import plot_confusion_matrix

def main():
    """
    Main function
    """
    # Make comparison data
    make_comparison_data()

    # Make metrics comparison table
    metrics_table = metrics_comparison_table()
    print(metrics_table)

    # Plot confusion matrix
    # Load the comparison data
    df_comparison = pd.read_csv('./data/comparison_data.csv')
    labels = ['Negative', 'Positive', 'Neutral']

    # Create figure for plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Comparison of Confusion Matrices')

    # Plot confusion matrices for each model
    plot_confusion_matrix(axes[0, 0],df_comparison['sentiment'],
                          df_comparison['predicted_blob'], 'TextBlob', labels)
    plot_confusion_matrix(axes[0, 1], df_comparison['sentiment'],
                          df_comparison['predicted_vader'], 'VADER', labels)
    plot_confusion_matrix(axes[1, 0], df_comparison['sentiment'],
                          df_comparison['predicted_svc'], 'SVC', labels)
    plot_confusion_matrix(axes[1, 1], df_comparison['sentiment'],
                          df_comparison['predicted_logistic'],
                          'Logistic Regression', labels)
    plot_confusion_matrix(axes[2, 0], df_comparison['sentiment'],
                          df_comparison['predicted_state_svc'],
                          'State of the Art SVC', labels)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig('./plots/comparison_confusion_matrix_art.png')
    plt.show()

if __name__ == "__main__":
    main()
