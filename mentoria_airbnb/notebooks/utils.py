import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error



def plot_confusion_matrix(y_true, y_pred, 
                          normalize=False,
                          title=None,
                          cmap='Blues'):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix"
            
    cnf_matrix = confusion_matrix(y_true, y_pred)

    #classes = classes[ unique_labels(y_true, y_pred)]
    classes = unique_labels(y_true, y_pred)
    
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(cnf_matrix)
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots( figsize=(10,10) )
    
    im = ax.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cnf_matrix.shape[1]),
           yticks=np.arange(cnf_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i in range(cnf_matrix.shape[0]):
        for j in range(cnf_matrix.shape[1]):
            ax.text(j, i, format(cnf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cnf_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    fig.show()
    return ax
