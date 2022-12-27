import numpy as np
import pandas
from sklearn.metrics import accuracy_score

def f_importance(centroids, X):
    """This function converts a dataframe of the original dataset into a dataframe with the same size, 
    but containing feature importance scores for each feature, using the proposed method.

    Args:
        centroids (Prototypes of model): The prototypes of the Kkmeans algorithm, acquired by kmeans.cluster_centers_
        X (df): A dataframe of the original dataset

    Returns:
        Feature importance: The feature importance scores, using the proposed method
    """   
    f_values = np.zeros(X.shape[1])
    for xi in enumerate(centroids):
        for xj in enumerate(centroids):
            f_values += np.abs(xi[1]-xj[1])
    return f_values/np.sum(f_values)


def kmeans_performance(df, labels, kmeansobject, original_y):
    """This function executes the process of removing a feature, and recalculating the centroids, then calculating the accuracy. 
    It returns a list of accuracy scores after progressively removing each feature.

    Args:
        df (Pandas Dataframe): A pandas dataframe containing the original data of the datset
        labels (List): List of labels in order of importance
        kmeansobject (object): The SKlearn KMeans object, fitted on the original dataset
        original_y (list): The membership values when no features have been removed in the dataset

    Returns:
        list: Returns a list of accuracy numbers, after removal of each feature.
    """    
    errors_kmeans = [1.0]
    exclude_kmeans = []

    for fi in labels:

        df_temp = df.copy()
        exclude_kmeans.append(fi)

        for column in exclude_kmeans:
            df_temp[column] = df_temp[column].mean()
        
        membership_conv = kmeansobject.predict(df_temp.values)
        membership_conv = membership_conv.tolist()

        membership_conv = membership_conversion(membership_conv)
        y_pred = np.argmax(membership_conv, axis=1)
        errors_kmeans.append(accuracy_score(original_y, y_pred))
    return errors_kmeans


def membership_conversion(membership):
    """Reformats an integer list of memberships of datapoints to clusters to an array of binary membership
    Example:
    membership_conversions([0,2], where the first point belongs to cluster 0, and the second point to cluster 2, would return array([[1,0,0],[0,0,1]])

    Args:
        membership (list): List containing which cluster prototype each value belongs to, eg. [5 5 5 1 0 7 0 0 0 5 5 0 0 5 0 5 0 5 5 0 0 5 1]

    Returns:
        Numpy array: A numpy array of values in dummisized format.
    """    
    cluster_nrs = max(membership)+1 #addition of 1, because of the existance of the zeroth cluster
    for index, num in enumerate(membership):
        insertion = np.zeros(cluster_nrs)
        insertion[num] = 1
        membership[index] = insertion
    membership = np.asarray(membership)
    return membership