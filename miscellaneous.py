import pandas

def featuresorting(df):   
    """This function takes as input a dataframe with feature importance scores for each feature for the 4 models evaluated, and returns lists
    of feature names in order of importance.

    Args:
        df (Pandas Dataframe): A pandas dataframe with feature importance scores for each feature

    Returns:
        Labels Names: Returns lists of feature names in order of importance scores descending for Fuzzy C Means, K-means, SHAP Fuzzy C Means, SHAP K Means
    """    
    df.sort_values(by='PFBI FuzzyCMeans', axis=0, ascending=False, inplace=True)
    FuzzyCMeans_labels = df['features'].tolist()

    df.sort_values(by='PFBI KMeans', axis=0, ascending=False, inplace=True)
    KMeans_labels = df['features'].tolist()

    df.sort_values(by='SHAP FuzzyCMeans', axis=0, ascending=False, inplace=True)
    Shap_labels_fcm = df['features'].tolist()

    df.sort_values(by='SHAP KMeans', axis=0, ascending=False, inplace=True)
    Shap_labels_kmeans = df['features'].tolist()

    return FuzzyCMeans_labels, KMeans_labels, Shap_labels_fcm, Shap_labels_kmeans