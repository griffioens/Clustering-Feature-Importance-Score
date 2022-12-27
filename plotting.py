import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(font_scale=1.6, style='whitegrid')

def resultplot(datasets, which="both"):
    """Creates the outcome plots for the feature importance scores, either for kmeans, fuzzy c means, or for both in the same plot.

    Args:
        datasets (list): List of dataset names
        which (str, optional): Has three modes, Kmeans, Fuzzycmean, or both. Defaults to "both".
    """ 
    # Formatting plot
    sns.set_style('whitegrid')
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 7} 
    sns.set_context('paper', font_scale=1.8, rc=paper_rc)   

    # For progress bar
    numberplots = 0
    totalplots = len(datasets)

    #Creation of plot by dataset
    for dataset in datasets:
        # loading the performance scores dataset
        df = pd.read_excel('./error_scores/error_scores_'+dataset+'.xlsx')
        df = df.iloc[:,1:]

        # Creating the lines in the plot
        if which == "Kmeans" or which == "both":
            sns.lineplot(x = range(0,len(df)), y=df["SHAP KMeans"], marker='o', markersize=10, label = "SHAP KMeans")
            sns.lineplot(x = range(0,len(df)), y=df["PFBI KMeans"], marker='o', markersize=10, label = "PBFI KMeans")
            plt.gca().fill_between(range(0,len(df)), df["SHAP KMeans"], df["PFBI KMeans"], alpha=0.2, color='grey')

        if which == "Fuzzycmeans" or which == "both":
            sns.lineplot(x = range(0,len(df)), y=df["SHAP FuzzyCMeans"], marker='D', markersize=10, label = "SHAP FuzzyCMeans")
            sns.lineplot(x = range(0,len(df)), y=df["PBFI FuzzyCMeans"], marker='D', markersize=10, label = "PBFI FuzzyCMeans")
            plt.gca().fill_between(range(0,len(df)), df["SHAP FuzzyCMeans"], df["PBFI FuzzyCMeans"], alpha=0.2, color='pink')


        # Creating the other items of the plot
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.xticks(np.arange(0, len(df), 1.0))

        plt.xlabel('rank', fontsize=18)
        plt.ylabel('performance', fontsize=18)

        # Saving the folder    
        plt.savefig("./plots/"+dataset+which+'.pdf', bbox_inches='tight')
        numberplots +=1
        print(f"Completed figure {numberplots} out of {totalplots} total")
        plt.close()
