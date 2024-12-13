import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys
from scipy.stats import chi2
orig_sys_path = sys.path[:]
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0 , dirname)
from train import *
sys.path = orig_sys_path

def visualize_importances(importances, title="Average Feature Importances"):
    """
    Visualizes the feature importances as a bar chart.

    Parameters:
    - importances (pd.DataFrame or np.ndarray): A matrix or 2D array of feature importances.
    - title (str, optional): Title of the plot. Defaults to "Average Feature Importances".

    Returns:
    - matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig = plt.figure(figsize=(12,6))
    importances.sort_values(ascending=False)[:20].plot(kind='bar')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.title(title)
    plt.show()
    
    return fig

def interpret(model, x, x_pred, savedir='', plot=True):
    """
    This function interprets the model by plotting and optionally saving feature importances
    of features, genes, and pathway levels based on the input data 'x'.

    Parameters:
    - model: The model for which interpretation is to be done.
    - x: Input data used for calculating feature importances.
    - savedir (str, optional): The directory where to save the plot images. If the path does
                               not exist, it will not save. Defaults to '' (does not save).

    Returns:
    - dict: A dictionary containing model layers' importance data.
    
    Raises:
    - FileNotFoundError: If the save directory does not exist and saving is attempted.
    """
    # Check if saving the plots is required and possible
    if os.path.exists(savedir) & plot:
        save_plots = True
    else:
        print('Save Path Not Found - Plots will not be saved')
        save_plots = False

    model_layers_importance = {}
    model_layers_importance_fig = {}

    # Calculate and visualize feature importance
    model_layers_importance['Features'] = model.deepLIFT_feature_importance(x)
    if plot :
        model_layers_importance_fig['Features'] = visualize_importances(
            model_layers_importance['Features'].mean(axis=0), title="Average Feature Importances")

    # Calculate and visualize layer-wise importance
    layer_importance = model.layerwise_importance(x, x_pred)
    for i, layer in enumerate(layer_importance):
        layer_title = f"Pathway Level {i} Importance" if i > 0 else "Gene Importance"
        model_layers_importance[layer_title] = layer
        if plot : 
            model_layers_importance_fig[layer_title] = visualize_importances(
                layer.mean(axis=0).abs(), title=f"Average {layer_title}")

    # Save the figures if the save directory is valid
    if save_plots :
        for name, fig in model_layers_importance_fig.items():
            # Ensure filename does not have any spaces and is properly formatted
            filename = name.replace(' ', '_')
            fig_path = os.path.join(savedir, filename)
            fig.savefig(fig_path, bbox_inches='tight')

    return model_layers_importance


def evaluate_interpret_save(model, test_dataset, path, n_classes, target_names):
    """
    Evaluates a model using a DataLoader for the test dataset, interprets feature importances, and saves
    the results including plots of confusion matrix and ROC curve, and prediction probabilities to a specified path.
    
    Parameters:
    - model: Trained model to be evaluated.
    - test_dataset (DataLoader): DataLoader containing the testing data.
    - path (str): Directory path where the evaluation and interpretation results will be saved.
    
    Raises:
    - FileNotFoundError: If the directory does not exist, it attempts to create it.
    """
    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Using list comprehensions to extract data and targets from the test_dataset
    data_batches, target_batches = zip(*[(data, target) for data, target in test_dataset])

    # Convert lists of batches into single tensors
    test_data = torch.cat(data_batches, dim=0)
    test_targets = torch.cat(target_batches, dim=0)
    
    # Getting predictions and probabilities from the model
    actuals, predictions = get_predictions(model.to('cpu'), test_dataset)

    # Compute the Confusion Matrix and save it
    cm = plot_confusion_matrix(actuals, predictions)
    cm_path = os.path.join(path, 'Confusion_Matrix.jpeg')
    cm.savefig(cm_path, bbox_inches='tight')

    # Computing AUC-ROC metrics and save the ROC Curve plot
    actuals, probs = get_probabilities(model.to('cpu'), test_dataset)
    auc = roc_auc_score(actuals, probs , multi_class='ovr')
    print("AUC Score:", auc)

    
    roc = plot_roc_curve(np.array(actuals), np.array(probs) , n_classes = n_classes , target_names = target_names)
    roc_path = os.path.join(path, 'ROC_Curve.jpeg')
    roc.savefig(roc_path, bbox_inches='tight')
    
    # Save prediction probabilities and predictions
    torch.save(probs, os.path.join(path, 'prediction_probabilities.pt'))
    torch.save(predictions, os.path.join(path, 'predictions.pt'))
    
    # Interpret the model (feature importances) and save the results
    model_importances = interpret(model, test_data, savedir=path)
    
    for name, importance in model_importances.items():
        # Save each layer's importances as a CSV file
        filename = name.replace(' ', '_')
        csv_path = os.path.join(path, f'{filename}.csv')
        importance.to_csv(csv_path)

def p_hit(S, r, hits, N_R, i, p):
    # Ensure i is within the bounds of the sequence
    if i >= len(r):
        raise IndexError("Index i is out of the bounds of the ranked scores.")

    # Ensure N_R is not zero to prevent division by zero
    if N_R == 0:
        raise ValueError("Sum of absolute values raised to the power of p is zero. Division by zero error.")
     
    # Calculate sum for indices j <= i (numerator)
    numerator = np.sum(np.abs(r[:i][hits[:i]])**p) # min(i+1, len(S)) handles index bounds
    
    # Calculate P_hit(S, i)
    p_hit_value = numerator / N_R
    
    return p_hit_value
    
def p_miss(S, hits , total_elements, i):

    # Validate if i is within the reasonable range
    if i > total_elements:
        raise IndexError("Index i exceeds the total number of elements.")
    
    N_H = len(S)  # The number of hits or included elements
    N_minus_N_H = total_elements - N_H  # The number of elements that are not hits
    
    # Handle zero division case
    if N_minus_N_H == 0:
        raise ValueError("No elements left outside of S up to index i; (N-N_H) is zero.")

    # Calculate the number of missing elements up to i (not in S, up to i)
    missing_count = sum(~hits[:i])
    
    # Calculate P_miss
    P_miss = missing_count / N_minus_N_H
    
    return P_miss
    
def calculate_es(S , r, hits, N_R, p=1, window=50) : 
    es = 0
    for i in range(len(S)-1) : 
        es_tmp = p_hit(S, r, hits, N_R, i , p) - p_miss(S, hits, len(r) ,i)
        es = es_tmp if np.abs(es_tmp) > es else es

    return es

def permutation_test(gene_scores, gene_list, n_perm=1000):
    S = np.array(gene_list)
    r = gene_scores.sort_values(ascending=False)

    r_index = np.array(r.index)
    r = r.to_numpy()
    hits = np.isin(r_index , S)
    # Calculate N_R (denominator of p_hits)
    p = 1
    N_R = np.sum(np.abs(r[hits])**p)
 
    real_es_pos = calculate_es(S , r, hits, N_R, p=p)
    print(f"Enrichment Score for Pos: {real_es_pos}")
    perm_es_scores = []

    for _ in range(n_perm):
        hits = np.isin(np.random.permutation(r_index) , S)
        perm_es = calculate_es(S, r, hits, N_R, p=p)
        perm_es_scores.append(perm_es)
    
    p_value =  np.sum(perm_es_scores > real_es_pos) / (1000 + 1)
    return real_es_pos, perm_es_scores, p_value

def pnet_model_significance_testing(genes_z_scores, gene_set) : 
    for i , score in genes_z_scores.iterrows() : 
        real_es_pos, perm_es_scores, p_value_pos = permutation_test(score, gene_set)
    
        print(f'The Observed Effect Size (ES) of genes related to outcome is {real_es_pos} with significance p-value {p_value_pos}')

        # Plotting the permutation ES scores
        plt.hist(perm_es_scores, bins=30, alpha=0.75, label='Permutation ES')
        plt.axvline(x=real_es_pos, color='red', label='Observed ES')
        plt.legend()
        plt.title('Permutation Test for Gene Set Enrichment')
        plt.xlabel('Enrichment Score (ES)')
        plt.ylabel('Frequency')
        plt.show()

def pnet_significance_testing(model_z_scores , key='mad') : 
    feature_significance = {}
    for layer in model_z_scores.keys() : 
        z_scores = model_z_scores[layer][key]
    
        # Calculate mean and SE across folds
        mean_z_scores = z_scores.mean(axis=0)
        se_z_scores = z_scores.std(axis=0, ddof=1) / np.sqrt(z_scores.shape[0])  # Standard Error of the mean
        
        # Calculating the Wald statistics
        wald_statistics = (mean_z_scores ** 2) / (se_z_scores ** 2)
        
        # Assuming chi-squared distribution to find p-values
        p_values = 1 - chi2.cdf(wald_statistics, df=1)
        
        # Result
        sig_feat = sum(p_values < 0.05/z_scores.shape[0])
        print(f"{sig_feat} Feautures have p-value < {0.05/z_scores.shape[0]} in Layer {layer}:")

        feature_significance[layer] = {'Features' : z_scores.columns,'p-values' : p_values}

    return feature_significance