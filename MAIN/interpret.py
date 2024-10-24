import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0 , './MAIN/')
from train import *

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
    abs(np.sum(importances, axis=0)).sort_values(ascending=False)[:20].plot(kind='bar')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.title(title)
    plt.show()
    
    return fig

def interpret(model, x, savedir=''):
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
    if os.path.exists(savedir):
        save_plots = True
    else:
        print('Save Path Not Found - Plots will not be saved')
        save_plots = False

    model_layers_importance = {}
    model_layers_importance_fig = {}

    # Calculate and visualize feature importance
    model_layers_importance['Features'] = model.deepLIFT_feature_importance(x)
    model_layers_importance_fig['Features'] = visualize_importances(
        model_layers_importance['Features'], title="Average Feature Importances")

    # Calculate and visualize layer-wise importance
    layer_importance = model.layerwise_importance(x, 0)
    for i, layer in enumerate(layer_importance):
        layer_title = f"Pathway Level {i} Importance" if i > 0 else "Gene Importance"
        model_layers_importance[layer_title] = layer
        model_layers_importance_fig[layer_title] = visualize_importances(
            layer, title=f"Average {layer_title}")

    # Save the figures if the save directory is valid
    if save_plots:
        for name, fig in model_layers_importance_fig.items():
            # Ensure filename does not have any spaces and is properly formatted
            filename = name.replace(' ', '_')
            fig_path = os.path.join(savedir, filename)
            fig.savefig(fig_path, bbox_inches='tight')

    return model_layers_importance


def evaluate_interpret_save(model, test_dataset, path):
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
    actuals, predictions = get_predictions(model, test_dataset)
    actuals, probs = get_probabilities(model, test_dataset)

    # Compute the Confusion Matrix and save it
    cm = plot_confusion_matrix(actuals, predictions)
    cm_path = os.path.join(path, 'Confusion_Matrix.jpeg')
    cm.savefig(cm_path, bbox_inches='tight')

    # Computing AUC-ROC metrics and save the ROC Curve plot
    auc = roc_auc_score(actuals, probs)
    fpr, tpr, _ = roc_curve(actuals, probs)
    print("AUC Score:", auc)

    roc = plot_roc_curve(fpr, tpr, auc)
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