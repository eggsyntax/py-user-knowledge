"""
Calculating/plotting Precision-Recall Curve and area under PRC on multi-class classification tasks.
Interpretation: https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import plotly.graph_objects as go
import loss

def plot_prc_and_calculate_auprc_arrays(y_true, y_scores, classes, category):
    """Inner fuction that takes nparrays and classes as input."""

    # Plot setup
    fig = go.Figure()
    
    # Calculate PRC and AUPRC for each class
    for i, clas in enumerate(classes):

        # Prepare binary true labels and scores for the current class
        true_binary = y_true[:, i]
        scores = y_scores[:, i]
        
        # Calculate the baseline AUPRC based on the fraction of positives
        baseline = np.mean(true_binary)
        if baseline == 0: # Skip classes with no positive examples
            continue

        # Calculate precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(true_binary, scores)
        
        # Calculate AUPRC using average_precision_score
        # Using average_precision_score over auc based on https://towardsdatascience.com/the-wrong-and-right-way-to-approximate-area-under-precision-recall-curve-auprc-8fd9ca409064
        auprc = average_precision_score(true_binary, scores)
        
        # Add trace for each class
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Class {clas} AUPRC: {auprc:.2f} (Baseline: {baseline:.2f})'))
    
    # Finalizing the plot
    fig.update_layout(title=f'Precision-Recall Curve: {category} ({len(y_scores)} profiles).',
                      xaxis_title='Recall',
                      yaxis_title='Precision',
                      yaxis=dict(scaleanchor="x", scaleratio=1),
                      xaxis=dict(constrain='domain'),
                      width=700, height=500)
    fig.show()


def plot_prc_and_calculate_auprc(matches, category_tokens, category):
    """Plot the Precision-Recall Curve and calculate the Area Under the Precision-Recall Curve (AUPRC) for each class."""
    print(f'plot_prc_and_calculate_auprc for {category} on {len(matches)} matches.')
    y_true, y_scores = loss.np_arrays(matches, category_tokens, category)
    if y_true is None or y_scores is None:
        return None
    # Number of classes
    classes = loss.get_classes(category_tokens)
    if classes is None:
        return None
    print(f'Classes: {classes}') # XXX
    return plot_prc_and_calculate_auprc_arrays(y_true, y_scores, classes, category)
    

# Example data
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y_scores = np.array([[0.7 , 0.15, 0.15], [0.2, 0.2, 0.6 ], [0.25, 0.25, 0.5 ], [0.65, 0.2 , 0.15], [0.15, 0.7 , 0.15], [0.1 , 0.2 , 0.7 ], [0.8 , 0.1 , 0.1 ], [0.25, 0.5 , 0.25]])
category_tokens = ['bis', 'gay', 'straight']

plot_prc_and_calculate_auprc_arrays(y_true, y_scores, category_tokens, 'sexuality')