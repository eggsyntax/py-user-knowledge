import plotly.express as px
import numpy as np

def generate_roc_and_analyze_skewness(data, category):
    y_true = []
    y_scores = []
    
    # Data Processing
    for item in data:
        if category in item and item[category]["match?"] is not None:
            match = item[category]["match?"]
            estimates = item[category]["estimate"]
            for label, score in estimates.items():
                score = float(score.strip('%')) / 100
                if label.strip().lower() == 'yes':  # Assuming 'Yes' is the positive class
                    if match:
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    y_scores.append(score)
    
    # ROC Curve Generation
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC Curve with Plotly
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={roc_auc:.2f}) for {category}',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()

    # Skewness Analysis
    pos_scores = [score for true, score in zip(y_true, y_scores) if true == 1]
    neg_scores = [score for true, score in zip(y_true, y_scores) if true == 0]

    pos_mean = np.mean(pos_scores)
    neg_mean = np.mean(neg_scores)

    if pos_mean > neg_mean:
        print(f"The classifier is skewed towards positive class in {category}.")
    else:
        print(f"The classifier is skewed towards negative class in {category}.")

    return {'roc_auc': roc_auc, 'pos_mean_score': pos_mean, 'neg_mean_score': neg_mean}
