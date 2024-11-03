import matplotlib.pyplot as plt
from path import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_eval(df_gt, dir_out):
    score = df_gt['reconstruction_error']

    auc_score = roc_auc_score(df_gt['anomaly'], score)
    n_neg = len(df_gt[df_gt["anomaly"] == 0])
    n_pos = len(df_gt[df_gt["anomaly"] == 1])
    print(f'auc = {auc_score}')
    print(f'n_pos, n_neg = {n_pos}, {n_neg}')
    print(f'reconstruction_error 0 = {df_gt[df_gt["anomaly"] == 0]["reconstruction_error"].mean()}')
    print(f'reconstruction_error 1 = {df_gt[df_gt["anomaly"] == 1]["reconstruction_error"].mean()}')

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(df_gt['anomaly'], score)
    roc_auc = auc(fpr, tpr)

    # Find the threshold where FPR is closest to X
    print()
    target_fpr = 0.01
    idx = (np.abs(fpr - target_fpr)).argmin()
    tpr_at_target_fpr = tpr[idx]
    threshold_at_target_fpr = thresholds[idx]
    print(f"TPR at FPR = 0.01: {tpr_at_target_fpr}")
    print(f"Threshold at FPR = 0.01: {threshold_at_target_fpr}")

    target_tpr = 0.85
    idx = (np.abs(tpr - target_tpr)).argmin()
    fpr_at_target_tpr = fpr[idx]
    threshold_at_target_tpr = thresholds[idx]
    print(f"FPR at TPR = 0.85: {fpr_at_target_tpr}")
    print(f"Threshold at TPR = 0.85: {threshold_at_target_tpr}")
    print()

    # Plotting
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(Path(dir_out) / 'roc_curve.jpg')
    # plt.show()

    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(df_gt['anomaly'], score)

    # Calculate Average Precision Score
    average_precision = average_precision_score(df_gt['anomaly'], score)
    print(f'average_precision = {average_precision}')

    # Plotting
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.legend(loc='lower left')
    plt.savefig(Path(dir_out) / 'precision-recall.jpg')
    # plt.show()

    # Plotting
    reconstruction_error_0 = df_gt[df_gt["anomaly"] == 0]["reconstruction_error"]
    reconstruction_error_1 = df_gt[df_gt["anomaly"] == 1]["reconstruction_error"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(reconstruction_error_0, bins=50, density=True, label="reconstruction_error 0", alpha=.6, color="green")
    ax.hist(reconstruction_error_1, bins=50, density=True, label="reconstruction_error 1", alpha=.6, color="red")
    plt.title("Distribution of the reconstruction errors")
    plt.legend()
    plt.savefig(Path(dir_out) / 'distribution.jpg')
    # plt.show()
