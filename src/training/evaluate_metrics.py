from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

data = torch.load(
    'data/graph/evaluation_outputs.pt',
    weights_only=False
)

test_probs = data['test_probs'].detach().cpu().numpy()
test_labels = data['test_labels'].detach().cpu().numpy()

test_pred = (test_probs >= 0.5).astype(int)

accuracy = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred, zero_division=0)
recall = recall_score(test_labels, test_pred, zero_division=0)
f1 = f1_score(test_labels, test_pred, zero_division=0)
roc_auc = roc_auc_score(test_labels, test_probs)
pr_auc = average_precision_score(test_labels, test_probs)

cm = confusion_matrix(test_labels, test_pred)

tn, fp, fn, tp = cm.ravel()

print('=' * 70)
print('STEP 109 - FINAL TEST EVALUATION')
print('=' * 70)

print()
print('TEST SAMPLES:', len(test_labels))
print('Positive labels:', int(test_labels.sum()))
print('Negative labels:', int((test_labels == 0).sum()))

print()
print('FINAL METRICS')
print('-' * 70)
print(f'Accuracy:   {accuracy:.6f}')
print(f'Precision:  {precision:.6f}')
print(f'Recall:     {recall:.6f}')
print(f'F1 Score:   {f1:.6f}')
print(f'ROC-AUC:    {roc_auc:.6f}')
print(f'PR-AUC:     {pr_auc:.6f}')

print()
print('CONFUSION MATRIX')
print('-' * 70)
print(cm)

print()
print('TN:', tn)
print('FP:', fp)
print('FN:', fn)
print('TP:', tp)

print()
print('Prediction distribution')
print('Predicted negative:', int((test_pred == 0).sum()))
print('Predicted positive:', int((test_pred == 1).sum()))

print()
print('Probability statistics')
print('Min:', float(test_probs.min()))
print('Max:', float(test_probs.max()))
print('Mean:', float(test_probs.mean()))
print('Median:', float(np.median(test_probs)))

status = (
    len(test_labels) == 740
    and len(test_probs) == 740
    and not np.isnan(test_probs).any()
    and not np.isinf(test_probs).any()
    and 0.0 <= roc_auc <= 1.0
    and 0.0 <= pr_auc <= 1.0
)

print()
print('=' * 70)
print(
    'STEP 109 STATUS:',
    'PASSED' if status else 'FAILED'
)
print('=' * 70)

out = Path('data/graph/test_metrics.pt')

torch.save(
    {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    },
    out
)

print('Metrics saved:', out)
