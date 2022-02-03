import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score


def compute_asag_metrics(eval_pred):
    """

    :param eval_pred:
    :return:
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_score(labels, predictions, labels=list(range(3)), average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
    }
