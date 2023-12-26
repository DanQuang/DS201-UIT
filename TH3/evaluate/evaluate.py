from torchmetrics import F1Score, Accuracy, Precision, Recall
import torch

def compute_score(num_classes, y_true, y_pred):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Acc_fn = Accuracy(task= "multiclass", num_classes= num_classes, average= 'macro').to(device)
    Prec_fn = Precision(task= "multiclass", num_classes= num_classes, average= 'macro').to(device)
    Recall_fn = Recall(task= "multiclass", num_classes= num_classes, average= 'macro').to(device)
    F1_score = F1Score(task= "multiclass", num_classes= num_classes, average= 'macro').to(device)

    acc = Acc_fn(y_true, y_pred)
    prec = Prec_fn(y_true, y_pred)
    recall = Recall_fn(y_true, y_pred)
    f1 = F1_score(y_true, y_pred)

    return acc, prec, recall, f1