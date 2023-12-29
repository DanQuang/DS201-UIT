from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def compute_score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred, average= 'macro')
    precision = precision_score(y_true, y_pred, average= 'macro')
    recall = recall_score(y_true, y_pred, average= 'macro')
    f1 = f1_score(y_true, y_pred, average= 'macro')

    print("*"*30)
    print("Evaluating:")
    print(f"Accuracy: {accuracy:.10f}")
    print(f"Precision: {precision:.10f}")
    print(f"Recall: {recall:.10f}")
    print(f"F1-score: {f1:.10f}")

def compute_score_per_class(y_true, y_pred):
    print(classification_report(y_true= y_true, y_pred= y_pred))