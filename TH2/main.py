import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import dataset, utils
from model.FashionMNISTModel import FashionMNISTModel
from torch import optim
from torchmetrics import Accuracy, F1Score, Precision, Recall
from tqdm.auto import tqdm

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# Load data
train_dataset = dataset.FashionMNISTDataset("D:/Project/DS201/TH2/Data/train-images-idx3-ubyte",
                                            "D:/Project/DS201/TH2/Data/train-labels-idx1-ubyte")

test_dataset = dataset.FashionMNISTDataset("D:/Project/DS201/TH2/Data/t10k-images-idx3-ubyte",
                                           "D:/Project/DS201/TH2/Data/t10k-labels-idx1-ubyte")

# Load Dataloader
train_dataloader = DataLoader(train_dataset, 64, True, collate_fn= utils.collate_fn)
test_dataloader = DataLoader(test_dataset, 32, True, collate_fn= utils.collate_fn)

# Create model
model_0 = FashionMNISTModel().to(device)

# Loss and optimizer, Accuracy, Precision, Recall, F1-score
# SGD
optimizer = optim.SGD(model_0.parameters(),
                      lr = 0.01,
                      momentum= 0.5)
# # RMSrpop
# optimizer = optim.RMSprop(model_0.parameters(),
#                           lr= 0.01,
#                           momentum= 0.9)
# # Adam
# optimizer = optim.Adam(model_0.parameters(),
#                           lr= 1e-5)

loss_fn = nn.CrossEntropyLoss().to(device)
Acc_fn = Accuracy(task= "multiclass", num_classes= 10)
Prec_fn = Precision(task= "multiclass", num_classes= 10, average= "macro")
Recall_fn = Recall(task= "multiclass", num_classes= 10, average= "macro")
F1_score = F1Score(task= "multiclass", num_classes= 10)



for epoch in range(5):
    print(f"Epoch {epoch + 1}: ---------")

    # Train model
    loss_all = 0.
    model_0.train()
    for batch, (X, y) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        # Forward
        y_logits = model_0(X)
        loss = loss_fn(y_logits, y)
        
        # # L2 Regularization - Use if need use L2
        # loss = loss_fn(y_logits, y) + model_0.l2_regularization()

        loss_all += loss.item()

        # Backward
        loss.backward()
        optimizer.step()

    loss_all /= len(train_dataloader)

    print("*"*30)
    print("Training:")
    print(f"Loss: {loss_all:.5f}")

    # Evaluate

    ev_acc = 0.
    ev_prec = 0.
    ev_recall = 0.
    ev_f1 = 0.
    model_0.eval()
    with torch.inference_mode():

        for batch, (X, y) in tqdm(enumerate(test_dataloader)):
            X, y = X.to(device), y.to(device)
            y_logits = model_0(X)
            y_pred = torch.softmax(y_logits, dim = 1)

            ev_acc += Acc_fn(y, y_pred.argmax(dim = 1))
            ev_prec += Prec_fn(y, y_pred.argmax(dim = 1))
            ev_recall += Recall_fn(y, y_pred.argmax(dim = 1))
            ev_f1 += F1_score(y, y_pred.argmax(dim = 1))      

        ev_acc /= len(test_dataloader)
        ev_prec /= len(test_dataloader)
        ev_recall /= len(test_dataloader)
        ev_f1 /= len(test_dataloader)

        print("*"*30)
        print("Evaluating:")
        print(f"Accuracy: {ev_acc:.5f}")
        print(f"Precision: {ev_prec:.5f}")
        print(f"Recall: {ev_recall:.5f}")
        print(f"F1-score: {ev_f1:.5f}")