import torch
from torch import nn 
from torch.utils.data import DataLoader
from data_utils import dataset, utils
from model.LeNet import LeNet
from model.GoogLeNet import GoogLeNet
from model.ResNet18 import ResNet18
from model.ResNet50 import ResNet50
from torch import optim
from evaluate import evaluate
from tqdm.auto import tqdm


# select device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

config = {
    'dataset': "CIFAR10", # MNIST, CIFAR10, PASCAL
    'model': "ResNet18",   # LeNet, GoogLeNet, ResNet18, ResNet50
    'num_classes': 10,
    'evaluate_per_class': False
}


# Load data
if config["dataset"] == "CIFAR10":
    # CIFAR10
    train_dataset = dataset.CIFAR10Dataset('/content/drive/MyDrive/DS201/Data/CIFAR10/data_train.pkl')
    test_dataset = dataset.CIFAR10Dataset('/content/drive/MyDrive/DS201/Data/CIFAR10/data_test.pkl')
elif config["dataset"] == "MNIST":
    # MNIST
    train_dataset = dataset.MNISTDataset("/content/drive/MyDrive/DS201/Data/MNIST/train-images.idx3-ubyte",
                                        "/content/drive/MyDrive/DS201/Data/MNIST/train-labels.idx1-ubyte")

    test_dataset = dataset.MNISTDataset("/content/drive/MyDrive/DS201/Data/MNIST/t10k-images.idx3-ubyte",
                                        "/content/drive/MyDrive/DS201/Data/MNIST/t10k-labels.idx1-ubyte")

# Load Dataloader
train_dataloader = DataLoader(train_dataset, 64, True, collate_fn= utils.collate_fn)
test_dataloader = DataLoader(test_dataset, 32, False, collate_fn= utils.collate_fn)

# load model
if config["model"] == "LeNet":
    model_0 = LeNet(config["num_classes"]).to(device)
elif config["model"] == "GoogLeNet":
    model_0 = GoogLeNet(config["num_classes"]).to(device)
elif config["model"] == "ResNet18":
    model_0 = ResNet18(config["num_classes"]).to(device)

# loss and optim
optimizer = optim.SGD(params= model_0.parameters(),
                  lr = 0.01,
                  momentum= 0.5)
loss_fn = nn.CrossEntropyLoss().to(device)

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
        loss_all += loss.item()

        # Backward
        loss.backward()
        optimizer.step()

    loss_all /= len(train_dataloader)

    print("*"*30)
    print("Training:")
    print(f"Loss: {loss_all:.10f}")

    # Evaluate
    ev_acc = 0.
    ev_prec = 0.
    ev_recall = 0.
    ev_f1 = 0.
    y_preds = []
    y_true = []
    model_0.eval()
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(test_dataloader)):
            X, y = X.to(device), y.to(device)
            y_logits = model_0(X)
            y_pred = torch.softmax(y_logits, dim = 1).argmax(dim= 1)
            y_preds += y_pred.tolist()
            y_true += y.tolist()

        if config['evaluate_per_class']:
            evaluate.compute_score_per_class(y_true= y_true, y_pred= y_preds)
        else:
            evaluate.compute_score(y_true= y_true, y_pred= y_preds)