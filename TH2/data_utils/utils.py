import torch

def collate_fn(Dataset):
    imgs = [torch.tensor(data["image"])/255 for data in Dataset]
    labels = [torch.tensor(data["label"]) for data in Dataset]

    return [torch.stack(imgs).type(torch.float32), torch.stack(labels)]