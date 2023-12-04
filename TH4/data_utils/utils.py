import torch

def collate_fn(Dataset):
    imgs = [torch.tensor(data["image"]).unsqueeze(0) for data in Dataset]
    labels = [torch.tensor(data["label"]).unsqueeze(0) for data in Dataset]

    return [torch.cat(imgs).type(torch.float32), torch.cat(labels).type(torch.LongTensor)]