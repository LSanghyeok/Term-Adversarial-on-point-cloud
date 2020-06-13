from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader


def get_dataloader(num_points, b_size, name='10'):
    path = 'ModelNet'+name
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(
        'dataset/' + path,
        name=name,
        train=True,
        transform=transform,
        pre_transform=pre_transform)
    test_dataset = ModelNet(
        'dataset/' + path,
        name=name,
        train=False,
        transform=transform,
        pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)
    
    return train_loader, test_loader

