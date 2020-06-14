import torch
import torch.nn.functional as F

from torch_geometric.nn import PointConv, radius_graph, fps, global_max_pool, XConv, fps, global_mean_pool
from torch_geometric.transforms import RadiusGraph
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class PointNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()

        nn = Seq(Lin(3, 64), ReLU(), Lin(64, 64))
        self.conv1 = PointConv(local_nn=nn)

        nn = Seq(Lin(67, 128), ReLU(), Lin(128, 128))
        self.conv2 = PointConv(local_nn=nn)

        nn = Seq(Lin(131, 256), ReLU(), Lin(256, 256))
        self.conv3 = PointConv(local_nn=nn)

        self.lin1 = Lin(256, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, pos, batch):
        radius = 0.2
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv1(None, pos, edge_index))

        idx = fps(pos, batch, ratio=0.5)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 0.4
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv2(x, pos, edge_index))

        idx = fps(pos, batch, ratio=0.25)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv3(x, pos, edge_index))

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

class PointCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(PointCNN, self).__init__()

        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2)
        self.conv3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        self.conv4 = XConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2)

        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, num_classes)

    def forward(self, pos, batch):
        x = F.relu(self.conv1(None, pos, batch))

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

    
class Defense_PointNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(Defense_PointNet, self).__init__()
        
        features = [] 
        nn = Seq(Lin(3, 64), ReLU(), Lin(64, 64))
        features.append(PointConv(local_nn=nn))
        nn = Seq(Lin(67, 128), ReLU(), Lin(128, 128))
        features.append(PointConv(local_nn=nn))
        nn = Seq(Lin(131, 256), ReLU(), Lin(256, 256))
        features.append(PointConv(local_nn=nn))
        self.features = torch.nn.ModuleList(features)
        
        layers=[]
        layers.append(Lin(256, 256))
        layers.append(Lin(256, 256))
        layers.append(Lin(256, num_classes))
        self.classifier = torch.nn.ModuleList(layers)
        
        layers=[]
        layers.append(Lin(256, 256))
        layers.append(Lin(256, 2))
        self.discriminator = torch.nn.ModuleList(layers)
        
    def forward(self, pos, batch):
        radius = 0.2
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.features[0](None, pos, edge_index))

        idx = fps(pos, batch, ratio=0.5)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 0.4
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.features[1](x, pos, edge_index))

        idx = fps(pos, batch, ratio=0.25)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.features[2](x, pos, edge_index))

        x = global_max_pool(x, batch)
        feat=x

        x = F.relu(self.classifier[0](x))
        x = F.relu(self.classifier[1](x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier[2](x)
        
        x2 = F.relu(self.discriminator[0](feat))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.discriminator[1](x2)
        return F.log_softmax(x, dim=-1), F.log_softmax(x2, dim=-1)