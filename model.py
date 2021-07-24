import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DNN_Landmark_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(42, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 28)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.output(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self, backbone='mobilenet_v2'):
        self.backbone = backbone

        addLayers = False
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            if addLayers:
                model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1024),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(0.2),
                                               torch.nn.Linear(1024, 29),
                                               torch.nn.LogSoftmax(dim=1)
                                               )
            else:
                model.fc = torch.nn.Linear(2048, 29)
        elif backbone == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            if addLayers:
                model.classifier = torch.nn.Sequential(torch.nn.Linear(1280, 1024),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Dropout(0.2),
                                                       torch.nn.Linear(1024, 29),
                                                       torch.nn.LogSoftmax(dim=1)
                                                       )
            else:
                model.classifier = torch.nn.Linear(1280, 29)
