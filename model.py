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