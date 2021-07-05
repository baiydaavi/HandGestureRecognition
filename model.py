import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models


class CNNClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.master = torch.nn.Sequential(torch.nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=2),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        torch.nn.init.xavier_normal_(self.master[0].weight)

        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                                          torch.nn.ReLU())

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU())

        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU())

        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.downsample1 = torch.nn.Sequential(torch.nn.Conv2d(16, 32, kernel_size=1),
                                               torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                                               torch.nn.ReLU())
        self.downsample2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=1),
                                               torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                                               torch.nn.ReLU())
        self.downsample3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=1),
                                               torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
                                               torch.nn.ReLU())

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 29),
        )

    def forward(self, x):
        # print(x.shape)
        # normalize image

        mu = torch.mean(torch.mean(x, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        sigma = torch.sqrt(torch.mean((x - mu) ** 2)) + 1e-8
        x -= mu
        x /= 4 * sigma

        # print("image", identity.shape)
        res1 = self.master(x)

        res2 = self.block1(res1)
        res2 = res2 + self.downsample1(res1)
        res2 = self.maxpool(res2)

        res3 = self.block2(res2)
        res3 = res3 + self.downsample2(res2)
        res3 = self.maxpool(res3)

        res4 = self.block3(res3)
        # print("4 ", res4.shape ,self.downsample3(res3).shape )
        res4 = res4 + self.downsample3(res3)

        res = self.maxpool(res4)
        # print("final shape : ", res.shape)
        res = res.mean(dim=[2, 3])
        res = self.classifier(res)
        return res

class CNNClassifier2(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=29, kernel_size=3):
        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        # z = self.network(x)
        res = self.classifier(z.mean(dim=[2, 3]))
        return res

class mobilenet(torch.nn.Module):
    backbone = 'mobilenet_v2'
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
    # print(model)

model_factory = {
    'cnn': CNNClassifier
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
