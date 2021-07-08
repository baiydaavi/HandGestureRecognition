import os
import torch
import torch.utils.tensorboard as tb
import torchvision
from model import CNNClassifier, save_model, load_model
from utils import load_data, ConfusionMatrix
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import os
from tqdm import tqdm
from time import sleep
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = CNNClassifier().to(device)


def load_split_train_test(datadir, batch_size, valid_size=.2):


    train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                                       transforms.GaussianBlur(kernel_size=501), transforms.ToTensor(), ])
    test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return trainloader, testloader

batch_size = 32
trainloader, testloader = load_split_train_test(data_dir, batch_size, .18)
print("Train Size:", len(trainloader) * batch_size, ", No of bacthes:", len(trainloader))
print("Test Size:", len(testloader) * batch_size, ", No of bacthes:", len(testloader))
print("Classes:", trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def train(args):
    from os import path
    model = CNNClassifier()
    loss = torch.nn.CrossEntropyLoss()
    train_logger, valid_logger = None, None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_cnn'),
    #                                     flush_secs=1)
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_cnn'),
    #                                     flush_secs=1)

    global_step_train = 0
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.015, momentum=0.9,
    # nesterov=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9,
                                weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                           patience=20)

    # use cropped_train for non-concatenated images
    path = 'asl_alphabet_train'
    # use cropped_valid for non-concatenated images
    # valid_path = 'asl_alphabet_test'

    import inspect
    transform = eval(args.transform,
                     {k: v for k, v in
                      inspect.getmembers(torchvision.transforms) if
                      inspect.isclass(v)})
    print(args.transform)
    print('loading train data...')
    trainloader = load_data(path, transform=transform, num_workers=4)
    # print('loading val data...')
    # validloader = load_data(valid_path, num_workers=4)

    if not os.path.exists('cnn.th'):
        epoch = 20
        model.train()

        for ep in range(epoch):
            train_confusionMatrix = ConfusionMatrix()
            valid_confusionMatrix = ConfusionMatrix()
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                frw = model(images)
                train_confusionMatrix.add(frw.argmax(1), labels)
                train_loss = loss(frw, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                global_step_train += 1
                # train_logger.add_scalar("loss", train_loss,
                #                         global_step=global_step_train)
            print(
                f'Running epoch={ep} with accuracy on train data = '
                f'{train_confusionMatrix.global_accuracy}')
            # train_logger.add_scalar("accuracy",
            #                         train_confusionMatrix.global_accuracy,
            #                         global_step=global_step_train)


            # train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'],
            #                         global_step=global_step_train)
            scheduler.step(valid_confusionMatrix.global_accuracy)
        model.eval()
        save_model(model)
    else:
        model = load_model(model)


def check():
    model = CNNClassifier()
    model.load_state_dict(torch.load('/cnn.th'))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].type())
    # print(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), '
                                'RandomHorizontalFlip(p=0.9), '
                                'RandomVerticalFlip(p=0.9), RandomGrayscale('
                                'p=1.0), ToTensor()])')

    args = parser.parse_args()
    train(args)
