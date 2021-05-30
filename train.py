import os

import torch
import torch.utils.tensorboard as tb
import torchvision
from model import CNNClassifier, save_model, load_model
from utils import load_data, ConfusionMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = CNNClassifier().to(device)


def train(args):
    from os import path
    model = CNNClassifier()
    loss = torch.nn.CrossEntropyLoss()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_cnn'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_cnn'), flush_secs=1)

    global_step_train = 0
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.015, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20)

    path = '/Users/asinha4/kaggle/HandGestureRecognition'
    valid_path = '/Users/asinha4/kaggle/HandGestureRecognition'
    import inspect
    transform = eval(args.transform,
                     {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})
    print('loading train data...')
    trainloader = load_data(path, transform=transform, num_workers=4)
    print('loading val data...')
    validloader = load_data(valid_path, num_workers=4)



    if not os.path.exists('cnn.th'):
        epoch = 100
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
                train_logger.add_scalar("loss", train_loss, global_step=global_step_train)
            print(f'Running epoch={ep} with accuracy on train data = {train_confusionMatrix.global_accuracy}')
            train_logger.add_scalar("accuracy", train_confusionMatrix.global_accuracy, global_step=global_step_train)

            for i, validdata in enumerate(validloader, 0):
                images, labels = validdata
                valid_confusionMatrix.add(model(images).argmax(1), labels)

            print(f'Running epoch={ep} with accuracy on valid data = {valid_confusionMatrix.global_accuracy}')

            valid_logger.add_scalar("accuracy", valid_confusionMatrix.global_accuracy, global_step=global_step_train)

            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step_train)
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
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), Resize([256,256]), ToTensor()])')

    args = parser.parse_args()
    train(args)
