
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
import glob
from pathlib import Path
torch.manual_seed(1)
np.random.seed(1)
import os



train_path = '/Users/asinha4/kaggle/HandGestureRecognition/asl_alphabet_train'
test_path = '/Users/asinha4/kaggle/HandGestureRecognition/asl_alphabet_test'
classes = os.listdir(train_path)
train_images = []
train_labels = []
test_images = []
test_labels = []
for c in classes:
    flist = os.listdir(train_path + '/' + c)
    for file in flist:
        file_path = os.path.join(train_path, c, file)
        train_images.append(file_path)
        train_labels.append(c)
testflist = os.listdir(test_path)
for file in testflist:
    file_path = os.path.join(test_path, file)
    test_images.append(file_path)
    test_label = file.split('_')[0]
    test_labels.append(test_label)


train_images = pd.Series(train_images, name='file_paths')
train_labels = pd.Series(train_labels, name='labels')
test_images = pd.Series(test_images, name='file_paths')
test_labels = pd.Series(test_labels, name='labels')
train_df = pd.DataFrame(pd.concat([train_images, train_labels], axis=1))
test_df = pd.DataFrame(pd.concat([test_images, test_labels], axis=1))
train_df


# **Visualize Images**


plt.figure(figsize=(14, 10))
for i in range(20):
    idx = np.random.randint(0, len(train_df) - 1)
    plt.subplot(4, 5, i+ 1)
    img = train_df.iloc[idx, 0]
    plt.imshow(plt.imread(img))
    plt.title(train_df.iloc[idx, 1], size=10, color="black")
    plt.xticks([])
    plt.yticks([])

plt.show()

# **Split Train Dataframe into Train and Valid**


train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=0)
print(train_df['labels'].value_counts())
print(valid_df['labels'].value_counts())

# **Encode Labels**


lb = LabelEncoder()
train_df['encoded_labels'] = lb.fit_transform(train_df['labels'])
valid_df['encoded_labels'] = lb.fit_transform(valid_df['labels'])
test_df['encoded_labels'] = lb.fit_transform(test_df['labels'])
train_df


# **Dataset Class**


class ASLAlphabet(torch.utils.data.Dataset):
    def __init__(self, df=train_df, transform=transforms.Compose([transforms.ToTensor()])):
        self.df = df
        self.transform = transform

    def __len__(self):
        length = len(self.df)
        return length

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 2]
        label = torch.tensor(label)
        image = Image.open(img_path).convert('RGB')
        img = np.array(image)
        image = self.transform(image=img)["image"]
        return image, label


train_transforms = A.Compose([
    # No need to resize images since they are all 200 x 200 x 3
    A.GaussNoise(),
    A.Blur(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_dataset = ASLAlphabet(df=train_df, transform=train_transforms)
valid_dataset = ASLAlphabet(df=valid_df, transform=test_transforms)
test_dataset = ASLAlphabet(df=test_df, transform=test_transforms)

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# **Model Architecture**


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(32 * 100 * 100, 81),
            nn.Dropout(0.2),
            nn.BatchNorm1d(81),
            nn.LeakyReLU(81, 29))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32 * 100 * 100)
        x = self.fc(x)
        return x


# **Training**

# From https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


model = SimpleCNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, cooldown=2, verbose=True)

model = model.to(device)
criterion = criterion.to(device)

epochs = 30

total_train_loss = []
total_valid_loss = []
best_valid_loss = np.Inf
early_stop = EarlyStopping(patience=5, verbose=True)

for epoch in range(epochs):
    print('Epoch: ', epoch + 1)
    train_loss = []
    valid_loss = []
    train_correct = 0
    train_total = 0
    valid_correct = 0
    valid_total = 0
    for image, target in train_loader:
        model.train()
        image, target = image.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

        loss.backward()
        optimizer.step()

    for image, target in valid_loader:
        with torch.no_grad():
            model.eval()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)
            valid_loss.append(loss.item())
            _, predicted = torch.max(output.data, 1)
            valid_total += target.size(0)
            valid_correct += (predicted == target).sum().item()

    epoch_train_loss = np.mean(train_loss)
    epoch_valid_loss = np.mean(valid_loss)
    print(
        f'Epoch {epoch + 1}, train loss: {epoch_train_loss:.4f}, valid loss: {epoch_valid_loss:.4f}, train accuracy: {(100 * train_correct / train_total):.4f}%, valid accuracy: {(100 * valid_correct / valid_total):.4f}%')
    if epoch_valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'asl_model.pth')
        print('Model improved. Saving model.')
        best_valid_loss = epoch_valid_loss

    early_stop(epoch_valid_loss, model)

    if early_stop.early_stop:
        print("Early stopping")
        break

    lr_scheduler.step(epoch_valid_loss)
    total_train_loss.append(epoch_train_loss)
    total_valid_loss.append(epoch_valid_loss)

model.load_state_dict(torch.load('asl_model.pth'))

correct = 0
total = 0

with torch.no_grad():
    model.eval()
    for image, target in test_loader:
        image, target = image.to(device), target.to(device)

        output = model(image)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Test Accuracy: %d %%' % (
        100 * correct / total))