import cv2
import mediapipe as mp
import time
from PIL import Image
from datetime import datetime
import time
from torchvision import datasets, transforms, models

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CNNClassifier, mobilenet, SimpleCNN


# now = datetime.now()
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(42, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 28)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        return x


class handDetector():

    def __init__(self, mode=False, maxHands=1, detectionCon=0.8, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        # self.model = SimpleCNN()
        self.model = mobilenet()
        # self.model.load_state_dict(torch.load('asl_model.pth', map_location='cpu'))
        # self.model.load_state_dict(torch.load('sl_recognition_6_0.3_0.907.pth', map_location='cpu'))

        self.model = torch.load('sl_recognition_6_0.3_0.907.pth', map_location='cpu')
        # self.model.cpu()
        self.model.eval()

        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                       'W', 'X', 'Y', 'Z', 'delete', 'space']

    def findHands(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findBox(self, img, draw=True):

        bbox = []

        if self.results.multi_hand_landmarks:
            xList = []
            yList = []
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                # if draw:
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            self.cropped_image = img[ymin-80:ymax+80, xmin-80:xmax+80]

            if draw:
                cv2.rectangle(img, (xmin - 80, ymin - 80), (xmax + 80,
                                                            ymax + 80), (0, 255, 0), 2)

        return img

    def detect_gesture(self):

        if not self.results.multi_hand_landmarks:
            gesture = 'Nothing'

        else:
            lm_array = []
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                if id != 0:
                    pass
                else:
                    ref_x = lm.x
                    ref_y = lm.y
                lm_array.append(lm.x - ref_x)
                lm_array.append(lm.y - ref_y)
            print("shape of cropped image ", self.cropped_image.shape)

            _image = np.array(self.cropped_image)
            imgRGB = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(imgRGB)
            im.save("output/video_output{}.jpg".format(str(time.time() * 100)))
            train_transforms = transforms.Compose(
                [transforms.Resize((224, 224)),transforms.ToTensor(), ])

            # image = torch.from_numpy(imgRGB)
            imgRGB = transforms.ToPILImage()(imgRGB)
            image = train_transforms(imgRGB)
            print("shape after transforms", image.shape)

            image = image[None, :]
            # image = image.permute(0, 3, 1, 2)
            # image = image.type(torch.FloatTensor)
            print("shape after processing", image.shape)
            # img = torch.tensor(image).type(torch.FloatTensor)
            output = self.model(image)
            gesture_id = np.argmax(output.detach().numpy())
            gesture = self.labels[gesture_id]
            # gesture = 'A'

        return gesture


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        img = detector.findBox(img)
        # print(img)

        gesture = detector.detect_gesture()
        cv2.putText(img, gesture, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 0, 255), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (1200, 70), cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()