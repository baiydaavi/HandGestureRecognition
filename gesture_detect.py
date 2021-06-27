import cv2
import mediapipe as mp
import time
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class handDetector():

    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                       'W', 'X', 'Y', 'Z', 'delete', 'space']

        # deep network landmark model
        self.model = DNN_Landmark_Model()
        self.model.load_state_dict(torch.load(
            'trained_models/DNN_landmarks_model.pth'))
        self.model.eval()

        # logistic regression landmark model
        #self.model = pickle.load(open('LogReg_landmarks_model.sav', 'rb'))

    def findHands(self, img, draw=True):
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
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 80, ymin - 80), (xmax + 80,
                                                            ymax + 80),
                (0, 255, 0), 2)

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
            lm_array = np.array(lm_array)

            # deep network landmark model
            output = self.model(torch.tensor(lm_array).type(torch.FloatTensor))
            gesture_id = np.argmax(output.detach().numpy())

            # logistic regression landmark model
            #gesture_id = self.model.predict(lm_array.reshape(1,-1))[0]

            gesture = self.labels[gesture_id]

        return gesture


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

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
