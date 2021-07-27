import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import DNN_Landmark_Model, CNN


class handDetector():
    """
    A hand detector class.
    :param : model = DNN_Landmark_Model / Mobile_Net / ResNet to choose the backbone model
    """

    def __init__(self, model_used="mobilenet", mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model = model_used
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                       'W', 'X', 'Y', 'Z', 'delete', 'space']

        # deep network landmark model
        if model_used == "landmark":
            self.model = DNN_Landmark_Model()
            self.model.load_state_dict(torch.load(
                'trained_landmarks_models/normalized_DNN_landmarks_model.pth'))
        elif model_used == "mobilenet":
            print("Came here")
            self.model = CNN(backbone="mobilenet_v2")
            self.model = torch.load(
                '/Users/asinha4/kaggle/HandGestureRecognition/HandLandmarkModel/trained_cnn_models/sl_recognition_5_0.247_0.925_mobilenet.pth',
                map_location='cpu')
        elif model_used == "resnet":
            self.model = CNN(backbone="resnet50")
            self.model = torch.load(
                'trained_cnn_models/sl_recognition_25_0.15_0.952_resnet.pth',
                map_location='cpu')

        self.model.eval()

        # logistic regression landmark model
        # self.model = pickle.load(open('LogReg_landmarks_model.sav', 'rb'))

    def findHands(self, img, draw=True):
        """
        Find hand landmarks
        :param img: image containing hand
        :param draw: True if landmark points are drawn
        :return: input image with or without landmarks drawn
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findBox(self, img, draw=True):
        """
        Find bounding box around the hand
        :param img: image containing hand
        :param draw: True if bounding box is drawn
        :return: input image with or without bounding box
        """
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

            self.cropped_image = img[ymin - 80:ymax + 80, xmin - 80:xmax + 80]

            if draw:
                cv2.rectangle(img, (xmin - 80, ymin - 80), (xmax + 80,
                                                            ymax + 80),
                              (0, 255, 0), 2)

        return img

    def detect_gesture(self, model="landmark"):
        """
        Find gesture using hand landmarks
        :param: model = landmark / resnet / mobilenet to choose the base model
        :return: gesture label
        """
        if not self.results.multi_hand_landmarks:
            gesture = 'nothing'

        else:
            if (model == "landmark"):
                lm_array = []
                myHand = self.results.multi_hand_landmarks[0]

                # get landmark points relative to landmark 0
                for id, lm in enumerate(myHand.landmark):
                    if id != 0:
                        pass
                    else:
                        ref_x = lm.x
                        ref_y = lm.y

                    lm_array.append(lm.x - ref_x)
                    lm_array.append(lm.y - ref_y)

                lm_array = np.array(lm_array)

                # scale the detected landmark points in x and y direction
                lm_array[0::2] = (lm_array[0::2] - np.min(lm_array[0::2])) / (
                        np.max(
                            lm_array[0::2]) - np.min(lm_array[0::2]))
                lm_array[1::2] = (lm_array[1::2] - np.min(lm_array[1::2])) / (
                        np.max(
                            lm_array[1::2]) - np.min(lm_array[1::2]))

                # predict gesture using deep network landmark model
                output = self.model(torch.tensor(lm_array).type(torch.FloatTensor))
                gesture_id = np.argmax(output.detach().numpy())

            elif (model == "mobilenet" or model == "resnet"):
                print("shape of cropped image ", self.cropped_image.shape)
                _image = np.array(self.cropped_image)
                imgRGB = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(imgRGB)
                # im.save("output/video_output{}.jpg".format(str(time.time() * 100))) SAVE IMAGES TO TEST

                # resize the image and convert to tensor
                train_transforms = transforms.Compose(
                    [transforms.Resize((224, 224)), transforms.ToTensor(), ])
                imgRGB = transforms.ToPILImage()(imgRGB)
                image = train_transforms(imgRGB)

                # add a dimension to image for batch
                image = image[None, :]
                # image = image.permute(0, 3, 1, 2)
                # image = image.type(torch.FloatTensor)
                output = self.model(image)
                gesture_id = np.argmax(output.detach().numpy())

            # predict gesture using logistic regression landmark model
            # gesture_id = self.model.predict(lm_array.reshape(1,-1))[0]

            # convert detected gesture to label
            gesture = self.labels[gesture_id]
        return gesture


def run_hand_gesture_recognition(train_model):
    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 60  # Brightness control (0-100)
    cap = cv2.VideoCapture(0)
    detector = handDetector(train_model)

    while True:
        # read image from video
        success, img = cap.read()

        # find hand landmarks
        if(train_model == "landmark"):
            img = detector.findHands(img)
        elif(train_model == "mobilenet" or train_model == 'resnet'):
            img = detector.findHands(img, draw=False)
            img = detector.findBox(img)

        # find gesture
        # params: model="landmark" / model="mobilenet" / model = "resnet"
        gesture = detector.detect_gesture(model=train_model)

        # Select the predicted gesture image to show as reference
        overlay = cv2.resize(cv2.imread(f'reference_images/'
                                        f'{gesture}_test.jpg'), (200, 200))
        overlay = cv2.convertScaleAbs(overlay, alpha=alpha, beta=beta)

        # Select the region in the background where we want to add the image
        # and add the gesture using cv2.addWeighted()
        added_image = cv2.addWeighted(img[0:200, -200:, :], 0,
                                      overlay, 1, 0)

        # Change the region with the reference gesture image
        img[0:200, -200:] = added_image

        # Add the gesture as text
        cv2.putText(img, gesture, (img.shape[1] - 120, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    run_hand_gesture_recognition(train_model="resnet")
