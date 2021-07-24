import os
import cv2
import mediapipe as mp
import pandas as pd

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# create train data
folder_train = 'asl_alphabet_train/'

files = []

# Image brightness and contrast control
alpha = 1.0  # Contrast control (1.0-3.0)
beta = 60  # Brightness control (0-100)

# define hand detection object
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
    for label_num, label in enumerate(labels):
        for file in os.listdir(folder_train + label):
            image = cv2.imread(folder_train + label + '/' + file)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                data = [label + '/' + file, label, label_num]
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        if id != 0:
                            pass
                        else:
                            ref_x = landmark.x
                            ref_y = landmark.y
                        data.append(landmark.x - ref_x)
                        data.append(landmark.y - ref_y)
                files.append(data)

data_columns = ['files', 'target', 'label']

for i in range(21):
    data_columns.append('x_'+ str(i))
    data_columns.append('y_' + str(i))

df = pd.DataFrame(files, columns=data_columns).to_csv(
    folder_train + 'labels.csv')

# Create test data
folder_test = 'asl_alphabet_test/'

files = []

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
    for label_num, label in enumerate(labels):
        image = cv2.imread(folder_test + '/' + label + '_test.jpg')
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            data = [label + '_test.jpg', label, label_num]
            for hand_landmarks in results.multi_hand_landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):
                    if id != 0:
                        pass
                    else:
                        ref_x = landmark.x
                        ref_y = landmark.y
                    data.append(landmark.x - ref_x)
                    data.append(landmark.y - ref_y)
            files.append(data)

df = pd.DataFrame(files, columns=data_columns).to_csv(
    folder_test + 'labels.csv')


