import os
import cv2
import numpy as np
import pandas as pd

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
train_persons = ['P1', 'P2', 'P3', 'P4']
valid_persons = ['P5']

img_nums = np.arange(1, 9)

train_path = 'concat_train/'
valid_path = 'concat_valid/'

path = 'cropped_train/'

for label in labels:
    for person in train_persons:
        for num in img_nums:
            img_id = label + '_' + person + '_00' + str(num) + '.jpg'

            file_1 = path + 'Right_CAM_' + img_id
            file_2 = path + 'Left_CAM_' + img_id
            file_3 = path + 'Front_CAM_' + img_id
            file_4 = path + 'Below_CAM_' + img_id

            if os.path.exists(file_1) and os.path.exists(
                    file_2) and os.path.exists(file_3) and os.path.exists(
                    file_4):
                img1 = cv2.imread(file_1)
                img2 = cv2.imread(file_2)
                img3 = cv2.imread(file_3)
                img4 = cv2.imread(file_4)

                im_v_1 = cv2.vconcat([img1, img2])
                im_v_2 = cv2.vconcat([img3, img4])

                im_v = cv2.hconcat([im_v_1, im_v_2])

                cv2.imwrite(train_path + img_id, im_v)

path = 'cropped_valid/'

for label in labels:
    for person in valid_persons:
        for num in img_nums:
            img_id = label + '_' + person + '_00' + str(num) + '.jpg'

            file_1 = path + 'Right_CAM_' + img_id
            file_2 = path + 'Left_CAM_' + img_id
            file_3 = path + 'Front_CAM_' + img_id
            file_4 = path + 'Below_CAM_' + img_id

            if os.path.exists(file_1) and os.path.exists(
                    file_2) and os.path.exists(file_3) and os.path.exists(
                    file_4):
                img1 = cv2.imread(file_1)
                img2 = cv2.imread(file_2)
                img3 = cv2.imread(file_3)
                img4 = cv2.imread(file_4)

                im_v_1 = cv2.vconcat([img1, img2])
                im_v_2 = cv2.vconcat([img3, img4])

                im_v = cv2.hconcat([im_v_1, im_v_2])

                cv2.imwrite(valid_path + img_id, im_v)

files = []

for file in os.listdir(train_path):
    if file != 'labels.csv':
        label = file.split('_')[0]
        files.append([file, label])

print(files[:5])

df = pd.DataFrame(files, columns=['files', 'target']).to_csv(
    train_path + 'labels.csv')

files = []

for file in os.listdir(valid_path):
    if file != 'labels.csv':
        label = file.split('_')[0]
        files.append([file, label])

print(files[:5])

df = pd.DataFrame(files, columns=['files', 'target']).to_csv(
    valid_path + 'labels.csv')
