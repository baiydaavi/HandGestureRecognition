import os
from tqdm import tqdm
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


num_hands = 0

IMAGE_FOLDER = 'asl_alphabet_test/'
IMAGE_FILES = sorted(os.listdir(IMAGE_FOLDER))
alpha = 1.0  # Contrast control (1.0-3.0)
beta = 60  # Brightness control (0-100)
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=1,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(tqdm(IMAGE_FILES)):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(IMAGE_FOLDER+file), 1)
#     image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     # Print handedness and draw hand landmarks on the image.
#     #print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     #image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # cv2.imwrite('check_images/' + file[:3] + '.png', cv2.flip(annotated_image,
    #                                                          1))
    # else:
    #     num_hands += 1
    #     #print(file)

# print(num_hands, len(IMAGE_FILES))

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    image = cv2.imread('asl_alphabet_test/C_test.jpg')
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

cv2.imshow('Image', image)
cv2.imshow('Annotated', annotated_image)
cv2.waitKey()
