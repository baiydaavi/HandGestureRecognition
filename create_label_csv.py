import os

import pandas as pd

# os.chdir(r'cropped_train')
os.chdir(r'asl_alphabet_test')

folders_train = ['A/', 'B/', 'C/', 'D/',
                 'E/', 'F/',
                 'G/', 'H/', 'I/', 'J/',
                 'K/', 'L/',
                 'M/', 'N/', 'O/', 'P/',
                 'Q/', 'R/',
                 'S/', 'T/', 'U/', 'V/',
                 'W/', 'X/',
                 'Y/', 'Z/']

# folders_valid = ['Below_CAM/A/', 'Below_CAM/B/', 'Below_CAM/C/', 'Below_CAM/D/',
#                  'Below_CAM/E/', 'Below_CAM/F/',
#                  'Below_CAM/G/', 'Below_CAM/H/', 'Below_CAM/I/', 'Below_CAM/J/',
#                  'Below_CAM/K/', 'Below_CAM/L/',
#                  'Below_CAM/M/', 'Below_CAM/N/', 'Below_CAM/O/', 'Below_CAM/P/',
#                  'Below_CAM/Q/', 'Below_CAM/R/',
#                  'Below_CAM/S/', 'Below_CAM/T/', 'Below_CAM/U/', 'Below_CAM/V/',
#                  'Below_CAM/W/', 'Below_CAM/X/',
#                  'Below_CAM/Y/', 'Below_CAM/Z/']

files = []

for folder in folders_train:
    for file in os.listdir(folder):
        print(folder)
        label = folder.split('/')[0]
        files.append([folder + file, label])
print(files)

df = pd.DataFrame(files, columns=['files', 'target']).to_csv('labels.csv')
# print(df.head())
