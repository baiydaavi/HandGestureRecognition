import os

import pandas as pd

os.chdir(r'HGM_data')

folders = ['Below_CAM/A/', 'Below_CAM/B/', 'Below_CAM/C/', 'Below_CAM/D/', 'Below_CAM/E/', 'Below_CAM/F/',
           'Below_CAM/G/', 'Below_CAM/H/', 'Below_CAM/I/', 'Below_CAM/J/', 'Below_CAM/K/', 'Below_CAM/L/',
           'Below_CAM/M/', 'Below_CAM/N/', 'Below_CAM/O/', 'Below_CAM/P/', 'Below_CAM/Q/', 'Below_CAM/R/',
           'Below_CAM/S/', 'Below_CAM/T/', 'Below_CAM/U/', 'Below_CAM/V/', 'Below_CAM/W/', 'Below_CAM/X/',
           'Below_CAM/Y/', 'Below_CAM/Z/',
           'Front_CAM/A/', 'Front_CAM/B/', 'Front_CAM/C/', 'Front_CAM/D/', 'Front_CAM/E/', 'Front_CAM/F/',
           'Front_CAM/G/', 'Front_CAM/H/', 'Front_CAM/I/', 'Front_CAM/J/', 'Front_CAM/K/', 'Front_CAM/L/',
           'Front_CAM/M/', 'Front_CAM/N/', 'Front_CAM/O/', 'Front_CAM/P/', 'Front_CAM/Q/', 'Front_CAM/R/',
           'Front_CAM/S/', 'Front_CAM/T/', 'Front_CAM/U/', 'Front_CAM/V/', 'Front_CAM/W/', 'Front_CAM/X/',
           'Front_CAM/Y/', 'Front_CAM/Z/',
           'Left_CAM/A/', 'Left_CAM/B/', 'Left_CAM/C/', 'Left_CAM/D/', 'Left_CAM/E/', 'Left_CAM/F/', 'Left_CAM/G/',
           'Left_CAM/H/', 'Left_CAM/I/', 'Left_CAM/J/', 'Left_CAM/K/', 'Left_CAM/L/', 'Left_CAM/M/', 'Left_CAM/N/',
           'Left_CAM/O/', 'Left_CAM/P/', 'Left_CAM/Q/', 'Left_CAM/R/', 'Left_CAM/S/', 'Left_CAM/T/', 'Left_CAM/U/',
           'Left_CAM/V', 'Left_CAM/W/', 'Left_CAM/X', 'Left_CAM/Y/', 'Left_CAM/Z/',
           'Right_CAM/A/', 'Right_CAM/B/', 'Right_CAM/C/', 'Right_CAM/D/', 'Right_CAM/E/', 'Right_CAM/F/',
           'Right_CAM/G/', 'Right_CAM/H/', 'Right_CAM/I/', 'Right_CAM/J/', 'Right_CAM/K/', 'Right_CAM/L/',
           'Right_CAM/M/', 'Right_CAM/N/', 'Right_CAM/O/', 'Right_CAM/P/', 'Right_CAM/Q/', 'Right_CAM/R/',
           'Right_CAM/S/', 'Right_CAM/T/', 'Right_CAM/U/', 'Right_CAM/V/', 'Right_CAM/W', 'Right_CAM/X/',
           'Right_CAM/Y/', 'Right_CAM/Z/']

files = []

for folder in folders:
    for file in os.listdir(folder):
        label = folder.split('/')[1]
        files.append([folder + file, label])
print(files)

df = pd.DataFrame(files, columns=['files', 'target']).to_csv('labels.csv')
# print(df.head())
