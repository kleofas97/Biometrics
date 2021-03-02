import os
import cv2
import pandas as pd
import scipy.io
import random
import numpy as np
for_randomness = [1,2,3,4]
filename_list = [f for f in os.listdir('caltech') if os.path.isfile(os.path.join('caltech', f))]
df = pd.read_csv("caltech/caltech_labels.csv")
mat = scipy.io.loadmat('caltech/ImageData.mat')
# get indexes only of images that appears more or equal than 20 times
df = df[df['1'].isin(df['1'].value_counts()[df['1'].value_counts() >= 20].index)]
train_images,train_label,test_images,test_label = [],[],[],[]
idx_list = df.index.tolist()
for file in filename_list:
    if file[0] == "i":
        img_number = int(file[-7:-4])
        if img_number in idx_list:
            img = cv2.imread(os.path.join("caltech/", file))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            y = int(mat['SubDir_Data'][0][img_number - 1])
            x = int(mat['SubDir_Data'][1][img_number - 1])
            h = int(mat['SubDir_Data'][4][img_number - 1])
            w = int(mat['SubDir_Data'][5][img_number - 1])
            crop_img = gray[w:x, y:h]
            res = cv2.resize(crop_img,(100,70))
            # cv2.imshow("resized",res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if random.choice(for_randomness) == 1:
                test_images.append(res)
                test_label.append(int(df['1'][df.index == img_number].to_numpy()))
            else:
                train_images.append(res)
                train_label.append(int(df['1'][df.index == img_number].to_numpy()))

FR = cv2.face.EigenFaceRecognizer_create()
FR = FR.train(train_images, np.asarray(train_label,dtype=np.int64))
