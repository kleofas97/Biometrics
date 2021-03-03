import os
import cv2
import pandas as pd
import scipy.io
import random
import numpy as np
from sklearn.svm import LinearSVC
from skimage import feature

for_randomness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filename_list = [f for f in os.listdir('caltech') if os.path.isfile(os.path.join('caltech', f))]
df = pd.read_csv("caltech/caltech_labels.csv")
mat = scipy.io.loadmat('caltech/ImageData.mat')
df = df[df['1'].isin(df['1'].value_counts()[df['1'].value_counts() >= 20].index)]
train_images, train_labels, test_images, test_labels = [], [], [], []
Pos, Neg = 0, 0
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
            res = cv2.resize(crop_img, (100, 70))
            # cv2.imshow("resized",res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if random.choice(for_randomness) == 1:
                test_images.append(res)
                test_labels.append(int(df['1'][df.index == img_number].to_numpy()))
            else:
                train_images.append(res)
                train_labels.append(int(df['1'][df.index == img_number].to_numpy()))
testing = list(zip(test_images, test_labels))
# ----------------------------------------------------------------
# EigenFaceRecognizer
# ----------------------------------------------------------------
FR_Eigen = cv2.face.EigenFaceRecognizer_create()
FR_Eigen.train(train_images, np.asarray(train_labels, dtype=np.int64))
# predicting
for image, label in testing:
    idx, confidence = FR_Eigen.predict(image)
    if idx == label:
        Pos += 1
    else:
        Neg += 1
print("For EigenFaceRecognizer, test set. Positive: {}, Negative: {}".format(Pos, Neg))
# ----------------------------------------------------------------
# FisherFaceRecognizer
# ----------------------------------------------------------------
Pos, Neg = 0, 0
FR_Fisher = cv2.face.FisherFaceRecognizer_create()
FR_Fisher.train(train_images, np.asarray(train_labels, dtype=np.int64))
# predicting
for image, label in testing:
    idx, confidence = FR_Fisher.predict(image)
    if idx == label:
        Pos += 1
    else:
        Neg += 1
print("For FisherFaceRecognizer, test set. Positive: {}, Negative: {}".format(Pos, Neg))
# ----------------------------------------------------------------
# LBPHFaceRecognizer
# ----------------------------------------------------------------
Pos, Neg = 0, 0
FR_LBPH = cv2.face.LBPHFaceRecognizer_create()
FR_LBPH.train(train_images, np.asarray(train_labels, dtype=np.int64))
# predicting
for image, label in testing:
    idx, confidence = FR_LBPH.predict(image)
    if idx == label:
        Pos += 1
    else:
        Neg += 1
print("For LBPHFaceRecognizer, test set. Positive: {}, Negative: {}".format(Pos, Neg))
# ----------------------------------------------------------------
# HOG + SVM
# ----------------------------------------------------------------
train_hog = []
for img in train_images:
    hog_desc = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    # update the data and labels
    train_hog.append(hog_desc)
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(train_hog, train_labels)
#predicting
Pos, Neg = 0, 0
for image,label in testing:
    (hog_desc, hog_image) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), transform_sqrt=True,
                                        block_norm='L2-Hys', visualize=True)
    pred = svm_model.predict(hog_desc.reshape(1, -1))[0]
    idx = pred
    if idx == label:
        Pos += 1
    else:
        Neg += 1
print("For HOG+SVM test set. Positive: {}, Negative: {}".format(Pos, Neg))