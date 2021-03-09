# from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import cv2
import pandas as pd
import scipy.io
import random
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
from sklearn.svm import LinearSVC
from skimage import feature

for_randomness = [1, 2, 3, 4]
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
            y = int(mat['SubDir_Data'][0][img_number - 1])
            x = int(mat['SubDir_Data'][1][img_number - 1])
            h = int(mat['SubDir_Data'][4][img_number - 1])
            w = int(mat['SubDir_Data'][5][img_number - 1])
            crop_img = img[w:x, y:h, :]
            res = cv2.resize(crop_img, (160, 160))
            # cv2.imshow("resized", res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            res = res / 255.0
            if random.choice(for_randomness) == 1:
                roi_tensor = torch.from_numpy(res).permute(2, 0, 1).float()
                test_images.append(roi_tensor)
                test_labels.append(int(df['1'][df.index == img_number].to_numpy()))
            else:
                roi_tensor = torch.from_numpy(res).permute(2, 0, 1).float()
                train_images.append(roi_tensor)
                train_labels.append(int(df['1'][df.index == img_number].to_numpy()))
testing = list(zip(test_images, test_labels))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# roi_tensor = torch.from_numpy(np.asarray(train_labels, dtype=np.int64)).permute(2, 0, 1).float()
aligned = []
for img in train_images:
    x_aligned, prob = mtcnn(img, return_prob=True) #tu sie wydupca
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)

aligned = torch.stack(aligned).to(device)
# embeddings = resnet(aligned).detach().cpu()
train_vec = resnet(aligned).detach().cpu().numpy()
pass
#
#
# # If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
#
# # Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
# from PIL import Image
#
# img = Image.open(<image path>)
#
# # Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path=<optional save path>)
#
# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))
#
# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
