import os
import cv2
import pandas as pd
import scipy.io
import random
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from sklearn.svm import LinearSVC
import itertools

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # y = int(mat['SubDir_Data'][0][img_number - 1])
            # x = int(mat['SubDir_Data'][1][img_number - 1])
            # h = int(mat['SubDir_Data'][4][img_number - 1])
            # w = int(mat['SubDir_Data'][5][img_number - 1])
            # crop_img = img[w:x, y:h, :]
            # res = cv2.resize(crop_img, (160, 160))
            # cv2.imshow("resized", res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # res = res / 255.0
            bbx, prob = mtcnn.detect(img)
            if (bbx is not None):
                roi_tensor = mtcnn.extract(img, bbx, None)
                if random.choice(for_randomness) == 1:
                    test_images.append(roi_tensor)
                    test_labels.append(int(df['1'][df.index == img_number].to_numpy()))
                else:
                    train_images.append(roi_tensor)
                    train_labels.append(int(df['1'][df.index == img_number].to_numpy()))
testing = list(zip(test_images, test_labels))

batch_size = 32
features_testing = []
features_training = []


def chunks(item_list, n):
    for i in range(0, len(item_list), n):
        yield item_list[i:i + n]


for batch in chunks(test_images, batch_size):
    aligned = torch.stack(batch).to(device)
    embeddings = resnet(aligned).detach().cpu().numpy()
    features_testing.append(embeddings)

features_testing = list(itertools.chain.from_iterable(features_testing))
test_ds = list(zip(features_testing, test_labels))

for batch in chunks(train_images, batch_size):
    aligned = torch.stack(batch).to(device)
    embeddings = resnet(aligned).detach().cpu().numpy()
    features_training.append(embeddings)

features_training = list(itertools.chain.from_iterable(features_training))

svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(features_training, train_labels)

# predict
Pos, Neg = 0, 0
for features, label in test_ds:
    pred = svm_model.predict(features.reshape(1, -1))[0]
    idx = pred
    print("Predicted: {}, Real: {}".format(idx, label))
    if idx == label:
        Pos += 1
    else:
        Neg += 1
print("HOG+SVM: Positive: {}, Negative: {}".format(Pos, Neg))

# result: HOG+SVM: Positive: 93, Negative: 8
