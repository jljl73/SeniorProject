from facenet_pytorch import MTCNN, InceptionResnetV1
from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions
import cv2
from PIL import Image
import os
import json
import numpy as np
from scipy import ndimage
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import scipy
from matplotlib import pyplot as plt
# from tqdm.notebook import tqdm

def radimal_mean(path_img):
    src = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

    dft = np.fft.fft2(src)  # Discrete Fourier transforms
    # print(dft)
    # 출처 :
    f = dft
    sx, sy = f.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(X - sx / 2, Y - sy / 2)
    rbin = (20 * r / r.max()).astype(np.int)
    radial_mean = ndimage.mean(f, labels=rbin, index=np.arange(1, rbin.max() + 1))  # azimuthal average

    return radial_mean

def normalize(path_img):
    src = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    # print(src)
    src = cv2.normalize(src, src, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # print(src.shape)
    src = np.reshape(src, (1, -1))
    # print(src.shape)
    # print(src, src.shape)
    return src

class Face_Detection:

    def __init__(self, path):
        self.path = path

    def save_face_image(self, name): # MTCNN
        self.name = name

        cap = cv2.VideoCapture(self.path + self.name)
        mtcnn = MTCNN(select_largest=False, margin=20,device='cuda')

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        face = mtcnn(frame, save_path='img/face_image_' + self.name + '.jpg')

        return face

class svm:

    def __init__(self):
        self.data = []
        self.clf = SVC(kernel='linear')
        self.count = 0
        self.y = []

    def load_data(self, path_img, label):

        # normalize
        radial_mean = normalize(path_img)
        if(len(self.data) != 0):
            self.data = np.vstack((self.data, radial_mean))
            self.y = np.append(self.y, 0 if label == "FAKE" else 1)
        else:
            self.data = radial_mean
            self.y = [0 if label == "FAKE" else 1]

    def split_data(self):

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data, self.y, test_size=0.2, random_state=25)

    def train(self):
        self.clf.fit(self.train_x, self.train_y)


    def predict(self):
        # new_data_pp = np.reshape(pp(new_data), (1, -1))
        # print(new_data, self.clf.predict(new_data_pp))
        pred = self.clf.predict(self.test_x)
        print(pred)
        print('Prediction Accuracy: %.2f' % accuracy_score(self.test_y, pred))
        zeros = np.zeros(pred.shape)
        print('All zeros Accuracy: %.2f' % accuracy_score(self.test_y, zeros))



    def print_data(self):
        # print(self.data)
        # print(self.data.shape)
        self.data = np.transpose(self.data)
        # print(self.data.shape)

class read_json:
    def read(self, name):
        self.name = name

        with open("metadata.json") as file:
            json_data = json.load(file)
            label = json_data[self.name]["label"]
            original = json_data[self.name]["original"]
            return label, original


if __name__ == '__main__':

    path_dir = "Train/"
    File_List = os.listdir(path_dir)
    rj = read_json()
    # label, original = read_json(path_dir).read(File_List[0])
    y = []
    fd = Face_Detection(path_dir)
    s = svm()
    # for i in range(len(File_List)):
    #     face = fd.save_face_image(File_List[i])

    ec = 0
    for i in range(len(File_List)):

        try:
            label, original = rj.read(File_List[i])
            s.load_data('img/face_image_' + File_List[i] + '.jpg', label)
        except:
            print(File_List[i], " is error {0}".format(ec))
            ec += 1
    #
    #
    s.split_data()
    s.train()
    s.predict()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device