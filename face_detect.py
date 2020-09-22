from facenet_pytorch import MTCNN, InceptionResnetV1
from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions
from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU, Concatenate
from keras.optimizers import Adam
import cv2
from PIL import Image
import os
import json
import numpy as np
from scipy import ndimage
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def radimal_mean(src):

    dft = np.fft.fft(src)  # Discrete Fourier transforms
    f = src
    sx, sy = f.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(X - sx / 2, Y - sy / 2)
    rbin = (20 * r / r.max()).astype(np.int)
    radial_mean = ndimage.mean(f, labels=rbin, index=np.arange(1, rbin.max() + 1))  # azimuthal average

    return radial_mean

def padding(img, size):# size = 만들고 싶은 크기
    padding_size = (size - len(img[0])) // 2
    pad = ((padding_size, padding_size), (padding_size, padding_size))
    pad_img = np.pad(img, pad, 'constant', constant_values=0)
    return pad_img

def normalize(src):
    src = cv2.normalize(src, src, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
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

        img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

        # normalize
        mean = radimal_mean(img)
        mean = normalize(img)

        print(img.shape)
        a = padding(img, 256)
        # cv2.imshow('img', a)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print(a.shape)


        mean = mean.reshape((1, -1))
        if(len(self.data) != 0):
            self.data = np.vstack((self.data, mean))
            self.y = np.append(self.y, 0 if label == "FAKE" else 1)
        else:
            self.data = mean
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


class read_json:
    def read(self, name):
        self.name = name

        with open("metadata.json") as file:
            json_data = json.load(file)
            label = json_data[self.name]["label"]
            original = json_data[self.name]["original"]
            return label, original


class MesoInception4:
    def __init__(self):
        optimizer = Adam(learning_rate=0.001)
        self.data = []
        self.y = []
        self.size = 160
        self.model = self.init_model()
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def load_data(self, path_img, label):

        img = cv2.imread(path_img, cv2.IMREAD_COLOR)

        # pad_img = np.stack([np.pad(img[:,:,i], 48, 'constant', constant_values=0) for i in range(3)])
        # pad_img = np.transpose(pad_img, (1, 2, 0))
        pad_img = np.reshape(img, (1, self.size, self.size, 3))

        if(len(self.data) == 0):
            self.data = pad_img
        else:
            self.data = np.concatenate((self.data, pad_img), axis=0)

        self.y = np.append(self.y, 0 if label == "FAKE" else 1)

    def Inception(self, a, b, c, d, x):
        x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

        x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

        x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
        x3 = Conv2D(c, (3, 3), padding='same', activation='relu')(x3)

        x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
        x4 = Conv2D(d, (3, 3), padding='same', activation='relu')(x4)

        y = Concatenate(axis=-1)([x1, x2, x3, x4])
        return y

    def init_model(self):
        ipt = Input(shape=(self.size, self.size, 3))

        ## Meso4
        # x = Conv2D(8, (3, 3), padding = 'same', activation='relu')(ipt)
        # x = BatchNormalization()(x)
        # x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
        #
        # x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
        # x = BatchNormalization()(x)
        # x= MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        x = self.Inception(1, 4, 4, 2, ipt)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2), padding='same')(x)

        x = self.Inception(2, 4, 4, 2, x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        ############################ 이후엔 Meso4 랑 동일
        x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(16)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        opt = Dense(1, activation='sigmoid')(x)

        return Model(inputs=ipt, outputs=opt)

    def predict(self, x):
        return self.model.predict(x)
    def train(self):
        x = self.train_x
        y = self.train_y
        self.model.fit(x, y, epochs=5, batch_size=32)
    def test(self):
        x = self.test_x
        y = self.test_y
        res = self.model.evaluate(x, y)
        print(res)
        zeros = np.zeros(y.shape)
        print('All zeros Accuracy: %.2f' % accuracy_score(self.test_y, zeros))
    def load(self, path):
        self.model.load_weights(path)

    def split_data(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data, self.y, test_size=0.2,
                                                                                    random_state=252)
        # print(self.train_x.shape, self.test_x.shape, self.train_y.shape, self.test_y.shape)

if __name__ == '__main__':

    path_dir = "Train/"
    File_List = os.listdir(path_dir)
    rj = read_json()
    # label, original = read_json(path_dir).read(File_List[0])
    y = []
    fd = Face_Detection(path_dir)

    # 얼굴 이미지로 저장
    # for i in range(len(File_List)):
    #     face = fd.save_face_image(File_List[i])


    #######SVM
    ec = 0
    # for i in range(1):#len(File_List)):

        # try:
        # label, original = rj.read(File_List[i])
        # s.load_data('img/face_image_' + File_List[i] + '.jpg', label)
        # except:
        #     print(File_List[i], " is error {0}".format(ec))
        #     ec += 1
    # s.split_data()
    # s.train()
    # s.predict()
    ##########

    ########## Mesonet

    meso = MesoInception4()
    for i in range(len(File_List)):
        try:
            label, original = rj.read(File_List[i])
            meso.load_data('img/face_image_' + File_List[i] + '.jpg', label)
        except:
            # print(File_List[i], " is error {0}".format(ec))
            ec += 1

    meso.split_data()
    meso.train()
    meso.test()


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device
