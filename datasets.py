import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.utils import shuffle
import cv2
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.dataset = ReplayAttackDataset(batch_size)

        file_path, label_truth, label_human_id = self.dataset.load_idx()
        # encode the label
        self.encoding_truth = preprocessing.LabelEncoder()
        self.encoding_truth.fit(label_truth)
        self.list_label_truth = self.encoding_truth.transform(label_truth)

        self.encoding_humanid = preprocessing.LabelEncoder()
        self.encoding_humanid.fit(label_human_id)
        self.list_label_human_id = self.encoding_humanid.transform(label_human_id)

        self.list_file_path_truth = file_path.copy()
        self.list_file_path_human_id = file_path.copy()

        self.shuffle_dataset()
        self.len_dataset = len(self.list_file_path_truth)

        print('loading face detector net')
        protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        modelPath = os.path.sep.join(["face_detector",
                                      "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        with open("config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)
        self.standard_img_size = cfg['net']['input_img_size']
        self.numb_faceid = len(self.encoding_humanid.classes_)

    def shuffle_dataset(self):
        self.list_file_path_truth, self.list_label_truth = shuffle(self.list_file_path_truth, self.list_label_truth,
                                                                   random_state=10)
        self.list_file_path_human_id, self.list_label_human_id = shuffle(self.list_file_path_human_id,
                                                                         self.list_label_human_id,
                                                                         random_state=100)

    def generate_minibatch(self):
        start_idx = 0
        humanid_idx = 0
        while True:
            # crop sub lists from the long lists
            if start_idx + self.batch_size <= self.len_dataset:
                batch_file_path_truth = self.list_file_path_truth[start_idx: start_idx + self.batch_size]
                batch_label_truth = self.list_label_truth[start_idx: start_idx + self.batch_size]

                batch_file_path_random = self.list_file_path_human_id[start_idx: start_idx + self.batch_size]
                batch_label_random = self.list_label_human_id[start_idx: start_idx + self.batch_size]
            if start_idx < self.len_dataset < (start_idx + self.batch_size):
                batch_file_path_truth = self.list_file_path_truth[start_idx: self.len_dataset]
                batch_label_truth = self.list_label_truth[start_idx: self.len_dataset]

                batch_file_path_random = self.list_file_path_human_id[start_idx: self.len_dataset]
                batch_label_random = self.list_label_human_id[start_idx: self.len_dataset]
            elif start_idx >= self.len_dataset:
                break

            # load image to numpy array
            # load image for anti spoofing task
            batch_img_4_truth = None
            batch_img_4_human_id = None
            for file_path_truth in batch_file_path_truth:
                img = cv2.imread(file_path_truth)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.standard_img_size, self.standard_img_size))
                img = np.expand_dims(img, axis=0)
                if batch_img_4_truth is None:
                    batch_img_4_truth = img
                else:
                    batch_img_4_truth = np.concatenate((batch_img_4_truth, img), axis=0)
            # load image for recognition task
            batch_img_4_human_id = []
            batch_label_human_id = []
            while len(batch_img_4_human_id) < len(
                    batch_file_path_truth):  # collect face-recog imgs with same quantity of anti-spoofing imgs
                img = cv2.imread(self.list_file_path_human_id[humanid_idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                self.detector_net.setInput(blob)
                detected = self.detector_net.forward()
                # it assumes that the img has only 1 face
                try:
                    detection = self.get_face_bbox(detected, img.shape[1], img.shape[0])
                    if detection is None:
                        raise TypeError("Detect face fail at ", self.list_file_path_human_id[humanid_idx])
                    minX, minY, maxX, maxY = detection
                    crop = img[minY:maxY, minX:maxX, :]
                    crop = cv2.resize(crop, (self.standard_img_size, self.standard_img_size))
                    batch_img_4_human_id.append(crop)
                    batch_label_human_id.append(self.list_label_human_id[humanid_idx])

                except:
                    print('face detection false')

                humanid_idx += 1
                if humanid_idx >= self.len_dataset:
                    humanid_idx = 0
            batch_img_4_human_id = np.asarray(batch_img_4_human_id)
            batch_label_human_id = np.asarray(batch_label_human_id)

            # select random images for lpc loss
            batch_random_1 = []
            batch_random_2 = []
            list_random_images_path1 = random.sample(self.list_file_path_human_id, k=len(batch_img_4_truth))
            list_random_images_path2 = random.sample(self.list_file_path_human_id, k=len(batch_img_4_truth))
            for file_path_1 in list_random_images_path1:
                img = cv2.imread(file_path_1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.standard_img_size, self.standard_img_size))
                batch_random_1.append(img)

            for file_path_2 in list_random_images_path2:
                img2 = cv2.imread(file_path_2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                img2 = cv2.resize(img2, (self.standard_img_size, self.standard_img_size))
                batch_random_2.append(img2)

            batch_img_random_1 = np.array(batch_random_1)
            batch_img_random_2 = np.array(batch_random_2)
            batch_label_random = [0 for i in range(len(batch_img_random_1))]
            batch_label_random = np.asarray(batch_label_random).astype(float)

            start_idx += self.batch_size
            yield batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2, batch_label_random, batch_img_4_human_id, batch_label_human_id

    def generate_batch(self):
        img_4_truth = None
        label_truth = None

        img_random_1 = None
        img_random_2 = None
        label_random = None

        img_4_human_id = None
        label_human_id = None
        for batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2, batch_label_random, batch_img_4_human_id, batch_label_human_id in self.generate_minibatch():
            if img_4_truth is not None:
                img_4_truth = np.concatenate((img_4_truth, batch_img_4_truth), axis=0)
            else:
                img_4_truth = batch_img_4_truth

            if label_truth is not None:
                label_truth = np.concatenate((label_truth, batch_label_truth), axis=0)
            else:
                label_truth = batch_label_truth

            if img_random_1 is not None:
                img_random_1 = np.concatenate((img_random_1, batch_img_random_1), axis=0)
            else:
                img_random_1 = batch_img_random_1

            if img_random_2 is not None:
                img_random_2 = np.concatenate((img_random_2, batch_img_random_2), axis=0)
            else:
                img_random_2 = batch_img_random_2

            if label_random is not None:
                label_random = np.concatenate((label_random, batch_label_random), axis=0)
            else:
                label_random = batch_label_random

            if img_4_human_id is not None:
                img_4_human_id = np.concatenate((img_4_human_id, batch_img_4_human_id), axis=0)
            else:
                img_4_human_id = batch_img_4_human_id

            if label_human_id is not None:
                label_human_id = np.concatenate((label_human_id, batch_label_human_id), axis=0)
            else:
                label_human_id = batch_label_human_id

        return img_4_truth/255.0, label_truth.astype(np.int), img_random_1/255.0, img_random_2/255.0, label_random.astype(np.int), img_4_human_id, label_human_id.astype(np.int)

    def get_face_bbox(self, detected, img_w, img_h):
        ad = 0.5
        # loop over the detections
        if detected.shape[2] > 0:
            confidence = detected[0, 0, :, 2]
            max_cfd_idx = np.argmax(confidence, axis=0)
            if (confidence[max_cfd_idx] <= 0.5):
                return None
            # print('max idx {} max conf {}'.format(max_cfd_idx, confidence[max_cfd_idx]))

            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            (h0, w0) = img_h, img_w
            box = detected[0, 0, max_cfd_idx, 3:7] * np.array([w0, h0, w0, h0])
            (startX, startY, endX, endY) = box.astype("int")
            # print((startX, startY, endX, endY))
            x1 = startX
            y1 = startY
            w = endX - startX
            h = endY - startY

            x2 = x1 + w
            y2 = y1 + h

            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)
            return xw1, yw1, xw2, yw2
        print('Error no face detected')
        return None


class ReplayAttackDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_idx(self):
        list_file_path = []
        list_label_truth = []
        list_label_human_id = []
        attack_dir = './dataset/replayattack_prep/attack'
        real_dir = './dataset/replayattack_prep/real'
        print('loading index for attack')
        for client in os.listdir(attack_dir):
            client_dir = os.path.join(attack_dir, client)
            for pic in os.listdir(client_dir):
                list_file_path.append(os.path.join(client_dir, pic))
                list_label_truth.append('attack')
                list_label_human_id.append(client)
        print('loading index for real')
        for client in os.listdir(real_dir):
            client_dir = os.path.join(real_dir, client)
            for pic in os.listdir(client_dir):
                list_file_path.append(os.path.join(client_dir, pic))
                list_label_truth.append('real')
                list_label_human_id.append(client)
        return list_file_path, list_label_truth, list_label_human_id


if __name__ == '__main__':
    replay_dataset = Dataset('replay', 7)
    img_4_truth, label_truth, img_random_1, img_random_2, label_random, img_4_human_id, label_human_id = replay_dataset.generate_batch()

    for i, img in enumerate(img_4_human_id):
        name = replay_dataset.encoding_humanid.inverse_transform([label_human_id[i]])[0]
        dir = os.path.join('debug_fr', name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, '{}_{}.jpg'.format(name, str(i)))
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = 10, 10  # for raster image
    nrows, ncols = 10, 10  # array of sub-plots
    figsize = [6, 8]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = img_4_human_id[i]
        axi.imshow(img)
        axi.set_title(replay_dataset.encoding_humanid.inverse_transform([label_human_id[i]]))

    plt.tight_layout(True)
    plt.show()



