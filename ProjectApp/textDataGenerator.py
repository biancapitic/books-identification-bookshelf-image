import os
import numpy as np
import cv2
from keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextDataGenerator(Sequence):
    def __init__(self, db_path,input_shape, batch_size, characters, max_label_len, shuffle=True):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_label_len = max_label_len
        self.data, self.labels = self.get_data(db_path)
        self.characters = characters
        self.indices = np.arange(len(self.data))

        print(self.__encode_label("HELLO"))

        self.on_epoch_end()

    def get_data(self, root_dir):

        labels = []
        paths = []
        for path in os.listdir(root_dir):
            paths.append(root_dir +"/" + path)
            label = path.split("_")[1]
            labels.append(label)
            self.max_label_len = max(self.max_label_len, len(str(label)))

        return paths, labels

    def __getCharacters(self):
        vocab = set("".join(map(str, self.labels)))
        return sorted(vocab)

    def __encode_label(self, txt):
        encodedLabel = []
        for ind, char in enumerate(txt):
            try:
                encodedLabel.append(self.characters.index(char))
            except:
                print("Something is wrong!")
        return encodedLabel

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getImage(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape

        # we want the images to be (height, width) = (32, 128)
        if height < 32:
            paddingValues = np.ones((32-height, width)) * 255
            img = np.concatenate((img, paddingValues))
            height = 32

        if width < 128:
            paddingValues = np.ones((height, 128-width)) * 255
            img = np.concatenate((img, paddingValues), axis=1)
            width = 128

        if width > 128 or height > 32:
            img = cv2.resize(img, (128,32))

        img = np.expand_dims(img, axis=2)

        # Normalizing image
        img = img / 255.
        return img

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_x_item_len = 31
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        batch_y_length = []
        batch_x_length = []

        for ind in batch_indices:
                try:
                    img = self.__getImage(self.data[ind])
                    batch_x.append(img)
                    batch_x_length.append(batch_x_item_len)

                    label = str(self.labels[ind]).strip()
                    encodedLabel = self.__encode_label(label)
                    batch_y.append(encodedLabel)
                    batch_y_length.append(len(label))
                except:
                    print("Something went wrong with picture: " + self.data[ind])

        res_1 = np.array(batch_x)
        res_2 = pad_sequences(batch_y, maxlen=self.max_label_len, padding='post', value=len(self.characters))
        res_3 = np.array(batch_x_length)
        res_4 = np.array(batch_y_length)

        return [res_1, res_2, res_3, res_4], np.zeros(len(batch_y))

    def on_epoch_end(self):
        """"
        Called at the end of each epoch (when we saw all the images from the dataset)
        """
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)