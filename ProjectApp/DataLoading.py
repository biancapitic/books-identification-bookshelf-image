import numpy as np
import tensorflow as tf
import glob
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, input_shape, batch_size, shuffle=True):
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.batch_size = batch_size
        # load the data_vers1 from the root directory
        self.class_names = []
        self.data = self.get_data(db_dir)
        self.indices = np.arange(len(self.data))

    def get_data(self, root_dir):
        """"
        Loads the paths to the images
        """
        paths = glob.glob(root_dir + "/*.jpg")
        self.data = paths
        print("Here")
        return self.data

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data_vers1
        """
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        batch_x = []
        try:
            for i in batch_indices:
                img = cv2.imread(self.data[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.square_image(img)
                img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                batch_x.append(img)
        except:
            print("Something went wrong!")

        return np.asarray(batch_x)

    def square_image(self, image):
        pad_width = 0
        pad_height = 0

        if image.shape[0] < image.shape[1]:
            pad_width = (image.shape[1] - image.shape[0]) // 2
        else:
            pad_height = (image.shape[0] - image.shape[1]) // 2

        return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0,0)), mode='edge')

    def on_epoch_end(self):
        """"
        Called at the end of each epoch (when we saw all the images from the dataset)
        """
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
