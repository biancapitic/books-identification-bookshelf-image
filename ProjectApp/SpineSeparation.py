class SpineSeparation:
    def __init__(self, data_batch):
        self.data_batch = data_batch

    def __imgToGrayScale(self, image):
        return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

    def __spineSeparationImage(self, img):
        # make image gray
        img = self.__imgToGrayScale(img)

        # TODO: process image to get the lines that mark spine separation
        return img

        #TODO: complete method

    def spineSeparation(self):
        processedImgs = [self.__spineSeparationImage(img) for img in self.data_batch]

        # TODO: save the spine images after they are generated
