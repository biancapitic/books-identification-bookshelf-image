import numpy as np
import cv2
import math

class SpineSeparation:
    def __init__(self, data_batch):
        self.data_batch = data_batch

    def __imgToGrayScale(self, image):
        return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

    def __markBookSpinesInImage(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
        edges = cv2.Canny(img_gray, 50, 200)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 20  # minimum number of votes (intersections in Hough grid cell)
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLines(edges, rho, theta, threshold, np.array([]))
        lines_edges = []
        lines_coord = []

        coord = []
        if lines is not None:
            for ind in range(len(lines)):
                for rho, theta in lines[ind]:
                    if theta < (math.pi / 2):
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        if (abs(x1 - x2) < 500 or abs(y1 - y2) < 500) and x1 == x2:
                            if len(lines_coord) > 0:
                                size = len(lines_coord) - 1
                                d1 = math.sqrt(((lines_coord[size - 1][0] - x1) ** 2) + ((lines_coord[size - 1][1] - y1) ** 2))
                                d2 = math.sqrt(((lines_coord[size - 1][2] - x2) ** 2) + ((lines_coord[size - 1][3] - y2) ** 2))
                                if d1 > 55 or d2 > 55:
                                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    lines_coord.append((x1, y1, x2, y2))
                                    coord.append(x1)
                                    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
                            else:
                                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                lines_coord.append((x1, y1, x2, y2))
                                coord.append(x1)
                                lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        coord.sort()

        return lines_edges, lines_coord, coord

    def __spinesImages(self, img):
        img, lines_coord, coord = self.__markBookSpinesInImage(img)

        spinesImages = []
        ind = 0

        while ind < len(coord) - 1:
            if coord[ind+1] - coord[ind] > 15:
                spine = img[:, coord[ind]:coord[ind+1] + 1]
                spine_v1 = cv2.rotate(spine, cv2.cv2.ROTATE_90_CLOCKWISE)
                spine_v2 = cv2.rotate(spine, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                spinesImages.append(spine_v1)
                spinesImages.append(spine_v2)
            ind += 1

        return spinesImages

    def saveSpines(self, spines, name_ind):
        spines_files_list = []
        for ind in range(len(spines)):
            img = spines[ind]
            filepath = "images/spines/spine" + str(name_ind) + "_" + str(ind) + ".jpg"
            spines_files_list.append(filepath)
            if img is not None:
                cv2.imwrite(filepath, img)
        return spines_files_list

    def spineSeparation(self):
        spines_file_list = []
        for ind in range(len(self.data_batch)):
            img = self.data_batch[ind]
            spines = self.saveSpines(self.__spinesImages(img), ind)
            spines_file_list += spines

        print("----------> Spines SAVED")
        return spines_file_list
