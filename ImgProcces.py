from PIL import Image
import numpy as np

from settings import *


class ImgProcces:

    def __init__(self):
        self.rects = np.empty((1, 1))
        self.rect_width = 0
        self.rect_height = 0
        self.width = 0
        self.height = 0

        self.image_name = ''


    def img_to_array(self, img_path='images/mono.jpg'):
        try:
            image = Image.open(img_path)
        except OSError as err:
            print(err)
            return

        self.image_name = img_path.split('/')[-1]
        print(self.image_name)

        img_arr = np.array(image)

        self.width = len(img_arr[0])
        self.height = len(img_arr)

        self.rects = self.__split(img_arr)
        self.rects = [[(2 * x / MAX_COLOR_VAL) - 1 for x in rect] for rect in self.rects]
        self.rects = np.array(self.rects)

        return self.rects

    
    def array_to_img(self, img_arr, img_name='result/out.jpg'):
        img_arr = (img_arr.astype(float) + 1) * MAX_COLOR_VAL / 2
        img_arr = np.clip(img_arr, 0, MAX_COLOR_VAL)
        img_arr = img_arr.tolist()

        img = self.__desplit(img_arr)
        img = Image.fromarray(np.asarray(img).astype('uint8'))
        img.save(img_name)

        return img

    
    def array_to_compressed_img(self, img_arr, img_name='result/compressed.jpg'):
        image = self.__desplit(img_arr.tolist())
        image = Image.fromarray(np.asarray(image).astype('uint8'))
        image.save(img_name)
        return image


    def __rectangolise(self):
        self.rect_width = 4
        self.rect_height = 4

    
    def __split(self, array):
        self.__rectangolise()
        array = array.tolist()

        self.__rectangolise()
        rect_count = int((self.height / self.rect_height) * (self.width / self.rect_width))
        result = [[] for _ in range(rect_count)]
        for row_index in range(self.height):
            for col_index in range(self.width):
                for element in array[row_index][col_index]:
                    result_index = ((row_index // self.rect_height) * int(self.width / self.rect_width)) + (col_index // self.rect_width)
                    result[int(result_index)].append(element)
       
        return result


    def __desplit(self, array):
        self.__rectangolise()
        result = [[] for _ in range(self.height)]
        for row_index in range(len(array)):
            for col_index in range(0, len(array[0]), COLOR_CODING_POSITIONS):
                pixel = array[row_index][col_index:col_index + COLOR_CODING_POSITIONS]
                result_index = (col_index // (COLOR_CODING_POSITIONS * self.rect_width)) + (row_index // (self.width / self.rect_width) * self.rect_height)
                result[int(result_index)].append(pixel)

        return result
