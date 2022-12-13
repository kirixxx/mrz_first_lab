
from PIL import Image, ImageDraw
from math import sqrt
import numpy as np
import random
import pickle
from tqdm import tqdm

class Work :
    def __init__(self, number):
        self.block_width = 4
        self.block_height = 4
        self.neuron = 2
        self.matrix_update = []
        self.matrix_x = []
        self.matrix_y = []
        self.matrix_dx = []
        self.blocks=[]
        self.select_action(number)
        
    
    def select_action(self, input):
        if input == '1':
            self.action_1()
        elif input == '2':
            self.action_2()
        elif input == '3':
            self.action_3()
        elif input == '4':
            self.action_4()
        else:
            print('error')
    
    def action_1(self):
        image_to_matrix = self.trans_image_to_matrix()       
        self.matrix_update = self.update(image_to_matrix, self.block_height, self.block_width)
        self.blocks = self.line_block(self.matrix_update, self.block_height, self.block_width)
        weight_1 = self.create_weight_matrix(self.neuron, self.blocks)
        weight_2 = self.transfom(weight_1)
        iter = 1

        while True:
            sum_err = 0
            for i in tqdm(range(len(self.blocks))):
                self.matrix_y = self.multiple_matrix([self.blocks[i]], weight_1)
                self.matrix_x = self.multiple_matrix(self.matrix_y, weight_2)
                self.matrix_dx = self.div_matrix(self.matrix_x, [self.blocks[i]])
                w_matrix_pred = weight_2
                weight_2 = self.correct_weight_matrix(weight_1, weight_2, self.matrix_y, self.matrix_dx, layer=2)
                weight_1 = self.correct_weight_matrix(weight_1, w_matrix_pred, self.matrix_y, self.matrix_dx, X = self.matrix_x)
                sum_err += self.sqrt_error(self.matrix_dx)            
            print(f'\nError {iter} : ', sum_err)
            iter += 1
            
            if sum_err < 2000:
                X_end = []
                for i in tqdm(range(len(self.blocks))):
                    self.matrix_y = self.multiple_matrix([self.blocks[i]], weight_1)
                    self.matrix_x = self.multiple_matrix(self.matrix_y, weight_2)
                    X_end.append(self.matrix_x)

                for i in range(len(X_end)):
                    X_end[i] = X_end[i][0]

                for i in range(len(X_end)):
                    X_end[i] = [X_end[i][j:j + 3] for j in range(0, len(X_end[i]), 3)]

                result_matrix = image_to_matrix

                result_matrix = self.image_to_pixels(result_matrix, X_end, self.block_width, self.block_height)
                self.draw_image(result_matrix, len(result_matrix),len(result_matrix[0]))
                self.save_weight(weight_1, "1")
                self.save_weight(weight_2, "2")
                break
    
    def action_2(self):
        from_img_matrix = self.trans_image_to_matrix()
        self.matrix_update = self.update(from_img_matrix, self.block_height, self.block_width)
        self.blocks = self.line_block(self.matrix_update, self.block_height, self.block_width)
        weight_1 = self.read_weight("1")
        weight_2 = self.read_weight("2")
        matrix_end = []
        for i_id in tqdm(range(len(self.blocks))):
                    # функции сети
            self.matrix_y = self.multiple_matrix([self.blocks[i_id]], weight_1)
            X = self.multiple_matrix(self.matrix_y, weight_2)
            matrix_end.append(X)

        for i_id in range(len(matrix_end)):
            matrix_end[i_id] = matrix_end[i_id][0]

        for i_id in range(len(matrix_end)):
            matrix_end[i_id] = [matrix_end[i_id][j:j + 3] for j in range(0, len(matrix_end[i_id]), 3)]
            
        self.com_image(matrix_end ,len(matrix_end), len(matrix_end[0]))
    
    def action_3(self):
        image_to_matrix = self.trans_image_to_matrix()
        self.matrix_update = self.update(image_to_matrix, self.block_height, self.block_width)
        Y_old = self.read_Y()
        Y = Y_old[0]
        
        result_matrix = image_to_matrix
    
        result_matrix = self.image_to_pixels(result_matrix, Y, self.block_width, self.block_height)
        self.draw_image(result_matrix, len(result_matrix),len(result_matrix[0]))
        
    def action_4(self):
        from_img_matrix = self.trans_image_to_matrix()
        self.matrix_update = self.update(from_img_matrix, self.block_height, self.block_width)
        self.blocks = self.line_block(self.matrix_update, self.block_height, self.block_width)
        weight_1 = self.read_weight("1")
        weight_2 = self.read_weight("2")
        end_matrix = []
        for i in tqdm(range(len(self.blocks))):
            self.matrix_y = self.multiple_matrix([self.blocks[i]], weight_1)
            self.matrix_x = self.multiple_matrix(self.matrix_y, weight_2)
            end_matrix.append(self.matrix_x)
        
        for i in range(len(end_matrix)):
            end_matrix[i] = end_matrix[i][0]

        for i in range(len(end_matrix)):
            end_matrix[i] = [end_matrix[i][j:j + 3] for j in range(0, len(end_matrix[i]), 3)]

        result_matrix = from_img_matrix
    
        result_matrix = self.image_to_pixels(result_matrix, end_matrix, self.block_width, self.block_height)
        self.draw_image(result_matrix, len(result_matrix),len(result_matrix[0]))
    
    def read_Y(self):
        with open('com_img.bin', "rb") as file:
            matrix =  pickle.load( file)
            return matrix
        
    def com_image(self, compress_img, width, height):
        with open('com_img.bin', "wb") as file:
            pickle.dump([compress_img, width, height], file)
        
    def read_weight(self, number):
        if number == "1":
            weight = [] 
            with open('matrix_1.txt', 'r') as f:
                    for l in f:
                        weight.append([float(x) for x in l.split( )])
            return weight
        elif number == "2":
            weight = []
            with open('matrix_2.txt', 'r') as f:
                    for l in f:
                        weight.append([float(x) for x in l.split( )])
            return weight
    
    def add_matrix(self):
        result_matrix = []
        for x in range(image.size[0]):
            rgb_row = []
        for l in range(len(blocks[0])):
            line_s = []
            line_s.append(random.uniform(0.1, 0.1))
            second_z = list(second_z)   
    
    
    def trans_image_to_matrix(self):
        result_matrix = []
        image = Image.open('pirat.jpg')
        pixel_matrix = image.load() 
        
        for x in range(image.size[0]):
            rgb_row = []
            for y in range(image.size[1]):
                rgb_pixel = []
                for rgb_id in range(3):
                    rgb = (2 * pixel_matrix[x, y][rgb_id] / 255) - 1
                    rgb_pixel.append(rgb)
                rgb_row.append(rgb_pixel)

            result_matrix.append(rgb_row)
    
        return result_matrix
    
    def update(self, matrix, height, width):
        count = 1
        if len(matrix) % width != 0:
            while len(matrix) % width != 0:
                matrix.append(matrix[-count])
                count += 1

        if len(matrix[0]) % height != 0:
            while len(matrix[0]) % height != 0:
                if len(matrix[0]) % height == 0:
                    break
                for i in range(len(matrix)):
                    matrix[i].append(matrix[i][-1])
        return matrix
    
    def line_block(self, matrix,width,height):
        block = []
        for wh in range(0, len(matrix), width):
            for ht in range(0, len(matrix[0]), height):
                line = []
                for x_id in range(width):
                    for y_id in range(height):
                        line += matrix[wh + x_id][ht + y_id]
                block.append(line)
        return block
    
    def create_weight_matrix(self, first_neuron, blocks):
        matrix = []
        for l in range(len(blocks[0])):
            line = []
            for j in range(first_neuron):
                line.append(random.uniform(-0.1, 0.1))
            matrix.append(line)
        return matrix   
    
    def transfom(self, matrix):
        result_matrix = []
        for i in range(len(matrix[0])):
            line = []
            for j in range(len(matrix)):
                line.append(matrix[j][i])
            result_matrix.append(line)
        return result_matrix
    
    def multiple_matrix(self, first_matrix, second_matrix):
        second_zip = zip(*second_matrix) 
        second_zip = list(second_zip)  
        return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)) for col_b in second_zip] for row_a in first_matrix]
    
    def div_matrix(self, first_matrix, second_matrix):
        matrix = []
        for i_id in range(len(first_matrix)):
            line = []
            for j_id in range(len(first_matrix[0])):
                line.append(first_matrix[i_id][j_id] - second_matrix[i_id][j_id])
            matrix.append(line)
        return matrix
    
    def draw_image(self, matrix, width, height):
        image = Image.new('RGB', (width,height))
        draw_image = ImageDraw.Draw(image)
        for x_id in range(width):
            for y_id in range(height):
                z = (int(255 * (matrix[x_id][y_id][0] + 1) / 2), int(255 * (matrix[x_id][y_id][1] + 1) / 2),
                                    int(255 * (matrix[x_id][y_id][2] + 1) / 2))
                draw_image.point((x_id, y_id), z)
        image.save("pirat_rezult.jpg")

    def image_to_pixels(self, matrix, X, height, width):
        n=0
        for i_id in range(0, len(matrix),width):
            for j_id in range(0, len(matrix[0]), height):
                m = 0
                for x_id in range(width):
                    for y_id in range(height):
                        matrix[i_id + x_id][j_id + y_id] = []
                        matrix[i_id + x_id][j_id + y_id] += X[n][m]
                        m += 1
                n += 1
        return matrix
    
    def sum_matrix(self, first_matrix, second_matrix):
        matrix = []
        for i_id in range(len(first_matrix)):
            line = []
            for j_id in range(len(first_matrix[0])):
                line.append(first_matrix[i_id][j_id] + second_matrix[i_id][j_id])
            matrix.append(line)
        return matrix

    def devision_matrix(self, alpha, matrix):
        for i_id in range(len(matrix)):
            for j_id in range(len(matrix[0])):
                matrix[i_id][j_id]=alpha * matrix[i_id][j_id]
        return matrix

    def correct_weight_matrix(self, weight_matrix_1, weight_matrix_2, Y, dX, X=[], layer=1, alphaX = 0.005, alphaY=0.005):
        if layer == 1:
            pred_matrix = self.multiple_matrix(self.transfom(X), dX)
            pred_matrix = self.multiple_matrix(pred_matrix, self.transfom(weight_matrix_2))
            devision_matrix = self.devision_matrix(alphaX,pred_matrix)
            return self.div_matrix(weight_matrix_1, devision_matrix)
        else:
            pred_matrix = self.multiple_matrix(self.transfom(Y), dX)
            devision_matrix = self.devision_matrix(alphaY,pred_matrix)
            return self.div_matrix(weight_matrix_2, devision_matrix)

    def to_second_step(self, x):
        return  x ** 2

    def sqrt_error(self, dX):
        sqrt = list(map(lambda x: x ** 2, dX[0]))
        sqrt = sum(sqrt)
        return sqrt
    
    def save_weight(self, weight_matrix, number):
        if (number == "1"):
            with open('matrix_1.txt', 'w') as f:
                        for i in range(len(weight_matrix)):
                            for j in range(len(weight_matrix[0])):
                                f.write(f"{weight_matrix[i][j]} ")
                            f.write('\n')
        elif number == "2": 
             with open('matrix_2.txt', 'w') as f:
                        for i in range(len(weight_matrix)):
                            for j in range(len(weight_matrix[0])):
                                f.write(f"{weight_matrix[i][j]} ")
                            f.write('\n')   
