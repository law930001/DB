import os
import cv2
import numpy as np
from tqdm import tqdm


def draw_result_image():

    image_dir = '/root/Storage/datasets/ICDAR2017/test/test_images/'

    result_dir = '/root/Storage/DB_v100/results/'

    result_img_dir = '/root/Storage/DB_v100/results_img/'


    if not os.path.exists(result_img_dir):
        os.makedirs(result_img_dir)

    for file in tqdm(os.listdir(image_dir)):
        img_num = file.replace('img_','').replace('.jpg','')
        
        img = cv2.imread(image_dir + 'img_' + str(img_num) + '.jpg')



        with open(result_dir + 'res_img_' + str(img_num) + '.txt', 'r') as res_file:
            for line in res_file.readlines():
                line = line.strip().split(',')
                del line[-1]
                line = list(map(int, line))
                x = line[::2]
                y = line[1::2]
                polygon = np.array(list(zip(x,y)))

                # print(polygon)

                cv2.drawContours(img, [polygon], 0, (0,0,255), 2)


        cv2.imwrite(result_img_dir + 'res_img_' + str(img_num) + '.jpg', img)

def draw_result_image_ans():
    image_dir = '/root/Storage/datasets/ICDAR2015/test/test_images/'

    ans_dir = '/root/Storage/datasets/ICDAR2015/test/test_gts/'

    result_img_dir = '/root/Storage/DB_v100/results_img/'


    if not os.path.exists(result_img_dir):
        os.makedirs(result_img_dir)

    for file in tqdm(os.listdir(image_dir)):
        img_num = file.replace('img_','').replace('.jpg','')
        
        img = cv2.imread(image_dir + 'img_' + str(img_num) + '.jpg')

        # print(img_num)

        with open(ans_dir + 'img_' + str(img_num) + '.jpg.txt', 'r') as res_file:
            for line in res_file.readlines():
                line = line.strip().split(',')
                if line[-1] == '###':
                    continue
                del line[-1]
                x = list(map(int, line[:8:2]))
                y = list(map(int, line[1:9:2]))
                polygon = np.array(list(zip(x,y)))

                # print(polygon)

                cv2.drawContours(img, [polygon], 0, (0,255,0), 2)


        cv2.imwrite(result_img_dir + 'res_img_' + str(img_num) + '_ans.jpg', img)

def draw_result_and_ans_image():
    image_dir = '/root/Storage/datasets/total_text/test/test_images/'

    ans_dir = '/root/Storage/datasets/total_text/test/test_gts/'
    result_dir = '/root/Storage/DB_v100/results/'

    result_img_dir = '/root/Storage/DB_v100/results_img/'


    if not os.path.exists(result_img_dir):
        os.makedirs(result_img_dir)

    for file in tqdm(os.listdir(image_dir)):
        img_num = file.replace('img','').replace('.jpg','').replace('_', '')
        
        img = cv2.imread(image_dir + file)

        # print(img_num)
        with open(result_dir + 'res_img' + str(img_num) + '.txt', 'r') as res_file:
            for line in res_file.readlines():
                line = line.strip().split(',')
                del line[-1]
                line = list(map(int, line))
                x = line[::2]
                y = line[1::2]
                polygon = np.array(list(zip(x,y)))

                # print(polygon)

                cv2.drawContours(img, [polygon], 0, (0,0,255), 5)

        with open(ans_dir + 'img' + str(img_num) + '.jpg.txt', 'r') as res_file:
            for line_o in res_file.readlines():
                line = line_o.strip().split(',')
                if line[-1] == '###':
                    continue
                del line[-1]
                if len(line) % 2 == 1:
                    del line[-1]
                x = list(map(int, line[::2]))
                y = list(map(int, line[1::2]))
                polygon = np.array(list(zip(x,y)))

                # print(polygon)

                cv2.drawContours(img, [polygon], 0, (0,255,0), 3)


        cv2.imwrite(result_img_dir + 'res_img_' + str(img_num) + '_all.jpg', img)
        


if __name__ == '__main__':
    # draw_result_image()
    # draw_result_image_ans()
    draw_result_and_ans_image()
