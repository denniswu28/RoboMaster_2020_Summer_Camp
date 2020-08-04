import os
import sys
import numpy as np
from colorama import Fore, Style
sys.path.append("../my_code/auto_grader/")
from auto_grader import auto_grader
import correction
import identify_num
import image_process_cv

if __name__ == '__main__':
    print(sys.path)
    ag = auto_grader()

    images = image_process_cv.get_images()
    # print('images:', images.shape)

    gray_images = np.reshape(np.array([image_process_cv.convert_to_gray(x) for x in images]), (-1, 784))
    # print('gray_images:', gray_images.shape)

    colors = np.array([image_process_cv.get_color(x) for x in images])
    # print('colors:', colors)

    raw_predict = identify_num.identify_numbers(gray_images)
    # print('raw_predict', raw_predict)

    numbers = correction.get_number_and_correct(raw_predict, colors)
    # print('numbers: ', numbers)

    # color_dict = ['R', 'G', 'B']
    color_dict = [Fore.RED, Fore.GREEN, Fore.BLUE]
    ans_list = [numbers[i] + 10 * (colors[i] + 1) for i in range(64)]

    # print("----------")
    # for each in raw_predict:
    #     print(each, end=' ')
    # for each in O0OO000OOO0O0O000:
    #     print(each[1], end=' ')
    # for i in range(8):
    #     for j in range(8):
    #         # print(str(numbers[i * 8 + j]) + color_dict[colors[i * 8 + j]], end=', ')
    #         print(color_dict[colors[i * 8 + j]], numbers[i * 8 + j], end=' ')
    #     print()
    # print(Style.RESET_ALL)
    # print('Below is for other programs')
    for x in ans_list:
        print(x, end=' ')
    print()

    os.system('pause')