import os
import sys
import time
import subprocess
import numpy as np
from colorama import Fore, Style
sys.path.append("../my_code/auto_grader/")
from auto_grader import auto_grader
import identify_num
import image_process_cv

if __name__ == '__main__':
    enable_ui = True

    ag = auto_grader(enable_ui)

    images = image_process_cv.get_images()

    gray_images = np.reshape(np.array([image_process_cv.convert_to_gray(x) for x in images]), (-1, 784))

    colors = np.array([image_process_cv.get_color(x) for x in images])

    numbers = identify_num.identify_numbers(gray_images)



    if enable_ui:
        color_dict = ['R', 'G', 'B']
    else:
        color_dict = [Fore.RED, Fore.GREEN, Fore.BLUE]
    ans_list = [numbers[i] + 10 * (colors[i] + 1) for i in range(64)]

    for i in range(8):
        for j in range(8):
            if enable_ui:
                print(str(numbers[i * 8 + j]) + color_dict[colors[i * 8 + j]], end=', ')
            else:
                print(color_dict[colors[i * 8 + j]], numbers[i * 8 + j], end=' ')
        print()
    if not enable_ui:
        print(Style.RESET_ALL)


    with open('./algorithms/map.txt', 'w') as f:
        for x in ans_list:
            print(x, end=' ', file=f)
        print(file=f)

    str = os.getcwd() + "/algorithms/solution.exe <"+os.getcwd()+"/algorithms/map.txt> " + os.getcwd() + "/algorithms/solution.txt "
    print(str)
    process = os.popen(str)

    print("Pending................")

    time.sleep(80)

    with open('./algorithms/solution.txt', 'r') as f:
            solutions = [(int(x) // 8, int(x) % 8) for x in f.readline().strip().split(' ')]


    for i in range(len(solutions) // 2):
        r1, c1 = solutions[i * 2]
        r2, c2 = solutions[i * 2 + 1]
        ag.link(r1, c1, r2, c2)


    os.system('pause')