from LMC import LMC

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d #导入scipy里interpolate模块中的interpld插值模块
from scipy.misc import derivative#求导函数
import math


if __name__ == "__main__":
    img = cv2.imread('../data/origin/01.bmp', 2)  # 读取图片，转为灰度图[0，255]
    # 创建一个全黑的灰度图，假设图片为180x240
    img_V = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    print("图片像素%d x %d " % (img.shape[0],img.shape[1]))

    # test=LMC(20,10,20,10)#04.bmp参数
    test=LMC(4,5,4,5)#01.bmp图参数

    img_V_col=test.col_scan(img,img_V)
    img_V_row=test.row_scan(img,img_V_col)
    img_d2u=test.oblique_d2u_scan_(img,img_V_row)
    img_u2d=test.oblique_u2d_scan_(img,img_d2u)
    img_V_end=test.connect_fun(img_u2d,1)
    test.show_img('end',img_V_end)