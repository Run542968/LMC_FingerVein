import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d #导入scipy里interpolate模块中的interpld插值模块
from scipy.misc import derivative#求导函数
import math

class LMC:
    def __init__(self,row_col_windowsize,row_col_cur_windowsize,oblique_widowsize,oblique_cur_widowsize):
        '''
        处理的图片大小不同，滤波时使用的windowSize(滤波窗口的大小)也要变化。滤波是曲线拟合后用到的,标准：图片大，滤波窗口大
        '''
        self.row_col_windowsize=row_col_windowsize#行和列扫描时的截面曲线滤波size
        self.row_col_cur_windowsize=row_col_cur_windowsize#行和列扫描时的曲率滤波size
        self.oblique_widowsize=oblique_widowsize#倾斜扫描时的截面曲线滤波size
        self.oblique_cur_widowsize=oblique_cur_widowsize#倾斜扫描时的曲率滤波size

    def filter_fun(self,interval,windowsize):
        '''
        滤波函数
        :param interval: 区间
        :param windowsize: 滤波窗口大小
        :return: 经过滤波后的list/ndarray
        '''
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(interval, window, 'same')
        return re

    def col_scan(self,img,img_V):
        '''
        按列扫描
        :param img: 原始图像
        :param img_V: 经过扫描后得到的img_V，每个点是Scr值
        :return: 处理后的img_V
        '''
        for k in range(img.shape[1]):
            x=np.linspace(0,img.shape[0]-1,img.shape[0])#0-239分成240份,即1，2，3.。。
            y=img[:,k]#图像中第一列像素点值

            #对y进行滤波处理
            y=self.filter_fun(y,self.row_col_windowsize)

            #三次样条插值,拟合出离散点曲线，用于求导数
            fun1=interp1d(x,y,kind='cubic')
            x_new=np.linspace(0,img.shape[0]-1,img.shape[0])#0-239，分为240个点
            y_new=fun1(x_new)

            # #画出拟合后的曲线图
            # plt.subplot(2,1,1)
            # plt.plot(x_new,y_new)

            #求出[1,238]各个点出的曲率
            x_z=[]#【1，238】
            k_z=[]#曲率值
            for i in range(1,img.shape[0]-1,1):
                x_z.append(i)
                k_z.append(derivative(fun1,i,dx=1e-6,n=2)/math.pow((1+math.pow(derivative(fun1,i,dx=1e-6),2)),1.5))

            #对曲率也滤波处理一下
            k_z = self.filter_fun(k_z, self.row_col_cur_windowsize)

            # #画出曲率图
            # plt.subplot(2,1,2)
            # plt.plot(x_z,k_z)

            #找出局部局部最大点做在凸起的宽度
            left_x=[]
            right_x=[]
            for i in range(len(k_z)-1):#i是索引，item是k_z中的项
                if(k_z[i]<0 and k_z[i+1]>=0):
                    left_x.append(i+0.5)
                elif(k_z[i]>=0 and k_z[i+1]<0):
                    right_x.append(i+0.5)

            #求出突起的距离
            if(left_x and right_x):
                if(len(left_x)==len(right_x)):
                    if(right_x[0]>left_x[0]):
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
                    else:
                        del(right_x[0])
                        left_x.pop()
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
                else:
                    if(len(right_x)>len(left_x)):
                        del(right_x[0])
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
                    else:
                        left_x.pop()
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
            else:
                distinct=[0]

            #找出局部最大值及其对应的x
            localMax=[]
            for j in range(min(len(left_x),len(right_x))):
                max=0
                index=0
                for i in range(int(left_x[j]+0.5),int(right_x[j]+0.5)):
                    if(k_z[i]>=max):
                        max=k_z[i]
                        index=i
                tup=(max,index,distinct[j])#（高度，高度对应的索引-z值，宽度）
                localMax.append(tup)

            #计算Scr(z')
            Scr=list(map(lambda x:(x[0]*x[2],x[1]),localMax))

            #把Src加进去
            for item in Scr:
                img_V[item[1],k]=img_V[item[1],k]+item[0]
            print('按照列进行描点，第%d列' % (k))
        return img_V

    def row_scan(self,img,img_V):
        '''
        按行扫描
        '''
        for k in range(img.shape[0]):
            x=np.linspace(0,img.shape[1]-1,img.shape[1])#0-239分成240份,即1，2，3.。。
            y=img[k,:]#图像中第一列像素点值

            #对y进行滤波处理
            y=self.filter_fun(y,self.row_col_windowsize)

            #三次样条插值,拟合出离散点曲线，用于求导数
            fun1=interp1d(x,y,kind='cubic')
            x_new=np.linspace(0,img.shape[1]-1,img.shape[1])#0-239，分为240个点
            y_new=fun1(x_new)

            # #画出拟合后的曲线图
            # plt.subplot(2,1,1)
            # plt.plot(x_new,y_new)

            #求出[1,238]各个点出的曲率
            x_z=[]#【1，238】
            k_z=[]#曲率值
            for i in range(1,img.shape[1]-1,1):
                x_z.append(i)
                k_z.append(derivative(fun1,i,dx=1e-6,n=2)/math.pow((1+math.pow(derivative(fun1,i,dx=1e-6),2)),1.5))

            #对曲率也滤波处理一下
            k_z = self.filter_fun(k_z, self.row_col_cur_windowsize)

            # #画出曲率图
            # plt.subplot(2,1,2)
            # plt.plot(x_z,k_z)

            #找出局部局部最大点做在凸起的宽度
            left_x=[]
            right_x=[]
            for i in range(len(k_z)-1):#i是索引，item是k_z中的项
                if(k_z[i]<0 and k_z[i+1]>=0):
                    left_x.append(i+0.5)
                elif(k_z[i]>=0 and k_z[i+1]<0):
                    right_x.append(i+0.5)

            #求出突起的距离
            if(left_x and right_x):
                if(len(left_x)==len(right_x)):
                    if(right_x[0]>left_x[0]):
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
                    else:
                        del(right_x[0])
                        left_x.pop()
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
                else:
                    if(len(right_x)>len(left_x)):
                        del(right_x[0])
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
                    else:
                        left_x.pop()
                        distinct=list(map(lambda x: x[0]-x[1], zip(right_x, left_x)))
            else:
                distinct=[0]

            #找出局部最大值及其对应的x
            localMax=[]
            for j in range(min(len(left_x),len(right_x))):
                max=0
                index=0
                for i in range(int(left_x[j]+0.5),int(right_x[j]+0.5)):
                    if(k_z[i]>=max):
                        max=k_z[i]
                        index=i
                tup=(max,index,distinct[j])#（高度，高度对应的索引-z值，宽度）
                localMax.append(tup)

            #计算Scr(z')
            Scr=list(map(lambda x:(x[0]*x[2],x[1]),localMax))

            #把Src加进去
            for item in Scr:
                img_V[k,item[1]]=img_V[k,item[1]]+item[0]
            print('按照行进行描点，第%d行' % (k))
        return img_V

    def compute_cross_point(self,X0, Y0, Xn, Yn, b, img,mode):  # X0表示直线x=0,同理
        '''
        计算直线和x=0,x=Xn,y=0,y=Yn的交叉点
        :param X0: 代表直线x=0
        :param Y0: 代表直线y=0
        :param Xn: 代表直线x=Xn
        :param Yn: 代表直线y=Yn
        :param b:  代表直线y=士x+b中的b
        :param img: 原始图像
        :param mode: 左下至右上，mode：0，左上至右下，mode:1
        :return:
        '''
        if(mode==0):
            if (b <= img.shape[0] - 1):  # 与x=0，y=0直线相交
                row_start = b - X0
                row_end = Y0
                col_start = X0
                col_end = b - Y0
            elif (b <= img.shape[1] - 1):
                row_start = Yn
                row_end = Y0
                col_start = b - Yn
                col_end = b
            else:
                row_start = Yn
                row_end = b - Xn
                col_start = b - Yn
                col_end = Xn
            return row_start, row_end, col_start, col_end
        else:
            if (b >= 0):  # 与x=0，y=Yn直线相交
                row_start = b + X0
                row_end = Yn
                col_start = X0
                col_end = Yn - b
            elif (b >= (img.shape[0] - img.shape[1])):
                row_start = Y0
                row_end = Yn
                col_start = Y0 - b
                col_end = Yn - b
            else:
                row_start = Y0
                row_end = b + Xn
                col_start = Y0 - b
                col_end = Xn
            return row_start, row_end, col_start, col_end

    def oblique_d2u_scan_(self,img,img_V):
        '''
        从左下向右上45°扫描
        '''
        #执行扫描，每个temp_list都是一列扫描值
        b=img.shape[1]+img.shape[0]-2
        for k in range(b+1):
            row_start,row_end,col_start,col_end=self.compute_cross_point(0,0,img.shape[1]-1,img.shape[0]-1,k,img,mode=0)
            temp_list=[]
            temp_list_pair=[]
            for row_col in zip(list(range(row_start,row_end-1,-1)),list(range(col_start,col_end+1))):
                temp_list.append(img[row_col[0]][row_col[1]])#扫描到的点
                temp_list_pair.append((row_col[0],row_col[1]))#扫描到的点对应的坐标（行，列）

            #对扫描到的一列值进行处理
            if(len(temp_list)>3):#至少4个点才能求曲率和三次插值
                x = np.linspace(0, len(temp_list)-1, len(temp_list))  # 0-239分成240份,即1，2，3.。。
                y = temp_list  # 图像中第一列像素点值

                # 对y进行滤波处理
                if(len(temp_list)>=self.oblique_widowsize):
                    y = self.filter_fun(y, self.oblique_widowsize)
                else:
                    y=self.filter_fun(y,len(temp_list))

                # 三次样条插值,拟合出离散点曲线，用于求导数
                fun1 = interp1d(x, y, kind='cubic')
                x_new = np.linspace(0, len(temp_list)-1, len(temp_list))  # 0-239，分为240个点
                y_new = fun1(x_new)

                # #画出拟合后的曲线图
                # plt.subplot(2,1,1)
                # plt.plot(x_new,y_new)

                # 求出[1,238]各个点出的曲率,首尾两个点无法求
                x_z = []  # 【1，238】
                k_z = []  # 曲率值
                for i in range(1, len(temp_list)-1, 1):
                    x_z.append(i)
                    k_z.append(
                        derivative(fun1, i, dx=1e-6, n=2) / math.pow((1 + math.pow(derivative(fun1, i, dx=1e-6), 2)), 1.5))
                #temp_list对应的首尾坐标也去除
                del(temp_list_pair[0])#除去第一个值
                temp_list_pair.pop()#除去最后一个值

                # 对曲率也滤波处理一下
                if (len(temp_list) >= self.oblique_cur_widowsize):
                    k_z = self.filter_fun(k_z, self.oblique_cur_widowsize)
                else:
                    k_z = self.filter_fun(k_z, len(k_z))

                # #画出曲率图
                # plt.subplot(2,1,2)
                # plt.plot(x_z,k_z)

                # 找出局部局部最大点所在凸起的宽度
                left_x = []
                right_x = []
                for i in range(len(k_z) - 1):  # i是索引，item是k_z中的项
                    if (k_z[i] < 0 and k_z[i + 1] >= 0):
                        left_x.append(i + 0.5)
                    elif (k_z[i] >= 0 and k_z[i + 1] < 0):
                        right_x.append(i + 0.5)

                # 求出突起的距离
                if (left_x and right_x):
                    if (len(left_x) == len(right_x)):
                        if (right_x[0] > left_x[0]):
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                        else:
                            del (right_x[0])
                            left_x.pop()
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                    else:
                        if (len(right_x) > len(left_x)):
                            del (right_x[0])
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                        else:
                            left_x.pop()
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                else:
                    distinct = [0]

                # 找出局部最大值及其对应的x
                localMax = []
                for j in range(min(len(left_x),len(right_x))):
                    max = 0
                    index = 0
                    for i in range(int(left_x[j] + 0.5), int(right_x[j] + 0.5)):
                        if (k_z[i] >= max):
                            max = k_z[i]
                            index = i
                    tup = (max, index, distinct[j])  # （高度，高度对应的索引-z值，宽度）
                    localMax.append(tup)

                # 计算Scr(z')
                Scr = list(map(lambda x: (x[0] * x[2], x[1]), localMax))

                # 把Src加进去
                for item in Scr:
                    img_V[temp_list_pair[item[1]][0], temp_list_pair[item[1]][1]] = img_V[temp_list_pair[item[1]][0], temp_list_pair[item[1]][1]] + item[0]
                print('左下至右上45°进行描点，第%d行' % (k))
        return img_V

    def oblique_u2d_scan_(self,img,img_V):
        '''
        左上至右下扫描
        '''
        #执行扫描，每个temp_list都是一列扫描值
        for k in range(img.shape[0],-(img.shape[1]+1),-1):
            row_start,row_end,col_start,col_end=self.compute_cross_point(0,0,img.shape[1]-1,img.shape[0]-1,k,img,mode=1)
            temp_list=[]
            temp_list_pair=[]
            for row_col in zip(list(range(row_start,row_end+1)),list(range(col_start,col_end+1))):
                temp_list.append(img[row_col[0]][row_col[1]])#扫描到的点
                temp_list_pair.append((row_col[0],row_col[1]))#扫描到的点对应的坐标（行，列）

            #对扫描到的一列值进行处理
            if(len(temp_list)>3):#至少4个点才能求曲率和三次插值
                x = np.linspace(0, len(temp_list)-1, len(temp_list))  # 0-239分成240份,即1，2，3.。。
                y = temp_list  # 图像中第一列像素点值

                # 对y进行滤波处理
                if(len(temp_list)>=self.oblique_widowsize):
                    y = self.filter_fun(y, self.oblique_widowsize)
                else:
                    y=self.filter_fun(y,len(temp_list))

                # 三次样条插值,拟合出离散点曲线，用于求导数
                fun1 = interp1d(x, y, kind='cubic')
                x_new = np.linspace(0, len(temp_list)-1, len(temp_list))  # 0-239，分为240个点
                y_new = fun1(x_new)

                # #画出拟合后的曲线图
                # plt.subplot(2,1,1)
                # plt.plot(x_new,y_new)

                # 求出[1,238]各个点出的曲率,首尾两个点无法求
                x_z = []  # 【1，238】
                k_z = []  # 曲率值
                for i in range(1, len(temp_list)-1, 1):
                    x_z.append(i)
                    k_z.append(
                        derivative(fun1, i, dx=1e-6, n=2) / math.pow((1 + math.pow(derivative(fun1, i, dx=1e-6), 2)), 1.5))
                #temp_list对应的首尾坐标也去除
                del(temp_list_pair[0])#除去第一个值
                temp_list_pair.pop()#除去最后一个值

                # 对曲率也滤波处理一下
                if (len(temp_list) >= self.oblique_cur_widowsize):
                    k_z = self.filter_fun(k_z, self.oblique_cur_widowsize)
                else:
                    k_z = self.filter_fun(k_z, len(k_z))

                # #画出曲率图
                # plt.subplot(2,1,2)
                # plt.plot(x_z,k_z)

                # 找出局部局部最大点做在凸起的宽度
                left_x = []
                right_x = []
                for i in range(len(k_z) - 1):  # i是索引，item是k_z中的项
                    if (k_z[i] < 0 and k_z[i + 1] >= 0):
                        left_x.append(i + 0.5)
                    elif (k_z[i] >= 0 and k_z[i + 1] < 0):
                        right_x.append(i + 0.5)

                # 求出突起的距离
                if (left_x and right_x):
                    if (len(left_x) == len(right_x)):
                        if (right_x[0] > left_x[0]):
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                        else:
                            del (right_x[0])
                            left_x.pop()
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                    else:
                        if (len(right_x) > len(left_x)):
                            del (right_x[0])
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                        else:
                            left_x.pop()
                            distinct = list(map(lambda x: x[0] - x[1], zip(right_x, left_x)))
                else:
                    distinct = [0]

                # 找出局部最大值及其对应的x
                localMax = []
                for j in range(min(len(left_x),len(right_x))):
                    max = 0
                    index = 0
                    for i in range(int(left_x[j] + 0.5), int(right_x[j] + 0.5)):
                        if (k_z[i] >= max):
                            max = k_z[i]
                            index = i
                    tup = (max, index, distinct[j])  # （高度，高度对应的索引-z值，宽度）
                    localMax.append(tup)

                # 计算Scr(z')
                Scr = list(map(lambda x: (x[0] * x[2], x[1]), localMax))

                # 把Src加进去
                for item in Scr:
                    img_V[temp_list_pair[item[1]][0], temp_list_pair[item[1]][1]] = img_V[temp_list_pair[item[1]][0], temp_list_pair[item[1]][1]] + item[0]
                print('左上至右下45°进行描点，第%d行' % (k))
        return img_V

    def connect_fun(self,img_V,threshold):
        '''
        连接函数，用于四个方向的连接
        :param img_V: 完成Scr计算的img_V
        threshold: 用于二值化图像的阈值
        :return: 完成连接并二值化后的img_V_end
        '''
        # 按列连接
        img_V1 = np.zeros((img_V.shape[0], img_V.shape[1]), dtype=np.uint8)
        for i in range(img_V.shape[1]):
            col_vector = img_V[:, i]
            for j in range(len(col_vector)):
                if (j == 0 or j == len(col_vector) - 1):
                    img_V1[j][i] = img_V[j][i]
                elif (j == 1 or j == len(col_vector) - 2):
                    img_V1[j][i] = min(img_V[j - 1][i], img_V[j + 1][i])
                else:
                    img_V1[j][i] = min(max(img_V[j - 1][i], img_V[j - 2][i]), max(img_V[j + 1][i], img_V[j + 2][i]))

        # 按行连接
        img_V2 = np.zeros((img_V.shape[0], img_V.shape[1]), dtype=np.uint8)
        for i in range(img_V.shape[0]):
            row_vector = img_V[i, :]
            for j in range(len(row_vector)):
                if (j == 0 or j == len(row_vector) - 1):
                    img_V2[i][j] = img_V[i][j]
                elif (j == 1 or j == len(row_vector) - 2):
                    img_V2[i][j] = min(img_V[i][j - 1], img_V[i][j + 1])
                else:
                    img_V2[i][j] = min(max(img_V[i][j - 1], img_V[i][j - 2]), max(img_V[i][j + 1], img_V[i][j + 2]))

        # 左下至右上45°
        img_V3 = np.zeros((img_V.shape[0], img_V.shape[1]), dtype=np.uint8)
        # 执行扫描，每个temp_list都是一列扫描值
        b = img_V.shape[1] + img_V.shape[0] - 2
        for k in range(b + 1):
            row_start, row_end, col_start, col_end = self.compute_cross_point(0, 0, img_V.shape[1] - 1, img_V.shape[0] - 1,
                                                                         k, img_V,mode=0)
            temp_list = []
            temp_list_pair = []
            for row_col in zip(list(range(row_start, row_end - 1, -1)), list(range(col_start, col_end + 1))):
                temp_list.append(img_V[row_col[0]][row_col[1]])  # 扫描到的点
                temp_list_pair.append((row_col[0], row_col[1]))  # 扫描到的点对应的坐标（行，列）

            for i in range(len(temp_list)):
                if (i == 0 or i == len(temp_list) - 1):
                    img_V3[temp_list_pair[i][0], temp_list_pair[i][1]] = img_V[
                        temp_list_pair[i][0], temp_list_pair[i][1]]
                elif (i == 1 or i == len(temp_list) - 2):
                    img_V3[temp_list_pair[i][0], temp_list_pair[i][1]] = min(
                        img_V[temp_list_pair[i - 1][0], temp_list_pair[i - 1][1]],
                        img_V[temp_list_pair[i + 1][0], temp_list_pair[i + 1][1]])
                else:
                    img_V3[temp_list_pair[i][0], temp_list_pair[i][1]] = min(
                        max(img_V[temp_list_pair[i - 1][0], temp_list_pair[i - 1][1]], \
                            img_V[temp_list_pair[i - 2][0], temp_list_pair[i - 2][1]]), \
                        max(img_V[temp_list_pair[i + 1][0], temp_list_pair[i + 1][1]], \
                            img_V[temp_list_pair[i + 1][0], temp_list_pair[i + 1][1]]))

        # 左上至右下45°
        img_V4 = np.zeros((img_V.shape[0], img_V.shape[1]), dtype=np.uint8)
        # 执行扫描，每个temp_list都是一列扫描值
        for k in range(img_V.shape[0], -(img_V.shape[1] + 1), -1):
            row_start, row_end, col_start, col_end = self.compute_cross_point(0, 0, img_V.shape[1] - 1, img_V.shape[0] - 1,
                                                                         k, img_V,mode=1)
            temp_list = []
            temp_list_pair = []
            for row_col in zip(list(range(row_start, row_end + 1)), list(range(col_start, col_end + 1))):
                temp_list.append(img_V[row_col[0]][row_col[1]])  # 扫描到的点
                temp_list_pair.append((row_col[0], row_col[1]))  # 扫描到的点对应的坐标（行，列）

            for i in range(len(temp_list)):
                if (i == 0 or i == len(temp_list) - 1):
                    img_V4[temp_list_pair[i][0], temp_list_pair[i][1]] = img_V[
                        temp_list_pair[i][0], temp_list_pair[i][1]]
                elif (i == 1 or i == len(temp_list) - 2):
                    img_V4[temp_list_pair[i][0], temp_list_pair[i][1]] = min(
                        img_V[temp_list_pair[i - 1][0], temp_list_pair[i - 1][1]],
                        img_V[temp_list_pair[i + 1][0], temp_list_pair[i + 1][1]])
                else:
                    img_V4[temp_list_pair[i][0], temp_list_pair[i][1]] = min(
                        max(img_V[temp_list_pair[i - 1][0], temp_list_pair[i - 1][1]], \
                            img_V[temp_list_pair[i - 2][0], temp_list_pair[i - 2][1]]), \
                        max(img_V[temp_list_pair[i + 1][0], temp_list_pair[i + 1][1]], \
                            img_V[temp_list_pair[i + 1][0], temp_list_pair[i + 1][1]]))

        # 最终连接,给定阈值标记每个点
        img_V_end = np.zeros((img_V.shape[0], img_V.shape[1]), dtype=np.uint8)
        for i in range(img_V_end.shape[0]):
            for j in range(img_V_end.shape[1]):
                if (max(img_V1[i][j], img_V2[i][j], img_V3[i][j], img_V4[i][j]) >= threshold):
                    img_V_end[i][j] = 255
                else:
                    img_V_end[i][j] = 0
        return img_V_end

    def show_img(self,title,img):
        cv2.imshow(title, img)
        cv2.waitKey(0)

if __name__=="__main__":
    test=LMC()
