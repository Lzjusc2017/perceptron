
'''

    一张车牌，字符分割作为训练集.
'''
from time import sleep

import cv2
import numpy as np
'''
    加载数据
'''
def load_image():
    img = cv2.imread('A01_N84E28_0.jpg')
    return img


'''
    灰度化
'''
def cvtGrayImage(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

'''
    二值化
'''
def dobinaryzation(img):
    '''
        二值化处理函数
        '''
    maxi = float(img.max())
    mini = float(img.min())

    x = maxi - ((maxi - mini) / 2)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret, thresh = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    # 返回二值化后的黑白图像
    return thresh

'''
    找终点
'''

def find_end(start,black,white,width,black_max,white_max):
    end = start + 1
    for m in range(start+1,width-1):
        if (black[m]>0.95*black_max):
            end = m
            break
    return end


'''
    分割字符
'''

def split_chars(img):
    white = []
    black = []
    height = img.shape[0]  # 高
    width = img.shape[1]  # 宽
    print('height %d,width %d'%(height,width))
    white_max = 0
    black_max = 0
    print(img[0][100])
    # 计算每一列的黑白像素总和
    for i in range(width):
        line_white = 0
        line_black = 0
        for j in range(height):
            if (img[j][i]==255):    # 行，列
                line_white = line_white + 1
            else:
                line_black = line_black + 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)

    print('white', white)
    print('black', black)
    n = 1
    count = 0
    mat = []
    while n<width-2:
        n = n + 1
        # 黑底白字
        if (white[n] >0.05*white_max):
            # 判断是字
            start = n
            end = find_end(start,black,white,width,black_max,white_max)
            n = end
            if (end-start>5):
                cj = img[:height,start:end]
                mat.append(cj)
                count = count + 1
    print(count)
    return mat

def welt_chars(img):

    # 针对各种情况，贴边
    start = 0
    end = 0
    list = []
    for m in range(len(img)):
        height = mat[m].shape[0]
        width = mat[m].shape[1]

        for i in range(height):
            line_white = 0
            for j in range(width):
                if (mat[m][i][j]==255):
                    line_white = line_white + 1
            if (line_white>1):
                start = i
                break
        for i in range(height-1,-1,-1):
            line_white = 0
            for j in range(width):
                if (mat[m][i][j]==255):
                    line_white = line_white + 1
            if (line_white>1):
                end = i
                break
        list.append(mat[m][start:end,:width])
        cv2.imshow(str(m),mat[m][start:end,:width])
    return list
    # height = mat[3].shape[0]    # 36
    # width = mat[3].shape[1]     # 12
    # print(height,width)
    # for i in range(height):
    #     line_white = 0
    #     for j in range(width):
    #         if (mat[3][i][j] == 255):
    #             line_white = line_white + 1
    #     print(line_white)
'''
    两边添加像素
'''

def addPixel(mat):
    # 上下贴边,然后转换成20x20
    max_height = 0
    for i in range(len(mat)):
        max_height = max(max_height,mat[i].shape[0])
    print('mat_height',max_height)
    for i in range(len(mat)):
        #print(mat[i].shape[0],mat[i].shape[1])  # 0是高,1是宽
        width = max_height-mat[i].shape[1]  # 宽（20,19）
        half_width = int(width/2)
        mat[i] = cv2.copyMakeBorder(mat[i],0,0,half_width,width-half_width,cv2.BORDER_CONSTANT)
        mat[i] = cv2.resize(mat[i],dsize = (20,20))
        print(mat[i].shape[0],mat[i].shape[1])
        cv2.imshow(str(i),mat[i])
        cv2.imwrite(str(i)+'.jpg',mat[i])


'''
    小膨胀
'''

def expansion(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素的形状和大小
    for i in range(len(mat)):
        img[i] = cv2.dilate(img[i],kernel)
    cv2.imshow(str('23'),img[0])

if __name__ == '__main__':
    img = load_image()              # 加载图片
    grayImage = cvtGrayImage(img)   # 灰度化
    binaryImage = dobinaryzation(grayImage) # 二值化
    cv2.imshow('c',binaryImage)     # 显示原图
    mat = split_chars(binaryImage)  # 分割图片
    mat = welt_chars(mat)           # 贴边
    addPixel(mat)                   # 边界拓展
    expansion(mat)                  # 膨胀
    cv2.waitKey(0)