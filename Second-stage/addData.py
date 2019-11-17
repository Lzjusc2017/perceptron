

'''
    图像数据集增强方式总结和实现
    旋转，旋转角度可调
    缩放，分辨率可调
    添加噪音，背景噪音程度可调
    图像模糊，模糊程度可调
    图像移动 将图像往x、y方向上按指定的数量移动图像像素
    支持拓展其他新的图像数据集增强方式
'''
import random

import cv2
import os
import numpy as np

'''
    加载文件
'''
def load_Plate_data(path,catalogue):
    # this loop is for read each image in this foder,directory_name is the foder name with images.

    # 一张变4张.
    # 读取文件,看看有几张

    for list_file in catalogue:
        Directory_name = path + "\\" + list_file +"\\"
        list = []
        count = 0
        for filename in os.listdir(r"./" + Directory_name):
            count = count + 1
        if count>2000:          # 大于2000张，不操作
            continue
        for filename in os.listdir(r"./" + Directory_name):
            src_img = cv2.imread(Directory_name + "\\" + filename,0)    # 灰度读取
            # 进行变换
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素的形状和大小
            #img1 = cv2.dilate(src_img,kernel)   # 膨胀
            g = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # 形态学处理,开运算
            img1 = cv2.morphologyEx(src_img, cv2.MORPH_OPEN, g)

            positiveAngle = random.randint(1, 4)    #产生 1 到 4 的一个整数型随机数
            negativeAngle = random.randint(-4,-1)   # 产生-4到-1的一个整数型随机数
            p = cv2.getRotationMatrix2D((src_img.shape[0]//2,src_img.shape[1]//2),positiveAngle,scale=1) # 按中心旋转1-5°
            n = cv2.getRotationMatrix2D((src_img.shape[0]//2,src_img.shape[1]//2),negativeAngle,scale=1)
            img2 = cv2.warpAffine(src_img, p, (src_img.shape[0],src_img.shape[1]))
            img3 = cv2.warpAffine(src_img, n, (src_img.shape[0], src_img.shape[1]))
            # 改变亮度
            # OpenCV中亮度和对比度应用这个公式来计算：g(x) = αf(x) + β，
            # 其中：α(>0)、β常称为增益与偏置值，分别控制图片的对比度和亮度。
            img4 = cv2.addWeighted(src_img,1.1,src_img,0,10)    # 改变亮度

            img5 = cv2.blur(src_img, (2, 2))    # 2x2均值滤波

            img6 = src_img[1:19,1:19]
            img6 = cv2.resize(img6,dsize = (20,20))
            list.append(img1)
            list.append(img2)
            list.append(img3)
            list.append(img4)

        #print(Directory_name + "\\" + str(1)+".jpg")
        for i in range(len(list)):
            cv2.imwrite(Directory_name + "\\" + str(i)+".jpg",list[i])

        #print((src_img==img5).all())
        print(Directory_name)
        cv2.waitKey(0)



if __name__ == '__main__':
    catalogue = ['0','1','2','3','4','5','6','7','8','9','10','A','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
    images = load_Plate_data('Train',catalogue)
