'''

    使用感知机实现车牌字母识别
    0-9,A-Z
	50轮,训练准确率99.7%,测试准确率99.3%
'''

import os
import time

import numpy as np
import cv2
from mxnet import gluon,init
from mxnet.gluon import loss as gloss,nn
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd
import d2lzh as d2l


'''
    数据的加载
'''

def load_Plate_data(path,plates_list,plates_number,plates_size):
    train_iter = np.zeros((len(plates_list)*plates_number,plates_size))
    # 3个值，每个400个训练集.
    l = 0
    for plate in plates_list:
        Directory_name = path + "\\" + plate + "\\"
        print(Directory_name)
        for filename in os.listdir(r"./" + Directory_name):
            img = cv2.imread(Directory_name + "/" + filename, 0)
            cv2.threshold(img, 140, 1, cv2.THRESH_BINARY, img)  # threshold the mat
            train_iter[l] = np.reshape(img,plates_size)
            l = l + 1  # 400 train image
            if (l%plates_number==0):
                break
    return train_iter

'''
    数据标签的设置
'''

def set_label(plates_labels,plates_number):
    train_label = np.zeros((len(plates_labels)*plates_number,))
    for i in range(len(plates_labels)):
        for j in range(plates_number):
            train_label[plates_number*i + j] = int(plates_labels[i])
    return train_label

'''
    打乱数据
'''

def shuffle_data(plates_labels,plates_list,plates_number,plates_size):
    '''
    :param plates_labels    车牌文件夹的数字顺序0-xxx
    :param plates_list:     车牌文件夹的名称
    :param plates_number:   每一类车牌的数量
    :param plates_size:     测试的大小是
    :return:
    '''
    # 先打乱后随即取，不然会出错

    datas = load_Plate_data('Train', plates_list, plates_number, plates_size)
    print('datas',len(datas))
    labels = set_label(plates_labels, plates_number)

    # 在转换成nd前打乱数据
    state = np.random.get_state()  # 同步打乱数据
    np.random.shuffle(datas)
    np.random.set_state(state)
    np.random.shuffle(labels)
    # 10张0-9
    train_iter = datas[:int(0.9*len(datas))]
    train_labels = labels[:int(0.9*len(datas))]
    test_iter = datas[int(0.9*len(datas)):]
    test_labels = labels[int(0.9*len(datas)):]

    return nd.array(train_iter),nd.array(train_labels),nd.array(test_iter),nd.array(test_labels)

'''
    找到样本数最少
'''

def find_plates_number(path):
    min_number = 99999
    catalogue = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
                 'M', 'N',
                 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for list_file in catalogue:
        Directory_name = path + "\\" + list_file + "\\"
        count = 0
        for filename in os.listdir(r"./" + Directory_name):
            count = count + 1
        if count<min_number:
            min_number = count
            filename = list_file
    print(list_file)
    return min_number

'''
    定义网络模型
'''

def Net(num_hiddens_one,num_hiddens_two,num_outputs):
    net = nn.Sequential()
    net.add(nn.Dense(num_hiddens_one, activation = 'relu'),
            nn.Dense(num_hiddens_two, activation = 'relu'),
            nn.Dense(num_outputs))
    return net

'''
    尝试使用gpu
'''

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


'''
    训练模型
'''

def train(train_iter,train_label,test_iter,test_label,net,trainer,ctx,num_epochs,batch_size,number_sum):
    loss_list = []
    acc_list = []
    test_acc_list = []
    loss = gloss.SoftmaxCrossEntropyLoss()
    print('training on',ctx)
    for epoch in range(num_epochs):
        # 批量
        train_l_sum,train_acc_sum,n,start= 0.0,0,0,time.time()
        for m in range(int(number_sum/batch_size)):
            X = train_iter[m*batch_size:m*batch_size + batch_size]
            y = train_label[m*batch_size:m*batch_size + batch_size]
            X,y = X.as_in_context(ctx),y.as_in_context(ctx)
            with d2l.autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum() # 损失函数和
            l.backward()    # 梯度下降
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum +=l.asscalar()
            train_acc_sum +=(y_hat.argmax(axis=1)==y).sum().asscalar()
            n +=y.size
        test_acc = evaluate_accuracy(net,ctx,test_iter,test_label)
        loss_list.append(train_l_sum / n)
        acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)
        if (epoch%10==0):
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,n is %d,time %.1f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, n,time.time()-start))
    return loss_list, acc_list, test_acc_list
'''
    测试准确率的测定
'''

def evaluate_accuracy(net,ctx,test_iter,test_label):
    test_iters,test_labels = test_iter.as_in_context(ctx),test_label.as_in_context(ctx)
    test_labels = test_labels.astype('float32')
    acc_sum = (net(test_iters).argmax(axis=1) == test_labels).sum()
    return acc_sum.asscalar()/test_labels.size

'''
    画出损失函数，训练准确率，测试准确率
'''

def plot(loss_list,acc_list,test_acc_list):
    '''

    :param loss_list:   损失函数列表
    :param acc_list:    训练准确率列表
    :param test_acc_list:   测试准确率列表
    :return:    None
    '''
    print('最大训练准确率 %.3f' % (max(acc_list)))
    print('最大测试准确率 %.3f' % (max(test_acc_list)))

    loss_list = np.array(loss_list)  # list转mumpy
    acc_list = np.array(acc_list)
    test_acc_list = np.array(test_acc_list)
    num = np.arange(num_epochs)

    plt.figure(12)  # 理解成画板
    ax1 = plt.subplot(311)
    ax1.plot(num, loss_list, '-o')  # 原点
    ax1.set_title("loss")

    ax2 = plt.subplot(312)
    ax2.plot(num, acc_list, '-o')
    ax2.set_title("train_acc")

    ax3 = plt.subplot(313)
    ax3.plot(num, test_acc_list, '-o')
    ax3.set_title("test_acc")
    plt.show()

'''
    画出具体的图形
'''

def Test_model(net,test_iter,test_label,plates_list):
    '''
        选取测试图片
    :return:
    '''
    fig1 = plt.figure(figsize=(8, 8))
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # 预测标签
    predict = net(test_iter).argmax(axis=1) # 预测标签
    test_iter = test_iter.asnumpy() # 转成numpy

    for i in range(100):  # 10x10
        ax = fig1.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(np.reshape(test_iter[i], [20, 20]), cmap=plt.cm.binary, interpolation='nearest')
        if (predict[i].asscalar() == test_label[i]):
            ax.text(0, 0, "pred:" + str(plates_list[int(predict[i].asscalar())]), color='blue')
        else:
            ax.text(0, 0, "pred:" + str(plates_list[int(predict[i].asscalar())]), color='red')
        # ax.text(0,32,"real:"+str(test_labels[i]),color='blue')
    plt.show()

if __name__ == '__main__':

    # 关于训练数据,这里不规定多少张。
    min_number = find_plates_number('Train')
    print(min_number)  # 打印最小的样本数，方便我们选取训练数据集，这里设定2000张
    # 这里设定2000张，取2000张的10%作为测试集.
    plates_number = 2000   # 训练数据2000
    # 测试300
    plates_size = 400   # 20x20
    plates_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
                 'M', 'N','P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    plates_labels = []
    for i in range(len(plates_list)):
        plates_labels.append(str(i))    # 标签的设置
    # 标签的标号必须从0开始，不然训练数据会出错..
    # 遍历所有样本,方便选取样本张数.
    num_hiddens_one = 256
    num_hiddens_two = 128
    num_outputs = len(plates_list)
    lr = 0.05
    num_epochs = 50
    batch_size = 200
    number_sum = len(plates_list)*(int(0.9*plates_number))     # 训练总数据集
    train_iter, train_labels, test_iter, test_labels = shuffle_data(plates_labels,plates_list,plates_number,plates_size)
    ctx = try_gpu()  # 使用gpu
    net = Net(num_hiddens_one,num_hiddens_two,num_outputs)
    net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    print('总共批量',int(number_sum/batch_size))
    print(len(train_iter),len(train_labels),len(test_iter),len(test_labels))    # 打印数据集的大小
    #
    loss_list, acc_list, test_acc_list = train(train_iter,train_labels,test_iter,test_labels,net,trainer,ctx,num_epochs,batch_size,number_sum)
    plot(loss_list, acc_list, test_acc_list)            # 画出损失函数的值
    # 测试E
    # img = cv2.imread('4.jpg',0)
    # cv2.threshold(img, 140, 1, cv2.THRESH_BINARY, img)  # threshold the mat
    # img = np.reshape(img, (1, 400))
    # img = nd.array(img)
    # print('predict')
    # print(net(img))
    # print(net(img).argmax(axis=1))
    # print(plates_list[int(net(img).argmax(axis=1).asscalar())])  # 打印标签
    # Test_model(net,test_iter,test_labels,plates_list)           # 画出数字


