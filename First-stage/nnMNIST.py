
# 运用LeNet卷积神经网络来写手写数字识别

import struct
import time
import numpy as np
import os
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss, nn
import mxnet as mx
import matplotlib.pyplot as plt


'''
    读取数据
'''

def load_mnist_data(path, kind="train"):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        # 'I'表示一个无符号整数，大小为四个字节
        # '>II'表示读取两个无符号整数，即8个字节
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


'''
    数据格式的封装
'''

def change_Dataformat():
    X_train, y_train = load_mnist_data("", kind="train")  # 数据加载,np格式
    X_test, y_test = load_mnist_data("", kind="t10k")
    X_train = X_train.reshape((-1, 1, 28, 28))  # 卷积层是28x28而不是786
    X_test = X_test.reshape((-1, 1, 28, 28))

    X_train = nd.array(X_train)  # 使用mxnet,转换成nd格式
    X_test = nd.array(X_test)
    y_train = nd.array(y_train)
    y_test = nd.array(y_test)

    state = np.random.get_state()   # 同步打乱数据
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    state = np.random.get_state()  # 同步打乱数据
    np.random.shuffle(X_test)
    np.random.set_state(state)
    np.random.shuffle(y_test)

    return X_train,y_train,X_test,y_test


'''
    使用gpu加速计算
'''

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


'''
    卷积层的定义
'''
def Net():
    net = nn.Sequential()
    # 输入28x28
    # 第一层之后就是28-5+1 = 24x24x6
    # 池化层之后应该就是12x12x6
    # 第三层之后就是12-5+1 = 8x8x16
    # 池化层之后就是4x4x16
    net.add(nn.Conv2D(channels=6,kernel_size =5,activation='sigmoid'),
            nn.MaxPool2D(pool_size=2,strides =2),
            nn.Conv2D(channels = 16, kernel_size = 5, activation = 'sigmoid'),
            nn.MaxPool2D(pool_size =2 , strides = 2),
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10))
    return net


'''
    准确率的评估
'''

def evaluate_accuracy(net):
    _ , _ , X_test, y_test = change_Dataformat()
    X = X_test
    y = y_test.astype('float32')
    acc_sum = (net(X).argmax(axis=1)==y).sum().asscalar()
    n = y.size
    return acc_sum/n


'''
    画图函数
'''

def plot(loss_list,acc_list,test_acc_list):
    # 画图，包括损失函数值，训练正确率，测试正确率
    print('最大训练准确率 %.3f' %(max(acc_list)))
    print('最大测试准确率 %.3f' % (max(test_acc_list)))

    loss_list = np.array(loss_list)     # list转mumpy
    acc_list = np.array(acc_list)
    test_acc_list = np.array(test_acc_list)
    num = np.arange(num_epochs)

    plt.figure(12)  # 理解成画板
    ax1 = plt.subplot(311)
    ax1.plot(num,loss_list,'-o')    # 原点
    ax1.set_title("loss")

    ax2 = plt.subplot(312)
    ax2.plot(num,acc_list,'-o')
    ax2.set_title("train_acc")

    ax3 = plt.subplot(313)
    ax3.plot(num, test_acc_list, '-o')
    ax3.set_title("test_acc")
    plt.show()


'''
    开始训练
'''
def train(batch_size,trainer,num_epochs):
    loss_list = []
    acc_list = []
    test_acc_list = []
    loss = gloss.SoftmaxCrossEntropyLoss()  # 交叉熵损失函数
    ctx = try_gpu()
    print('training on',ctx)
    X_train,y_train,_,_ = change_Dataformat()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        # 训练集是60000张,那么我们每次取batch_size 的大小.
        for i in range(int(60000/batch_size)):
            X = X_train[i * batch_size: i * batch_size + batch_size]    # 批量
            y = y_train[i * batch_size: i * batch_size + batch_size]
            X,y = X.as_in_context(ctx),y.as_in_context(ctx)

            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()                # 更新权重
            trainer.step(batch_size)    # 训练
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum +=(y_hat.argmax(axis=1)==y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(net)
        loss_list.append(train_l_sum / n)
        acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f,time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))
    return loss_list, acc_list, test_acc_list
if __name__ == '__main__':
    lr,num_epochs = 0.9,10
    batch_size = 200
    net = Net() # 定义网络模型
    net.initialize(force_reinit=True,init=init.Xavier())    # 首次对模型初始化需要指定force_reinit为真
    # init=init.Xavier() 一种参数初始化的方式
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
    loss_list, acc_list, test_acc_list = train(batch_size,trainer,num_epochs)
    plot(loss_list, acc_list, test_acc_list)
