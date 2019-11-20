from time import sleep

import math
from mxnet import nd
import random
import zipfile
from mxnet.gluon import loss as gloss
from mxnet import autograd,nd
import time
import mxnet as mx
import d2lzh as d2l
'''
    加载歌词和字符转索引，索引转字符
'''

def load_data_jay_lyrics():
    with zipfile.ZipFile('data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    '''
        数据集的字符有60000多个字符,把换行符替换成空格
    '''
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:10000]

    '''
        建立字符索引
    '''
    idx_to_char = list(set(corpus_chars))  # 集合取不同元素，索引到字符
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])  # 根据索引取字符
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 字符转索引
    return corpus_indices,char_to_idx,idx_to_char,vocab_size

'''
    随机取样
'''


def data_iter_random(corpus_indices,batch_size,num_steps,ctx=None):
    # 减一是因为输出的索引是相应输入的索引加一
    num_examples = (len(corpus_indices)-1)
    epochs_size = num_examples//batch_size      # 轮数大小
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)             # 打乱数据
    print('num_examples',num_examples)

    # 返回从pos开始的长味num_steps的序列
    def _data(pos):
        return corpus_indices[pos:pos + num_steps]

    for i in range(epochs_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size              # 第几次的batch_size
        batch_indices = example_indices[i:i + batch_size]
        # print('example_indices',example_indices)
        # print('batch_indices',batch_indices)
        # sleep(500)
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps +1) for j in batch_indices]
        print(X)    # [[], [0, 1, 2, 3, 4, 5]]  越界问题
        # sleep(500)
        yield nd.array(X,ctx),nd.array(Y,ctx)


def to_onehot(X,size):
    # 返回one-hot向量
    return [nd.one_hot(x,size) for x in X.T]        # 返回one-hot 数组


'''
    使用gpu加速
'''
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

'''
    初始化模型参数
'''
def get_params(num_inputs,num_hiddens,num_outputs,ctx):
    def _one(shape):
        return nd.random.normal(scale = 0.01,shape=shape,ctx = ctx)
    # 隐藏层的参数
    W_xh = _one((num_inputs,num_hiddens))
    W_hh = _one((num_hiddens,num_hiddens))
    b_h = nd.zeros(num_hiddens,ctx=ctx)
    # 输出层参数
    W_hq = _one((num_hiddens,num_outputs))
    b_q = nd.zeros(num_outputs,ctx=ctx)

    # 附上梯度
    params = [W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.attach_grad()
    return params

'''
    定义模型
'''
def init_rnn_state(batch_size,num_hiddens,ctx):
    # 返回batch_size,num_hiddens的大小
    return (nd.zeros(shape=(batch_size,num_hiddens),ctx = ctx),)

def rnn(inputs,state,params):
    '''
    :param inputs:  num_steps个形状(batch_size,vocab_size)矩阵，batch_size可以理解为词的个数
    :param state:   batch_size*vocab_size
    :param params:
    :return: num_steps个形状(batch_size,vocab_size)矩阵
    '''
    # inputs和outputs都是num_steps个形状(batch_size,vocab_size)矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state          # 元组转化
    outputs = []
    for X in inputs:
        # H是上一次的输入，然后保存到H
        # H是 batch_size * num_hiddens
        # Y是 batch_size * vocab_size
        H = nd.tanh(nd.dot(X,W_xh) + nd.dot(H,W_hh) + b_h)
        Y = nd.dot(H,W_hq) + b_q
        outputs.append(Y)
    return outputs,(H,) # outputs是每一个神经元的输出
    # H会传递到下一层


'''
    定义预测函数
'''
def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,
                num_hiddens,vocab_size,ctx,idx_to_char,char_to_idx):
    '''
    :param prefix:              单词前缀
    :param num_chars:           预测输出的字符个数
    :param rnn:
    :param params:
    :param init_rnn_state:
    :param num_hiddens:
    :param vocab_size:
    :param ctx:
    :param idx_to_char:
    :param char_to_idx:
    :return:
    '''

    state = init_rnn_state(1,num_hiddens,ctx)
    output = [char_to_idx[prefix[0]]]       # 转换为索引列表
    for t in range(num_chars + len(prefix) - 1):
        # output[-1] 表示预测最后一个字符(索引表示)
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(nd.array([output[-1]],ctx = ctx),vocab_size)
        # 计算输出和更新隐藏状态
        (Y,state) = rnn(X,state,params) # state经过激活函数到下一层的输出
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t<len(prefix) - 1:
            # 当前还没有到最后一个字符
            # 把下一个字符添加进来
            output.append(char_to_idx[prefix[t+1]])
        else:
            # 添加预测的字符
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    # 返回索引对应的字符
    return ''.join([idx_to_char[i] for i in output])

'''
    裁剪梯度
'''

def grad_clipping(params,theta,ctx):
    '''
    :param params:
    :param theta:
    :param ctx:
    :return:
    '''
    norm = nd.array([0],ctx)
    for param in params:
        norm = norm + (param.grad**2).sum()     # 求第二范数
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] = param.grad[:] * theta / norm

'''
    (1) 使用困惑度评价模型
    (2) 迭代前裁剪梯度
'''

def train_and_predict_rnn(rnn,get_params,init_rnn_state,num_hiddens,vocab_size,ctx,corpus_indices,
                          idx_to_char,char_to_idx,is_random_iter,num_epochs,num_steps,
                          lr,clipping_theta,batch_size,pred_period,pred_len,prefixes):
    '''
    :param rnn:                 循环神经网络
    :param get_params:          参数权重
    :param init_rnn_state:      模型初始化
    :param num_hiddens:         隐藏层大小
    :param vocab_size:          不同字符的个数
    :param ctx:
    :param corpus_indices:      字符的索引(不同)
    :param idx_to_char:
    :param char_to_idx:
    :param is_random_iter:      数据是否随机采样
    :param num_epochs:          总轮数
    :param num_steps:
    :param lr:                  学习率
    :param clipping_theta:      梯度裁剪
    :param batch_size:          批量大小
    :param pred_period:         预测周期
    :param pred_len:
    :param prefixes:            需要预测的字符
    :return:
    '''

    if is_random_iter:
        # 一共vocab_size的大小
        # 每次返回batch_size * num_steps的大小，一共vocab_size/xx 次
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params(vocab_size, num_hiddens, vocab_size,ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如果使用相邻采样，开始时初始化隐藏状态
            state = init_rnn_state(batch_size,num_hiddens,ctx)
        l_sum,n,start = 0.0,0,time.time()
        data_iter = data_iter_fn(corpus_indices,batch_size,num_steps,ctx)
        for X,Y in data_iter:
            if is_random_iter:  # 随机采样，每个小批量开始前初始化状态.
                state = init_rnn_state(batch_size,num_hiddens,ctx)
            else:
                # 使用detach函数从计算图分离隐藏状态
                for s in state:
                    # 将某个node变成不需要梯度的Varibale。因此当反向传播经过这个node时，梯度就不会从这个node往前面传播
                    # 不想计算A网络的，那么可以把Y通过detach()函数分离出来
                    s.detach()
            with autograd.record():
                # inputs 是num_steps个(batch_size,vocab_size) = num_steps*batch_size*vocab_size
                inputs = to_onehot(X,vocab_size)
                # outputs 有num_steps 个形状为(batch_size,vocab_size)的矩阵
                (outputs,state) = rnn(inputs,state,params)
                # 连结之后形状为(num_steps*batch_size,vocab_size)的矩阵
                outputs = nd.concat(*outputs,dim=0)
                # 转置就是vocab_size,num_steps*batch_size
                # Y的形状时(batch_size,num_steps),转置变成长度时
                # batch_size * num_steps的向量，一一对应
                y = Y.T.reshape((-1,))  # 转换为一维矩阵
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs,y).mean()
            l.backward()
            grad_clipping(params,clipping_theta,ctx)    # 裁剪梯度
            d2l.sgd(params,lr,1)                        # 误差取过均值,梯度不做平均
            l_sum += l.asscalar() * y.size              # 平均损失*总数
            #
            n +=y.size
        if (epoch+1)%pred_period == 0:
            # perplexity
            print('epoch %d,perplexity %f,time %.2f sec' % (epoch+1,math.exp(l_sum/n),time.time()-start))
            for prefix in prefixes:
                print(' -' ,predict_rnn(prefix,pred_len,rnn,params,init_rnn_state,num_hiddens,vocab_size,ctx,idx_to_char,char_to_idx))

if __name__ == '__main__':
    corpus_indices,char_to_idx,idx_to_char,vocab_size = load_data_jay_lyrics()
    num_hiddens = 256
    # one-hot向量
    # 数据集是(批量大小，时间步数)
    ctx = try_gpu()
    print('will use',ctx)
    num_epochs,num_steps,batch_size,lr,clipping_theta = 500,35,32,1e2,1e-2
    pred_period,pred_len,prefixes = 50,50,['分开','不分开']
    data_iter = d2l.data_iter_random(corpus_indices, batch_size, num_steps, ctx)
    # train_and_predict_rnn(rnn,get_params,init_rnn_state,num_hiddens,vocab_size,ctx,corpus_indices,
    #                       idx_to_char,char_to_idx,True,num_epochs,num_steps,
    #                       lr,clipping_theta,batch_size,pred_period,pred_len,prefixes)




