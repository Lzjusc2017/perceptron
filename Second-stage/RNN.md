

# 关于循环神经网络的理解

|适用文件|说明|
|--|--|
|rnn|简单的rnn网络|
|gru|门控循环单元|

<br>
<hr>
<br>


![rnn11_20.png](https://i.loli.net/2019/11/20/3LPNQqBrkVxDJvX.png)


输出层的输出和多层感知机中的计算类似:
<center ><font size='5'> O<sub>t</sub> = H<sub>t</sub>W<sub>hq</sub> + b<sub>q</sub></font></center>
<center ><font size='5'> H<sub>t</sub> =∮( X<sub>t</sub>W<sub>xh</sub> + H<sub>t-1</sub>W<sub>hh</sub> + b<sub>h</sub>)</font></center>


### 隐藏层权重
<font size='4'>W<sub>xh</sub> ∈ R<sup>dxh</sup> (h是隐藏单元个数)</font><br>
<font size='4'>W<sub>hh</sub> ∈ R<sup>dxh</sup></font><br>
<font size='4'>b<sub>h</sub> ∈ R<sup>1xh</sup></font>

### 输出层权重
<font size='4'>W<sub>hq</sub> ∈ R<sup>hxq</sup></font><br>
<font size='4'>b<sub>q</sub> ∈ R<sup>1xq</sup>(q是输出个数)</font>

这里，样本数是n，输入个数（特征数或特征向量维度）为d。
所以输入的小批量数据样本是<font size='4'>X ∈ R<sup>nxd</sup></font><br>

<font size='4'>X * W<sub>xh</sub>∈ R<sup>nxh</sup></font>
<font size='4'>H<sub>t-1</sub> * W<sub>xh</sub>∈  R<sup>nxh</sup></font>
<font size='4'>H<sub>t</sub> ∈ R<sup>nxh</sup> </font> 

有上面的基础，我们可以书写
```python
	num_inputs = vocab_size		 #  不同字符的个数，比如1027
	num_hiddens = 256
	num_outputs = vocab_size 	#	输出预测的个数，比如1027
	W_xh = nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens),ctx=ctx)
	W_hh = nd.random.normal(scale=0.01,shape=(num_hiddens,num_hiddens),ctx=ctx)
	b_h = nd.zeros(num_hiddens,ctx=ctx)
	# 输出层参数
	W_hq = nd.random.normal(scale=0.01,shape=(num_hiddens,num_outputs),ctx=ctx)
	b_q = nd.zeros(num_outputs,ctx=ctx)
	
	H = nd.tanh(nd.dot(X,W_xh) + nd.dot(H,W_hh) + b_h)
	Y = nd.dot(H,W_hq) + b_q
```

<br>

关于概率

例如，一段含有4个词的文本序列的概率

<center><font size='4'>P(w1,w2,w3,w4) = P(w1)*P(w2|w1)*P(w3|w1,w2)*P(w4|w1,w2,w3)</font></center>


为了计算语言模型，我们需要计算词的概率，以及一个词在给定前几个词的情况下的条件概率，即语言模型参数。
例如P(w1) 可以计算为w1在训练数据集中的词频(词出现的次数)与训练数据集的总词数之比。因此，根据条件概率定义，一个词在给定前几个词的情况下的条件概率也可以通过训练数据集中的相对词频来计算。例如，P(w2 | w1) 可以计算为w1和w2两词相邻的频率与w1词频的比值，也就是P(w1,w2)/P(w1)

<br>
<hr>
<br>

# 关于GRU

![批注 2019-11-21 171916.png](https://i.loli.net/2019/11/21/4i31vErQRd6DbMl.png)

<center ><font size='5'> R<sub>t</sub> =∮( X<sub>t</sub>W<sub>xr</sub> + H<sub>t-1</sub>W<sub>hr</sub> + b<sub>r</sub>)</font></center>
<center ><font size='5'> Z<sub>t</sub> =∮( X<sub>t</sub>W<sub>xz</sub> + H<sub>t-1</sub>W<sub>hz</sub> + b<sub>z</sub>)</font></center>
<center ><font size='5'> H<sup>~</sup><sub>t</sub> = ∮(X<sub>t</sub>W<sub>xh</sub> + (R<sub>t</sub>⊙H<sub>t-1</sub>)W<sub>hh</sub> + b<sub>h</sub>)</font></center>
<center ><font size='5'> H<sub>t</sub> = Z<sub>t</sub>⊙ H<sub>t-1</sub> + (1-Z<sub>t</sub>)⊙H<sup>~</sup><sub>t</sub></font></center>

### 关于参数
输入X ∈ (n,d)(样本数,输入个数)<br>
H<sub>t-1</sub> ∈ (n,h)<br>
W<sub>xr</sub>，W<sub>xz</sub> ∈ (d,h)<br>
W<sub>hr</sub>，W<sub>hz</sub> ∈ (h,h)<br>

有上面的基础，我们可以书写
```python
	def _one(shape):
        return nd.random.normal(scale = 0.01,shape=shape,ctx = ctx)
    # 隐藏层的参数
    def _three():
        return (_one((num_inputs,num_hiddens)),
                _one((num_hiddens,num_hiddens)),
                nd.zeros(num_hiddens,ctx=ctx))

    W_xz,W_hz,b_z = _three()
    W_xr,W_hr,b_r = _three()
    W_xh,W_hh,b_h = _three()
    # 输出层参数
    W_hq = _one((num_hiddens,num_outputs))
    b_q = nd.zeros(num_outputs,ctx=ctx)
	# 重置门，更新门
	Z = nd.sigmoid(nd.dot(X,W_xz) + nd.dot(H,W_hr) + b_z)
	R = nd.sigmoid(nd.dot(X,W_xr) + nd.dot(H,W_hr) + b_r)
	H_tilda = nd.tanh(nd.dot(X,W_xh) + nd.dot(R*H,W_hh)+ b_h)
	H = Z*H + (1-Z)*H_tilda
	Y = nd.dot(H,W_hq) + b_q
```

<br>

我们对门控循环单元的设计稍作总结：<br>
 - 重置门有助于捕捉时间序列里短期的依赖关系
 - 更新门有助于捕捉时间序列里长期的依赖关系

<br>


# 参考
[RNN](https://blog.csdn.net/zhaojc1995/article/details/80572098)  <br>

[GRU](http://zh.gluon.ai/chapter_recurrent-neural-networks/gru.html)





