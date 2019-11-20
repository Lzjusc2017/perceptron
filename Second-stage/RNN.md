

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

<font size='4'>X * W<sub>xh</sub>∈ </font>  R<sup>nxh</sup>
<font size='4'>H<sub>t-1</sub> * W<sub>xh</sub>∈ </font>  R<sup>nxh</sup>
<font size='4'>H<sub>t∈ </font>  R<sup>nxh</sup>

有上面的基础，我们可以书写
```python
	num_inputs = vocab_size		# 
```



