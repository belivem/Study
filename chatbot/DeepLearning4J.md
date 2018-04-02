#DeepLearning4J

## 1, 神经网络基础知识
神经网络有五大超参数，而这些超参数不能通过常规方法获得。其具有的五大超参数如下：1，学习率。2，权值初始化。3，网络层数。4，单层神经单元数。5，正则惩罚项。**SVM支持向量机通过巧妙的变换目标函数，避免神经网络的大部分超参数，尤其是自适应的支持向量替代人工设置神经元，使得SVM避免过拟合**。

## 1.1 Updaters ==> 学习率
**1. SGD[随机梯度，一阶方法]**

	sgd为每一次迭代计算一次梯度，然后再对梯度进行更新，是一种非常常见的做法。其计算公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;-\eta&space;g_{t}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;-\eta&space;g_{t}" title="x_{t+1} = x_{t}+\Delta x_{t}; \Delta x_{t} = -\eta g_{t}" /></a></div>
	
其中 x_{t} 代表任一神经单元对应的t时刻权值，\Delta x_{t}代表权值的增量， \eta 代表学习率，g_{t}代表t时刻的梯度。

	缺点如下：
	
		1. 选择合适的learning rate比较困难
		2. 对所有的参数使用同一个learning rate,比如对于不常见的特征或者稀疏的特征可能想更新快一点，减少时间，就可以设置较大的learning rate.
		3. sgd算法容易局部最优，这就需要多次设置参数的初始值。
		4. sgd算法的更新方向完全依赖于当前的batch，使得其十分不稳定。

**2. 牛顿法[二阶方法]**
牛顿法是一个自适应算法，使用Hessian矩阵代替人工设置的学习率，在梯度下降时可以完美的找出下降的方法，同时也是一种相对理想的方法，其计算方法如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;H_{t}^{-1}g_{(t)}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;H_{t}^{-1}g_{(t)}" title="x_{t+1} = x_{t}+\Delta x_{t}; \Delta x_{t} = H_{t}^{-1}g_{(t)}" /></a></div>

优点：采用牛顿法可以较为完美的找出下降的方向，不会陷入局部极小值?

缺陷：1，求Hessian的逆矩阵需要花费大量的计算资源，时间较长、代价较高。2，不适用于大数据。

**3. 动量法**
为解决sgd算法缺陷4，引入一个动量，也就成为了动量法。其模拟的是物体运动的惯性 ==> 更新时在一定程度上保留了之前更新的方向，同时利用当前的梯度微调最终的更新方向。如此可在一定程度上增加稳定性，也有利于摆脱局部最小值的影响。其公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=X_{t&plus;1}&space;=&space;X_{t}&plus;\Delta&space;X_{t}&space;;&space;\Delta&space;X_{t}&space;=&space;\rho&space;\Delta&space;X_{t&space;-&space;1}-\eta&space;g_{t}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{t&plus;1}&space;=&space;X_{t}&plus;\Delta&space;X_{t}&space;;&space;\Delta&space;X_{t}&space;=&space;\rho&space;\Delta&space;X_{t&space;-&space;1}-\eta&space;g_{t}" title="X_{t+1} = X_{t}+\Delta X_{t} ; \Delta X_{t} = \rho \Delta X_{t - 1}-\eta g_{t}" /></a></div>

其中，公式中的ρ即为动量，表示在多大程度上保留原来的更新方向。ρ和η之和不一定为1.

优点：
	1， 相对于SGD算法，其更稳定。
	2， 在一定程度上减少了局部最小值出现的可能性。

缺点：
	1， 需要增加设置一个参数--动量，而参数的选择仍然是一个问题。
	2， 仍然存在着局部最小值的隐忧。

**4. Adagrad**
	
Adagrad方法类似与正则化L2[不同之处一个在于权值，一个在于梯度]，可以解决SGD的缺陷2，使得对于不常见的参数更新的步数较大，而对于常见的参数更新的步数较小，*无需手动调节学习率*。其公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;-\frac{\eta&space;}{\sqrt{\sum_{t&space;=&space;1}^{t}g_{t}^{2}}}g_{t}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;-\frac{\eta&space;}{\sqrt{\sum_{t&space;=&space;1}^{t}g_{t}^{2}}}g_{t}" title="x_{t+1} = x_{t}+\Delta x_{t}; \Delta x_{t} = -\frac{\eta}{\sqrt{\sum_{t = 1}^{t}g_{t}^{2}}}g_{t}" /></a></div>
	
Adagrad算法，常见的特征都有值，故而其梯度的平方基本上也都为正值，并且比较平稳，很少出现接近0的数。所以其T时刻以前的梯度平方和是单调递增的，而对于不常出现的特征其梯度很有可能接近于0，故而其T时刻之前的梯度平方和一般情况下要小于前者的梯度平方和，所以根据 梯度平方和越大其学习率就越小，反之越大 ==> 对于不常见的参数更新的步数较大，而对于常见的参数更新的步数较小。
	
特点：1，训练前期，梯度较大，放大梯度[激励阶段]。训练后期，梯度较小，缩小梯度[惩罚阶段]。
	
	优点:1，自适应学习算法，只需设置初始的学习率。2，对于每个参数，随着其更新总距离的增多，其学习率逐渐降低。
	
	缺陷:1，公式中分母会累加梯度平方，而在训练中持续增大的话，会使学习率非常小，从而趋于无限小，从而出现梯度消失问题。Adagrad算法其学习率是单调递减的。2，仍需手工设置初始学习率。

**5. Adadelta**
	
Adadelta是对Adagrad的一个扩展，其目的在于采用一阶的方法，近似模拟二阶的牛顿法。Adagrad会累加之前所有的梯度平方，但是Adadelta只会累加固定大小的项，仅仅计算固定大小项的[平均值]。其公式如下：	
1, Adadelta采用了类似动量的平均方法

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=E[g_{t}^{2}]&space;=&space;\rho&space;E[g_{t-1}^{2}]&plus;(1-\rho)g_{t}^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?E[g_{t}^{2}]&space;=&space;\rho&space;E[g_{t-1}^{2}]&plus;(1-\rho)g_{t}^{2}" title="E[g_{t}^{2}] = \rho E[g_{t-1}^{2}]+(1-\rho)g_{t}^{2}" /></a></div>

2, Adadelta公式如下：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=X_{t&plus;1}&space;=&space;X_{t}&plus;\Delta&space;X_{t};&space;\Delta&space;X_{t}&space;=&space;-&space;\frac{&space;\sqrt{E[\Delta&space;X_{t-1}^{2}]}}{&space;\sqrt{E[g_{t}^{2}]}}&space;g_{t};" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{t&plus;1}&space;=&space;X_{t}&plus;\Delta&space;X_{t};&space;\Delta&space;X_{t}&space;=&space;-&space;\frac{&space;\sqrt{E[\Delta&space;X_{t-1}^{2}]}}{&space;\sqrt{E[g_{t}^{2}]}}&space;g_{t};" title="X_{t+1} = X_{t}+\Delta X_{t}; \Delta X_{t} = - \frac{ \sqrt{E[\Delta X_{t-1}^{2}]}}{ \sqrt{E[g_{t}^{2}]}} g_{t};" /></a></div>

优点：
	1， 完全抛弃了手动设置学习率。
缺点：
	1， 仍然无法摆脱局部最小值的约束。

**6. RmsProp**

	类似于Adadelta。

**7. Adam**
	
	类似于Adadelta。
	
总结：对于稀疏特征，尽量使用可自适应的优化方法，而不去手动调节。通常情况下，SGD采用了手动设置学习率的方式，训练时间会长一点。如果在意 更快的收敛，并且需要训练较深的复杂网络时，推荐采用自适应的方法。而Adadelta、RmsProp和Adam都是比较相近的算法，表现差不多。

各种updater算法结果图如下:

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/FocderlHxjXwR9KsrXg4rkEDaNaWV3zIgRaikVsDIpg!/b/dEIBAAAAAAAA&bo=SgPZAQAAAAADB7M!&rf=viewer_4" width="800" height="400" alt="updater算法"/>
</div>


## 1.2 损失函数
损失函数是模型对于数据拟合程度的度量，损失函数值越大，那么模型的拟合性越差。同时我们还期望，在损失函数值较大时，模型的梯度也能够较大，这样模型变量也就会更新的越快 -- 这个在一定程度上是合理存在的，但是并不现实，因为梯度代表了变化程度。

	1.	回归任务通常选择MSE、MEAN_ABSOLUTE_ERROR等。
	2.	分类任务通常选择MCXENT、NEGATIVELOGLIKELIHOOD。

**1. MSE[最小平方误差]**

最小平方误差是一个非常常见的损失函数度量，经常用于线性回归中，其中BP算法也经常用到。其公式如下：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=L[G(X)&space;-&space;Y]&space;=&space;\sum_{i}^{n}[G(x_{i})&space;-&space;y_{i}]^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L[G(X)&space;-&space;Y]&space;=&space;\sum_{i}^{n}[G(x_{i})&space;-&space;y_{i}]^{2}" title="L[G(X) - Y] = \sum_{i}^{n}[G(x_{i}) - y_{i}]^{2}" /></a></div>

优点:
	1, 简单有效

缺点：
	1, 经常应用于回归预测中，而对于分类输出多为概率的情况不适用。
	2，MSE通常不能用Sigmoid系的激活函数，因为Sigmoid激活函数在图像两端非常平缓，易于出现梯度消失的情况，而MSE函数无法处理梯度消失问题。

**2. MCXENT[交叉熵损失函数]**

交叉熵是一个非常神奇的应用，其目的在于 使用预测分布Q来表示样本的真实分布P的平均编码长度。其公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=H(P,Q)&space;=&space;H(P)&plus;D(P||Q)&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)log(P(x))&plus;\sum_{x\subset&space;X}^{X}P(x)log\frac{P(x)}{Q(x)}&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)logQ(x)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?CEH(P,Q)&space;=&space;H(P)&plus;D(P||Q)&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)log(P(x))&plus;\sum_{x\subset&space;X}^{X}P(x)log\frac{P(x)}{Q(x)}&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)logQ(x)" title="CEH(P,Q) = H(P)+D(P||Q) = -\sum_{x\subset X}^{X}P(x)log(P(x))+\sum_{x\subset X}^{X}P(x)log\frac{P(x)}{Q(x)} = -\sum_{x\subset X}^{X}P(x)logQ(x)" /></a></div>

其中H(P)代表真实分布P的熵 ==> 衡量一个样本所需的编码长度的期望。D(P||Q)代表KL距离也称之为相对熵 ==> 衡量相同时间空间内两个概率分布的差异。

交叉熵损失函数 是一个非常基本的损失函数，主要用于逻辑回归和softmax分类中。其基本公式(以逻辑回归为例):
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=L(Y,H(X))&space;=&space;-\frac{1}{m}\sum_{i=1}^{m}[y_{i}log(h(x_{i}))&plus;(1-y_{i})log(1-h(x_{i}))]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L(Y,H(X))&space;=&space;-\frac{1}{m}\sum_{i=1}^{m}[y_{i}log(h(x_{i}))&plus;(1-y_{i})log(1-h(x_{i}))]" title="L(Y,H(X)) = -\frac{1}{m}\sum_{i=1}^{m}[y_{i}log(h(x_{i}))+(1-y_{i})log(1-h(x_{i}))]" /></a></div>


## 1.3 Activation ==> 激活函数

神经网络中激活函数的作用是能够给神经网络提供一些非线性因素，使得神经网络可以更好的解决相对复杂的问题。

**Sigmod函数:**

sigmod函数一个单调并且相对稳定的函数。其计算公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}" title="f(x) = \frac{1}{1+e^{-x}}" /></a></div>

图形为：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/4v0U4SMl01uakhTIDLyUNaiMiKanpVvsjm3yzzt9tNA!/b/dEEBAAAAAAAA&bo=kgHUAAAAAAADF3U!&rf=viewer_4" width="400" height="200" alt="tanh"/>
</div>

优点：

	1， sigmod函数的输出连续且单调递增，并且值域为(0,1),是一个相对稳定的函数，主要用于输出层。
	2， sigmod函数求导容易。
缺点：

	1， 属于软饱和函数，容易产生梯度消失。
	2， 不是关于0点对称。

**tanh函数:**
tanh函数公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tanh(x)&space;=&space;\frac{1&space;-&space;e^{-2x}}{1&space;&plus;&space;e^{-2x}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?tanh(x)&space;=&space;\frac{1&space;-&space;e^{-2x}}{1&space;&plus;&space;e^{-2x}}" title="tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}" /></a></div>

图形如下：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/2UizcooplnjbHvdkpsFIhHkBc5PqRe21R8jVdnl32os!/b/dPIAAAAAAAAA&bo=WwHPAAAAAAADB7c!&rf=viewer_4" width="400" height="200" alt="tanh"/>
</div>

优点：

	1， 单调递增，相对于sigmod函数，其收敛速度较快。
	2， tanh函数关于0点对称。
缺点：

	1， 由于饱和性产生的梯度消失问题。

**RELU类激活函数:**

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/pWaib0xVAgHqBayGWllXnf41OHscBiZpYWM4nDuaKhw!/b/dFYBAAAAAAAA&bo=PAIYAQAAAAADBwU!&rf=viewer_4" width="400" height="200" alt="RELU"/>
</div>

RELU函数：
RELU函数是近年来较为流行的函数，其公式如下：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/lHSu7JCOiaAD5ugjC3RVus6nleWHKmZE8uRKYMCjgb4!/b/dEQBAAAAAAAA&bo=pQBDAAAAAAADF9Q!&rf=viewer_4" width="150" height="70" alt="RELU"/>
</div>
图形如下:
<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/s45u97QRyu0cpuO7XBfNckiZ.l8Rp1V1*77dR7sk07g!/b/dFYBAAAAAAAA&bo=XgHMAAAAAAADF6E!&rf=viewer_4" width="400" height="200" alt="RELU"/>
</div>	

优点：

	1， 相比于sigmod和tanh，RELU能够收敛的更快，因为它是线性并且非饱和的函数，并且计算速度更快。
	2， 有效的环节了梯度消失的问题。
缺点：

	1， 随着训练的进行，RELU可能出现神经元死亡的现象，并且流经神经元的梯度将永远是0，不可逆，故而RELU函数经常用于CNN的训练，因为图像的像素点基本上都不为0.

ELU函数：
ERU函数被定义为：	
优点：

	1， 在负的限制条件下能够更有鲁棒性。
	2， 计算也相对容易，导数与sigmod类似。


## 2, 循环神经网络
### 2.1， 循环神经网络简介
循环神经网络已经在自然语言处理中取得了巨大成功以及广泛的应用。并且循环神经网络中目前使用的最广泛为LSTM(长短时记忆网络)。

RNNs包含输入单元、隐藏单元和输出单元，其中 输出单元 输出的数据可以回流至隐藏单元[称之为"Back Projections"]，同时隐藏单元中的各个神经单元可以自连也可以相互连接。

下图为“循环神经网络”展开成一个全连接神经网络：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/ZR3f0FltbHRXwYGFkm9W.wT7C.fAxHwnJANBj34M6ag!/b/dF4BAAAAAAAA&bo=RANGAQAAAAADByI!&rf=viewer_4" width="650" height="300" alt="循环神经网络"/>
</div>

RNNs可以展开为一个全连接神经网络(多个神经网络叠加)，但是多个神经网络中所用的参数则是相同的，例如 给定一个字符串序列[X1,X2,X3]预测其情感倾向性,其中X1,X2和X3分别代表一个神经网络的输入，其每个输入指代的神经网络参数U/W/V都是相同的，并且输出O1,O2 可以没有必要存在(因为输入X3只需要输出倾向性即可)。


**RNNs的作用：**

1，语言模型和文本生成。给定语句中前面一个词，预测后一个词的可能性。[Generating Text with Recurrent Neural Networks]

2，机器翻译。给定一个字符串序列，输出其他相似含义的其他语言。[A Recursive Recurrent Neural Network for Statistical Machine Translation]

3，语音识别。[Towards End-to-End Speech Recognition with Recurrent Neural Networks]

4，图像描述生成。给定一个图像，其自动生成图像描述。[Towards End-to-End Speech Recognition with Recurrent Neural Networks]

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/D5fPVUAAJiBH5SkGC3JyoqkNiJ6CKqm.xUI7QtcakuI!/b/dGcBAAAAAAAA&bo=4AInAQAAAAADB.Y!&rf=viewer_4" width="650" height="300" alt="循环神经网络"/>
</div>

上图中的5个例子从左到右分别是：

	1.	没有使用RNN，从固定大小的输入得到固定大小输出（比如图像分类）
	2.	序列输出（比如图片描述，输入一张图片输出一段文字序列）
	3.	序列输入（比如情感分析，输入一段文字然后将它分类成积极或者消极情感）
	4.	序列输入和序列输出（比如机器翻译）
	5.	同步序列输入输出（比如视频分类，对视频中每一帧打标签）

**RNNs的训练：**

RNNs与传统的神经网络训练一样，都是使用BP误差反向传播算法。但是在使用梯度下降算法时，每一步的输出不仅依赖于当前步的网络，同时也依赖于前面若干步网络的状态，此种BP误差反向传播算法称之为 BPTT算法。

但是BPTT算法有一个非常大的局限性--其无法解决长时依赖问题[即当前的输出与前面很长一段序列有关，一般不能超过10步]，因为BPTT算法会带来*梯度消失或者爆炸问题*，当然LSTM则可以应对这种问题。

**RNNs的扩展和改进模型：**

**Simple RNNs:**

Simple RNNs是RNNs的一种特例，是一个三层网络，但是其在隐藏层增加了上下文单元，每一个隐藏层单元与一个上下文节点一一对应，并且任一上下文节点和其对应的隐藏层节点间的权值也是固定不变的。而上下文的每一个节点保存其连接的隐藏层节点的上一步输出，即保存上文，并作用于当前步对应的隐藏层节点状态。其图如下所示：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/SlB55ULS7PkUrLCPk2G85vzQ4HcL5YjstIW8Q8DqrtI!/b/dF4BAAAAAAAA&bo=CAJEAgAAAAADB24!&rf=viewer_4" width="350" height="400" alt="马尔科夫链"/>
</div>

如上图所示，y1,y2..等属于隐藏层的神经单元，而u1,u2..等属于上下文单元节点，保存其对应的上一步隐藏单元的值。

**Deep RNNs:**

已知语句的上下文信息，预测一个语句中缺失的词语就可以使用Deep RNNs，原因在于Deep RNNs既需要使用前一步隐层的输出数据同时也需要后一步隐层的输出数据。其结构图如下所示--双向循环：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/A.B*5wzJNJbOKwFsh5j9EOelXd29s436cqvg7cC3Wak!/b/dGEBAAAAAAAA&bo=KwE5AQAAAAADBzA!&rf=viewer_4" width="350" height="400" alt="马尔科夫链"/>
</div>

**Gated Recurrent RNNs:**

### 2.2 长短时记忆网络

LSTM与Gated RNNs非常相似，不同之处就在于 隐藏单元 中的结构相对复杂。例如，如下为一隐藏单元的结构：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/ifflaG3o1sDWtt3qgrJYCfDdJf7ipMBG4RsfObp02aQ!/b/dHMAAAAAAAAA&bo=4AIxAQAAAAADB*A!&rf=viewer_4" width="600" height="300" alt="马尔科夫链"/>
</div>

LSTM中最重要的核心思想就是cell state,如下如所示：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/wh.fn2oi7.RjD5sN8QVoS4Wq0WM13vqmTUBKakULUE0!/b/dFYBAAAAAAAA&bo=qwEKAQAAAAADB4M!&rf=viewer_4" width="400" height="300" alt="马尔科夫链"/>
</div>

上图中的flow即承担着之前所有状态的信息，每当flow流经一个重复结构A的时候，都会由相应的操作来决定舍弃什么旧的信息和增加什么新的信息。 LSTM中含有三个门结构，分别是“遗忘门”、“输入门”和“输出门”。

遗忘门：

遗忘门决定了cell state中舍弃的信息，其输入为 上一状态的输出、当前状态的输入。

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/247ndKtN2hN7oXrhGHl*.OSULeR2hxYsVdvuC6mr9a8!/b/dFYBAAAAAAAA&bo=vgLuAAAAAAADB3A!&rf=viewer_4" width="400" height="200" alt="遗忘门"/>
</div>

输入门：

输入门决定了要往cell state中保存什么新的信息，其输入为 上一状态的输出、当前的输入。

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/XzGeTnr80wj7ZfWAmZw1ckqglGXknb4MND.2hPr7g0U!/b/dGcBAAAAAAAA&bo=2QIcAQAAAAADF*Q!&rf=viewer_4" width="400" height="200" alt="输入门"/>
</div>

输出门：

输出门决定了本结构A要向外界输出的信息，其输入为上一状态的输出、当前状态的输入和当前结构的Cell state信息。

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/n42BwvFJhAlIhQ.xO9OaRf999UV5q9pI7uhKJpM9UMQ!/b/dAgBAAAAAAAA&bo=wAL*AAAAAAADFw8!&rf=viewer_4" width="400" height="200" alt="输出门"/>
</div>

LSTM增加网络层数：

多个独立的LSTM网络进行叠加，构造多层循环神经网络。如图:

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/idQiz7ZOekU*wvxhsjLofEJHTZcKIoHtArELCbjwyU4!/b/dPMAAAAAAAAA&bo=5AJEAQAAAAADB4E!&rf=viewer_4" width="600" height="300" alt="输出门"/>
</div>

**LSTM变体 **

1， peephole connection：LSTM网络的一种变体，即将当前的cell state也作为输入传递给Sigmod函数，如下图

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/.7x0BfWRy*W56SaxZuky8WhxnH5dy1oa6VXFeOdEH3E!/b/dFsBAAAAAAAA&bo=1QIcAQAAAAADB.g!&rf=viewer_4" width="600" height="300" alt="输出门"/>
</div>

2，Gated Recurrent Unit (GRU): 目前是一种LSTM网络的一个变体，将cell state和hidden state进行合并，将原来的三个门更换为“更新门”和“重置门”

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/p*xUnevwcxF2Afb24JC7EOTriI.n5c7fxVvST6xLG2I!/b/dF4BAAAAAAAA&bo=wgIjAQAAAAADB8A!&rf=viewer_4" width="600" height="300" alt="输出门"/>
</div>

# 引用
[1, 循环神经网络介绍] (http://blog.csdn.net/heyongluoyao8/article/details/48636251)
[2, RNN & LSTM 网络结构及应用] (https://www.jianshu.com/p/f3bde26febed)
[3, 自适应学习率调整] (http://www.cnblogs.com/neopenx/p/4768388.html)
[4, 各种优化算法比较] (http://blog.csdn.net/luo123n/article/details/48239963)







