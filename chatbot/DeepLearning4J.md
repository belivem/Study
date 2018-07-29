#DeepLearing 

##0, 深度学习相关

1, 深度学习解决的核心问题之一就是自动的将简单的特征组合成更加复杂的特征，并且使用这些特征组合解决相关问题。
2，模拟人类大脑不再是深度学习研究的主导方向，我们此时不应该再认为深度学习是在试图模仿人类大脑。
3, 深度学习两个重要特征：非线性和多层。

##1, 神经网络基础

机器学习算法的常用优化方式：1>牛顿法。2>梯度下降法。3>最小二乘法等，其中神经网络就隶属于**梯度下降法**这个分支。

###1.1, 激活函数

神经网络有五大超参数，而这些超参数不能通过常规方法获得。其具有的五大超参数如下：1，学习率。2，权值初始化。3，网络层数。4，单层神经单元数。5，正则惩罚项。SVM支持向量机通过巧妙的变换目标函数，避免神经网络的大部分超参数，尤其是自适应的支持向量替代人工设置神经元，使得SVM避免过拟合。

<div align=center>
<img src="http://w3.huawei.com/t/data/uploads/miniblog/2018/0506/dc05cf2babeae009e6b468dd4af1fba2_middle.jpg" width="550" height="350" alt="Activate Functions"/>
</div>

一般情况下，在卷积神经网络中，推荐的激励函数是relu; 而在循环神经网络(RNN)中，我们推荐的激励函数是tanh或者relu。激励函数运行时激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经系统。激励函数的实质是非线性方程。

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


**softplus 函数**

**softmax函数**

softmax更多的用于分类，是sigmod的一般化扩展，在tensorflow中也作为一个激活函数使用，往往作为神经网络最后输出的输入(向量[1.2,3.1,0.1])。其公式为:

<a href="http://www.codecogs.com/eqnedit.php?latex=softmax(X)&space;=&space;P(y=j|x)&space;=&space;\frac{e^{w_{j}X}}{\sum_{k=1}^{K}e^{w_{k}X}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?softmax(X)&space;=&space;P(y=j|x)&space;=&space;\frac{e^{w_{j}X}}{\sum_{k=1}^{K}e^{w_{k}X}}" title="softmax(X) = P(y=j|x) = \frac{e^{w_{j}X}}{\sum_{k=1}^{K}e^{w_{k}X}}" /></a>

输出也为一个向量[0.1,0.2,0.4],取概率最大值的索引即为输入X所对应的输出类别。Tensorflow最后使用的softmax函数并没有分类器的功能，而仅仅作为一个归一化的工具，其图像为：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/ucT9nkH1Tugb8cp2E6a.wDzKLX0c5dQ9HERunTEM6FY!/b/dDIBAAAAAAAA&bo=oAU4BAAAAAABF6k!&rf=viewer_4" width="400" height="350" alt="RELU"/>
</div>


###1.2 学习率 -- 加速训练过程

**1. SGD[随机梯度，一阶方法]**

sgd为每一次迭代计算一次梯度，然后再对梯度进行更新，是一种非常常见的做法。其计算公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;-\eta&space;g_{t}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;-\eta&space;g_{t}" title="x_{t+1} = x_{t}+\Delta x_{t}; \Delta x_{t} = -\eta g_{t}" /></a></div>
	
其中x_{t}代表任一神经单元对应的t时刻权值，\Delta x_{t}代表权值的增量， \eta 代表学习率，g_{t}代表t时刻的梯度。

	缺点如下：	
		1. 选择合适的learning rate比较困难
		2. 对所有的参数使用同一个learning rate,比如对于不常见的特征或者稀疏的特征可能想更新快一点，减少时间，就可以设置较大的learning rate.
		3. sgd算法容易局部最优，这就需要多次设置参数的初始值。
		4. sgd算法的更新方向完全依赖于当前的batch，使得其十分不稳定。
	
**2. 牛顿法[二阶方法]**

牛顿法是一个自适应算法，使用Hessian矩阵代替人工设置的学习率，在梯度下降时可以完美的找出下降的方法，同时也是一种相对理想的方法，其计算方法如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;H_{t}^{-1}g_{(t)}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;x_{t}&plus;\Delta&space;x_{t};&space;\Delta&space;x_{t}&space;=&space;H_{t}^{-1}g_{(t)}" title="x_{t+1} = x_{t}+\Delta x_{t}; \Delta x_{t} = H_{t}^{-1}g_{(t)}" /></a></div>

	优点：	
		采用牛顿法可以较为完美的找出下降的方向，不会陷入局部极小值?

	缺陷：
		1，求Hessian的逆矩阵需要花费大量的计算资源，时间较长、代价较高。
		2，不适用于大数据。


**3. 动量法**

为解决sgd算法缺陷4，引入一个动量，也就成为了动量法[类似于滑动平均模型，不一致的在于滑动平均模型主要用于偏置值和权重参数]。其模拟的是物体运动的惯性 ==> 更新时在一定程度上保留了之前更新的方向，同时利用当前的梯度微调最终的更新方向。如此可在一定程度上增加稳定性，也有利于摆脱局部最小值的影响。其公式如下：

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
	
	优点:
		1，自适应学习算法，只需设置初始的学习率。
		2，对于每个参数，随着其更新总距离的增多，其学习率逐渐降低。
	
	缺陷:
		1，公式中分母会累加梯度平方，而在训练中持续增大的话，会使学习率非常小，从而趋于无限小，从而出现梯度消失问题。Adagrad算法其学习率是单调递减的。
		2，仍需手工设置初始学习率。

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

###1.3, 防止过拟合方法
大规模神经网络的两大缺陷：1，费时。2，容易过拟合。

####1.3.1, L1和L2正则化

**L1正则化**
	L1正则化使得参数变得更为稀疏，会有更多的参数变为0，从而达到类似特征选取的功能，但是L1正则化求导艰难
**L2正则化**
	L2正则化有利于求导

常见的损失函数有Zero-one Loss（0-1损失），Perceptron Loss（感知损失），Hinge Loss（Hinge损失），Log Loss（Log损失），Cross Entropy（交叉熵），Square Loss（平方误差），Absolute Loss（绝对误差），Exponential Loss（指数误差)等。

损失函数公式：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\sum_{i=1}^{n}l(y_{i},f(x_{i}))&plus;\lambda&space;R(w)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L&space;=&space;\sum_{i=1}^{n}l(y_{i},f(x_{i}))&plus;\lambda&space;R(w)" title="L = \sum_{i=1}^{n}l(y_{i},f(x_{i}))+\lambda R(w)" /></a></div>

公式前半部分为经验风险，整体为结构风险，其中R(w)为正则项，常见为L1和L2正则项。


####1.3.2, dropOut

DropOut方法指训练模型时，随机让网络的某些**隐藏层节点**的权重不工作，不工作的那些节点可暂时认为不是网络结构的一部分，但是其对应的权重需要保留下来(因为下次可能就又工作了)。

在标准神经网络中，每个参数接收的导数表明其应该如何变化才能使最终损失函数降低，并给定所有其它神经网络单元的状态。因此神经单元可能以一种可以修正其它神经网络单元的错误的方式进行改变。而这就可能导致复杂的共适应(co-adaptations)。由于这些共适应现象没有推广到未见的数据，将导致过拟合。我们假设对每个隐藏层的神经网络单元，Dropout通过使其它隐藏层神经网络单元不可靠从而阻止了共适应的发生。因此，一个隐藏层神经元不能依赖其它特定神经元去纠正其错误。

DropOut核心思想:通过阻止特征检测器的共同作用来提高神经网络的性能。理由如下：

	1. 由于每次用输入网络的样本进行权值更新时，隐含节点都是以一定概率随机出现，因此不能保证每2个隐含节点每次都同时出现，这样权值的更新不再依赖于有固定关系隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况。
	2. 可以将dropout看作是模型平均的一种。对于每次输入到网络中的样本（可能是一个样本，也可能是一个batch的样本），其对应的网络结构都是不同的，但所有的这些不同的网络结构又同时share隐含节点的权值。这样不同的样本就对应不同的模型，是bagging的一种极端情况。

DropOut步骤：

	1. 随机删除网络中一半的隐藏神经元，而输入输出神经元保持不变[保存删除神经元的权值]。
	2. 前向传播输入值，后梯度下降法更新有效神经元的权值。
	3. DropOut的单位为batch，恢复被删除单元的权值[重新合入网络]，重复1，2步

DropOut方法与滑动平均模型方法有异曲同工之妙，不同的是滑动平均模型在于平均化参数，而DropOut方法在于平均化网络。

问题一: dropOut方法为何只用于隐藏层?

答：1，dropOut不能用于输入层：因为输入层使用dropout时，可能直接将相对重要的输入特征做删除，影响最终的结果。2, dropOut不能用于输出层：因为dropOut是随机的，可能会将正确的结果分类节点删除。3, dropOut也是可以用于卷积层，但是首先从作用上来看，卷积层的目的在于特征提取而不是特征综合，故而效果可能不大。而对于卷积层强制使用dropOut相当于增加了噪声，对于小卷积核[卷积核不应该是节点，而应该存放参数，而卷积之后的输出可以看做是节点]不推荐。

###1.4, 提高准确度

####1.4.1 局部响应归一化(LRN)
局部响应归一化技术 是一种提高深度学习准确率的一种方法，一般用于激活、池化之后。技术首次提出AlexNet模型。

LRN基于如下的动机：
	
	被激活的神经元抑制其相邻的神经元，局部归一化即抑制其局部的相邻神经元，实现“强者愈强”。

AlexNet模型是一种创新性的应用，其主要新技术点如下：

	1. 成功使用ReLU作为CNN的激活函数，并验证其效果在较深的网络超过了Sigmoid，成功解决了Sigmoid在网络较深时的梯度弥散问题。虽然ReLU激活函数在很久之前就被提出了，但是直到AlexNet的出现才将其发扬光大。
	
	2. 训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。Dropout虽有单独的论文论述，但是AlexNet将其实用化，通过实践证实了它的效果。在AlexNet中主要是最后几个全连接层使用了Dropout。
	
	3. 在CNN中使用重叠的最大池化。此前CNN中普遍使用平均池化，AlexNet全部使用最大池化，避免平均池化的模糊化效果。并且AlexNet中提出让步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性。

	4. 提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

LRN公式：

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/Bjw47zCsKyEAO5fm78F3hYlSd5FJJrKMWBYJQorHqtE!/b/dDMBAAAAAAAA&bo=.gLzAAAAAAADByk!&rf=viewer_4" width="800" height="150" alt="LRN公式"/>
</div>

如上述公式所示,alpha,beta,n/2,k(通常情况下，其表示偏移量)都为需人为设定，其表达的含义是当前像素点的输出值与当前像素点和其前n/2，后n/2维度相同位置上的像素点平方和相关。

##2, 循环神经网络


##3, 卷积神经网络

卷积神经网络(CNN)是一种新型的网络架构方式，主要有卷积层、池化层和全连接层组成。通常意义上讲：卷积层的主要功能在于特征提取，池化层的主要功能在于特征选择。

###3.1, 基本概念

####卷积层

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/D*U.W1YpzS8L8NvXZgQrBWSBAtms2pofNKhQcFgdW28!/b/dAgBAAAAAAAA&bo=zAIMAQAAAAADB.E!&rf=viewer_4" width="500" height="200" alt="卷积层"/>
</div>
	
卷积层的作用在于特征提取功能，常用的卷积层大小一般为3-3，5-5等，卷积层可有多个权重矩阵组成。 

**权重矩阵的个数称之为卷积层的深度， 每个权重矩阵同时处理的矩阵深度和当前层神经网络输入矩阵的深度是一致的，可看做提取一组同维度的特征，PS:卷积层也可提取不同维度的特征，详见Inception-v3模型。**

由此可见，一个卷积层可设置参数如下：

1. 权重矩阵的边长
2. 权重矩阵的深度
3. 卷积层的步数
4. 是否采用全“0”填充[仅适用于计算]
	
卷积层的前向传播过程就是通过将一个过滤器从神经网络当前层的左上角移动至右下角。

卷积层参数个数： 输入层矩阵32-32-3, 第一层卷积层尺寸5-5，深度16的过滤器，其参数个数为5-5-3-16+16个参数【-代表相乘,其中5-5-3可看做是**一个神经元**的权值矩阵】。**由此可知，卷积层的参数个数与图片的大小无关，而只与卷积层的尺寸、深度和输入矩阵的深度有关。**

通常来讲，卷机网络在过滤器的深度上普遍采用逐层递增的方式，通常下一层的卷积层深度旺旺是上一层卷积层深度的2倍。



####池化层

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/Inx17J.L7BQw8o1FaJKkWyh.YksKX6Jc5YFz6J8RTms!/b/dDIBAAAAAAAA&bo=xwIHAQAAAAADF*E!&rf=viewer_4" width="500" height="200" alt="池化层"/>
</div>

池化层主要分为两种:"max_pool()"和"mean_pool()"。其作用在于大规模的缩小输入矩阵的大小，加快计算速度，并且池化层还具有防止过拟合的作用[根据值的大小，将一些不是很关键的特征去掉，仅留下作用比较大的特征]。

池化层与卷积层有着一个重要的区别：卷积层使用的过滤器是横跨整个深度的，而池化层使用的过滤器只影响一个深度的节点。池化层可设置参数如下：

1. 池化层过滤器尺寸。
2. 池化层步长。
3. 是否采用全“0”填充[仅适用于计算]

池化层举例：

	pool = tf.nn.max_pool(activate_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
	
其中ksize代表池化层过滤器的尺寸，而strides代表步长，其中第一维度为1代表池化层不可以跨越不同的输入样例，而第四维度为1代表池化层不可以跨越不同的节点矩阵深度。

池化层的作用在于减少输入维度，提取较明显的特征。但是池化层并不是必须的，因为若改变卷积层的步长，也就相当于减少输入维度。

####Padding

Tensorflow中的padding操作，主要分为两种"VALID"和"SAME"，其中"VALID"代表不进行全0填充，而 "SAME"代表全0填充。

引入全0填充原因：
	
CNN中，输入数据为矩阵，卷积过后其尺寸往往缩小，这就使得矩阵的边缘部分只会参加一次运算，而其他部分则参加多次，从而导致结果弱化或者丢失边缘信息[尽管大多数情况下边缘信息不重要，但是不排除边缘信息有可能会重要]。

Tensorflow中的padding的计算公式如下：

	1, 输入矩阵: W*W
	2, Filter矩阵: F*F
	3, Stride值S,步长 
	4, 输出值的newheight,newwidth

**Padding = VALID:**

<div align=center>
<a href="https://www.codecogs.com/eqnedit.php?latex=newhight&space;=&space;newwidth&space;=&space;\left&space;\lceil&space;(W&space;-&space;F&plus;1)/S&space;\right&space;\rceil" target="_blank"><img src="https://latex.codecogs.com/gif.latex?newhight&space;=&space;newwidth&space;=&space;\left&space;\lceil&space;(W&space;-&space;F&plus;1)/S&space;\right&space;\rceil" title="newhight = newwidth = \left \lceil (W - F+1)/S \right \rceil" /></a></div>

**Padding = SAME:**

<div align=center>
<a href="https://www.codecogs.com/eqnedit.php?latex=newhight&space;=&space;newwidth&space;=&space;\left&space;\lceil&space;W/S&space;\right&space;\rceil" target="_blank"><img src="https://latex.codecogs.com/gif.latex?newhight&space;=&space;newwidth&space;=&space;\left&space;\lceil&space;W/S&space;\right&space;\rceil" title="newhight = newwidth = \left \lceil W/S \right \rceil" /></a></div>

从公式中可以看出,tensorflow中padding的圈数必须符合以上的公式，由内部计算出来，可能左右、上下都不相等[左右上下分别对应着特定的公式]。

###3.2, 经典网络架构

####LeNet-5模型

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/PUAU7lcICnO2fXslQmLRrESul7GsrNfS8J7g7wDd4Tw!/b/dEcBAAAAAAAA&bo=pgSmAQAAAAADByc!&rf=viewer_4" width="500" height="200" alt="LeNet-5模型"/>
</div>

如图所示，LeNet-5模型卷积神经网络共包含7层，分别为输入层、卷积层1、池化层1、卷积层2、池化层2、全连接层1和全连接层2。

####Inception-v3模型 

Inception架构的主要思想是找出如何用密集成分来近似最优的局部稀疏结。

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/XHzzowYLHhvXFM50u9AGwFrxdXSTJEV0KDMFDlVkexc!/b/dC8BAAAAAAAA&bo=RQOfAQAAAAADB*o!&rf=viewer_4" width="500" height="200" alt="Inception-v3模型"/>
</div>

对于上图，需要知道如下知识：

1. 采用不同大小的卷积核意味着不同大小的感受野，最后拼接意味着不同尺度特征的融合； 
2.  

Inception-v3网络，也即GoogleNet网络。如图所示，其是将不同的卷积层通过并联的方式结合在一起。 虽然过滤器的大小不同，但是如果所有的过滤器都使用全0填充且步长为1，那么前向传播得到的结果矩阵其长宽都和输入矩阵一致，最后经过不同过滤器处理的结果矩阵可以拼接成一个更深的矩阵。

###3.3，卷积神经网络的迁移学习

迁移学习:将一个问题上训练好的模型通过简单的调整使其适用于一个新的问题。

#引用

[1, 循环神经网络介绍] <http://blog.csdn.net/heyongluoyao8/article/details/48636251>

[2, RNN & LSTM 网络结构及应用] <https://www.jianshu.com/p/f3bde26febed>

[3, 自适应学习率调整] <http://www.cnblogs.com/neopenx/p/4768388.html>

[4, 各种优化算法比较] <http://blog.csdn.net/luo123n/article/details/48239963>

[5, numpy.random] <https://blog.csdn.net/vicdd/article/details/52667709>

[6, Tensorflow的可视化工具Tensorboard的初步使用] <https://blog.csdn.net/sinat_33761963/article/details/62433234>

[7, Tensorflow中卷积的padding操作] <https://www.jianshu.com/p/05c4f1621c7e>

[8, CS231n课程笔记翻译：卷积神经网络笔记] <https://zhuanlan.zhihu.com/p/22038289>

[8, 深度学习卷积神经网络——经典网络GoogLeNet(Inception V3)网络的搭建与实现] <https://blog.csdn.net/loveliuzz/article/details/79135583>

[9, 在AlexNet中LRN 局部响应归一化的理解]
<https://blog.csdn.net/program_developer/article/details/79430119>
