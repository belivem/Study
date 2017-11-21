#Tensorflow学习
##Install Tensorflow
Tensorflow的安装有多种形式，本文采用基于 Anaconda 的安装方式，步骤如下：

	1. conda create -n tensorflow python=3.6  --新建一个tensorflow环境
	2. source activate/deactivate tensorflow --激活/停止环境[/users/liyanan/anaconda/envs]
	3. conda search tensorflow  --conda库中搜索tensorflow
	4. conda install tensorflow --conda库中安装tensorflow
	5. conda info -e   --列出当前所有的环境
	 
##Base Tensorflow

Tensorflow的op中兼有GPU和CPU实现[Tensorflow安装时的选择]，那么当算子被指派设备时，GPU有优先权，系统会自动将OP指派到GPU中执行。当然也可以手工指派设备，利用"with tf.device('/cpu:0')"。

1. Tensorflow采用graph(图)来表示计算任务。
2. Tensorflow在Session(绘画)的上下文中执行图。
3. Tensorflow使用tensor表示数据，通过变量(Variable)维护状态。
4. Tensorflow使用feed和fetch方法为操作赋值和拉取数据。

Tensor是一个对象，代表一个op的输出，但是Tensor并不存储op输出的值，而仅是提供了计算这些输出值的一种方式。Tensor数据结构有两个目的：1，作为另一个op的输入，表征图的连接。2，Tensor作为参数传递给Session run，但是run(Tensor...)后的结果并不属于Tensor。
Tensor具有以下几个属性：
	
	1, dtype,代表Tensor包含元素的类型。
	2，name，代表Tensor的字符串名字。
	3，graph,包含此Tensor的Graph。
	4，op,生成此Tensor的op。
	5，consumers,对应op，代表传递此Tensor的op。
	6，value_index，Tensor代表的元素的索引。
	7，get_shape()，返回此Tensor的形状，例如2*3。
	8，eval，在Session中执行此Tensor。
	
Variables也是一种tensor，训练模型时，变量用来存储和更新参数，使用之前需要显示的初始化操作，而在模型训练后必须存放至磁盘中。神经网络中，Variables一般都是模型参数，在每轮训练时保存更新后的参数，并且变量还可以被模型保存和加载。一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。

##TensorFlow 基本API
##变量
	

	tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, trainable=True, collections=None)  --此方法用户获得变量的值




##Tensorflow 数据输入

Tensorflow支持多种输入数据类型，例如csv,txt等。但是Tensorflow推荐数据类型为TFRecords。此种数据类型可以支持QueuRunner和Coordinator进行多线程数据读取，并且可以通过batch size和epoch参数来控制训练时单次batch的大小和对样本文件迭代训练的轮数。


##Examples Tensorflow
本小节讲述常用的机器学习经典事例。
###MNIST  --手写数字识别
####1、 Softmax Regression  Model
多分类问题属于指数分布族，广义线性模型拟合多项分布，由广义线性模型推导出假设函数。[0-1]分布属于伯努利分布，p(y = 1; φ) = φ; p(y = 0; φ) = 1 − φ;伯努利分布转换为指数分布如下：

<a href="http://www.codecogs.com/eqnedit.php?latex=p(y;\o)=\o^{y}*(1-\o)^{1-y}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(y;\o)=\o^{y}*(1-\o)^{1-y}" title="p(y;\o)=\o^{y}*(1-\o)^{1-y}" /></a>

Softmax回归模型是逻辑回归在多分类问题上的推广，且是逻辑回归的一般形式。SoftMax回归假设函数输出值为一个向量，且向量中每个值代表属于此类的概率，则其假设函数如下：

<div align=center>
<img src="http://a3.qpic.cn/psb?/V14Ifnin2f6pWC/HY*dcCqTaSB7vMZJ1X8rH1hYeqskgs7KDoKKwfArziQ!/b/dPIAAAAAAAAA&bo=vAYKAgAAAAADB5A!&rf=viewer_4" width="1000" height="180" alt="go struct结构"/>
</div>

从图中可以看出，K个类，每个类都会有一种参数向量，所以K个类训练需要得出K种不同的参数向量。

损失函数如下：

<div align=center>
<img src="http://a3.qpic.cn/psb?/V14Ifnin2f6pWC/Jzo0l1zwMcw0KFRg4U3Rqt1u6667cNrdxptk3yJfZUw!/b/dPIAAAAAAAAA&bo=dAT.AAAAAAADAKs!&rf=viewer_4" width="600" height="80" alt="损失函数"/>
</div>

SoftMax函数中样本属于第j类的概率如下：

<div align=center>
<img src="http://a1.qpic.cn/psb?/V14Ifnin2f6pWC/BjhqA0DU74W8woxRUM2pwF67NA2d7BmZgZTCe.Geazs!/b/dGsBAAAAAAAA&bo=vALOAAAAAAADB1I!&rf=viewer_4" width="300" height="80" alt="样本属于第j个类概率"/>
</div>

训练与逻辑回归类似，也采用梯度下降的方式，但是SoftMax回归模型的参数具有“冗余”性，即最优解不止一个，即从一个参数向量中减去另一个向量，完全不影响其值，这也就导致了假设函数的最优解附近为平的，基本上不引起函数值的变化。根据这个冗余性，可以采用函数衰减的方法解决，代价函数如下：

<div align=center>
<img src="http://a3.qpic.cn/psb?/V14Ifnin2f6pWC/QwHzOnU0MdgZgQKB.6VW5gx2yy1O50P.9*ZgWA9whfU!/b/dPIAAAAAAAAA&bo=8wWAAgAAAAADAFE!&rf=viewer_4" width="1000" height="200" alt="加入衰减因子损失函数"/>
</div>  
其中损失函数的倒数应为参数矩阵，行代表类别，列代表每个类别中特征的权值。

####2、深入mnist数据集

通常的，为了进行高效的矩阵计算，通常情况下我们会使用Numpy这样的矩阵计算库，将一些耗时的操作例如矩阵计算在Python的外部环境中，计算完成后再切入通常的Python环境中，但是这样会产生损耗--数据传输，尤其是在多CPU,GPU的环境下。但是Tensorflow为了避免这种数据传输的损耗，采用了这样的方法：构建一个操作图，然后完全运行此图在Python环境外面。

深入mnist数据集采用卷积神经网络的方法拟合训练数据集，步骤如下:

1. 卷积层和pooling层都为2层[卷积->pooling->卷积->pooling]，在最后一个pooling层结束的时候，其feature map的大小为7*7,共64个feature map。
2. 对所有的feature map作为新的输入，采用全连接的方式，生成新的1024个隐藏层神经单元。
3. 隐藏层神经单元与softmax回归单元相连接，输出10维向量，结束。

图示如下：
<div align=center>
<img src="http://a3.qpic.cn/psb?/V14Ifnin2f6pWC/uZPblmfQcYpVJeC9g3h.aXDQCD9nInBNOa7EYkvKUgM!/b/dPIAAAAAAAAA&bo=bQaAAgAAAAARB9k!&rf=viewer_4" width="800" height="300" alt="cnn_mnist"/>
</div>  

注：1，步骤1中的第二卷积层每个feature map数据计算来源于第一pooling层的所有的32个feature map。2，pooling采用max_pooling的方式。3，第一卷积层的feature map大小为28*28，是因为卷积时采用了padding="SAME"的方法。

####3、 tensorflow通用运作方式



#问题
1. softmax为何将交叉熵作为损失函数?  首先在y取值为一个向量时，那么交叉熵即为上文中的损失函数，交叉熵作为损失函数，依然可以采用逻辑回归中的定义解释：softmax回归依然将损失函数定义为训练样本Y关于X的分布，依然用于描述训练样本的分布，而对此分布采用极大似然函数 即可。
#文献
[1. 指数分布簇](http://www.cnblogs.com/BYRans/p/4735409.html)

