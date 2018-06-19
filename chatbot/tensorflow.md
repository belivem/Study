#Tensorflow 

Tensorflow是Google开发的一款神经网络Python外部的结构包，也是一个采用数据流图来进行数值计算的开源软件库。Tensorflow让我们先绘制计算流程图，让后将其转换为更高效的C++代码在后端进行计算。Tensorflow首先要定义神经网络的结构，然后再把数据放入结构当中去运算和training。

Tensorflow框架可以很好地支持深度学习算法，但是其不限于深度学习[深度学习重要特征：1，多层。2，非线性]。	

## 1, Tensorflow数据类型

**图(Graph)**

	tf.get_default_graph() 	--> 获得当前线程的默认计算图，a.graph代表变量a所属的计算图
	
	Tensorflow支持通过tf.Graph()生成新的计算图，并且不同的计算图上的张量和运算都不会共享 ==> 私有！

**张量(Tensor)**

张量有多种，其中零阶张量称为标量==》 也就是一个数值，比如[1]；一阶张量为向量，比如一维数组[1,2,3]；二阶张量为矩阵，比如二维的[[1,2,3],[4,5,6]...],以此类推；

tf.concat([col1,col2],0)  ==> 将张量col1和col2按照axis=0合并为一个新的tensor

张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程。一个张量中主要保存了三个属性：[名字/shape/类型]

**会话控制(Session)**

Session是Tensorflow为了控制的关键语句，运行Session.run(result/option)可以获得你要的运算结果或者是你要运算的部分。

会话拥有并管理Tensorflow程序运行时的所有资源，当所有计算完成之后需要关闭会话来对资源进行回收，否则可能出现资源泄露的情况。
**变量(Variable)**

在 Tensorflow 中，变量的作用就是保存和更新神经网络中的参数；在Tensorflow中设定了变量，那么对其进行初始化将是十分重要的；

Tensorflow中，所有的变量都会自动的加入GraphKeys.VARIABLES这个集合，可通过函数tf.global_variables()查看图中所有的变量；tf.trainable_variables()查看神经网络中所有需要优化的参数。

**常量(constant)**
神经网络中，*偏置项*通常使用常数设置初始值。Tensorflow中，每生成一个常量，Tensorflow都会在计算图中增加一个节点。

**占位符(Placeholder)**

Tensorflow中，占位符(tf.placeholder())用于暂时存储变量；占位符往往作用于外部传入data,其传输数据格式如下：sess.run(y,feed_dict={x:value})

placeholder节点被声明的时候是未初始化的， 也不包含数据， 如果没有为它供给数据， 则TensorFlow运算的时候会产生错误。

x = tf.placeholder(dtype=tf.float32,shape=[None,2]) : 使用None可以方便的使用不同的batch

##2, IO[输入输出]
TensorFlow程序读取数据一共有3种方法:
	•	供给数据(Feeding)： 在TensorFlow程序运行的每一步， 让Python代码来供给数据。
	•	从文件读取数据： 在TensorFlow图的起始， 让一个输入管线从文件中读取数据。
	•	预加载数据： 在TensorFlow图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况)。

产生文件名列表：
[("file%d" % i) for i in range(2)]  / tf.train.match_filenames_once()  ==> filename_list
tf.train.string_input_producer(filename_list)   ==>  filename_queue

tf.TextLineReader() || tf.decode_csv  ==> 从文本文件读取数据
tf.FixedLengthRecordReader() || tf.decode_raw  ==> 读取二进制文件
tf.python_io.TFRecordWriter()    ==> 写入文件 TFRecord
tf.TFRecordReader() || tf.parse_single_example()  ==> 读取TFRecord文件

tf.train.shuffle_batch()  ==>   输入数据乱序处理
tf.train.shuffle_batch_join()  ==> 乱序处理升级版，更强的乱序处理[不同文件样本 作为一个batch_size]

threads = tf.train.start_queue_runners(coord=coord) ==> 开始数据读取[因为数据读取为线程，启动输入管道的线程]，需配合使用coord = tf.train.Coordinator() 使得在发生错误的时候正常关闭线程

##3, 模型的持久化

模型持久化指将训练后的模型持久化至硬盘中，使得我们后续可以继续使用已训练好的模型。Tensorflow中模型持久化保存文件类型为"CKPT"格式。

**保存为CKPT格式**



**保存为PB格式**



##4, 常用函数

1, tf.multiply和tf.matmul
    
		1> tf.matmul() 为矩阵的乘法 shape[2,3].shape[3,2]  = shape[2,2]
		2> tf.multiply() 为矩阵中各个数的乘法 shape[2,3].shape[2,3] = shape[2,3]

2, 数学计算相关
	
		1> tensor.eval()： 在一个Seesion里面“评估”tensor的值（其实就是计算），当然首先执行计算值之前的各个必要操作。
		2> tf.argmax()/tf.argmin(): 得到向量或者矩阵中每行的最大值/最小值所在位置索引，其中对于向量,参数(axis = 0),矩阵(axis = 1)。
		3> tf.diag_part(): 返回矩阵的对角线元素

3, 分布相关 ==>

		1> tf.random_normal()  : 正太分布
		2> tf.truncated_normal() : 正太分布，若随机数偏离，则重新分布 
		3> tf.random_uniform() : 均匀分布
		4> tf.random_poisson() : 泊松分布
		5> tf.random_gamma() : 伽马分布

4，常量声明方法 ==> 

		1> tf.zeros()  生成全0的数组
		2> tf.ones()   生成全1的数组
		3> tf.fill()   生成一个全部为给定数值的数组

5, 常用Tensor转换方法

		1> tf.clip_by_value(t,min,max) : 改变Tensor t中值，使得最小为min,最大为max

6，激活函数相关

		1> tf.sigmoid()
		2> tf.nn.softmax()
		3> tf.tanh()
		4> tf.nn.relu()
		5> tf.nn.crelu()
		
		
7，

		1> 

8, 损失函数相关

	**softmax损失函数**
	
		1> c1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=input_y)
        2> c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(input_y,axis=1))
		注: c1与c2值都为一个向量，而不是一个具体的实数值，若要求得交叉熵那么需要tf.reduce_sum(c1)[向量的各个维度相加]，若要求平均交叉熵损失函数，需要tf.reduce_mean(tf.reduce_sum(c1)),
			c1和c2函数的区别在于输入参数labels，c1为类别的one_hot表示，而c2为特定的类别(例如，one_hot表示的最大值索引，[0.0,0.0,1.0] ==> 最大索引为2)。其中，logits为神经网络的最后输出，
			也即softmax函数的输入。
		3> c3 = tf.nn.sigmoid_cross_entropy_with_logits()
		4> c4 = tf.nn.weighted_sigmoid_cross_entropy_with_logits()
	
	**l1,l2正则化**

		1> tf.keras.regularizers.l1(l=0.01)  ==> l1正则化
		2> tf.keras.regularizers.l2(l=0.01)  ==> l2正则化
	
	**学习率相关**

		1> tf.train.exponential_decay(learning_rate,global_step,decay_steps,decay_rate,staircase=true/false) ==>指数衰减法 非常有用，目的训练起始较大学习率，随着训练轮次进行，学习率变小。
			公式如下：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
			learning_rate :  起始学习率
			global_step   :  训练第几个轮次，从0起始依次增大
			decay_steps   :  衰减轮数
			decay_rate    :  衰减稀疏 0.99,0.98等
			staircase     :  是否采用阶梯型，采用阶梯型则指每经过100轮其学习率才变化
	
	**滑动平均模型**
		作为一个优化方案==>滑动平均值往往在测试集上表现更好，也更加健壮。为每个变量[weights and biases]维护一个影子变量,预测的时候采用影子变量，公式见下图
		1> ema = tf.train.ExponentialMovingAverage(decay=0.99,num_updates=global_steps); moving_average_op = ema.apply(tf.trainable_variables());
			1 训练阶段：为每个可训练的权重维护影子变量，并随着迭代的进行更新；
			2 预测阶段：使用影子变量替代真实变量值，进行预测。

	**l2泛化**
		
		1> tf.nn.l2_normalize()   ==> L2泛化  公式如下:

<div align=center>
<a href="https://www.codecogs.com/eqnedit.php?latex=norm_{l2}[x_{1},x_{2},x_{3}]&space;=&space;\frac{[x_{1},x_{2},x_{3}]}{\sqrt{{x_{1}}^{2}&plus;x_{2}^{2}&plus;x_{3}^{2}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?norm_{l2}[x_{1},x_{2},x_{3}]&space;=&space;\frac{[x_{1},x_{2},x_{3}]}{\sqrt{{x_{1}}^{2}&plus;x_{2}^{2}&plus;x_{3}^{2}}}" title="norm_{l2}[x_{1},x_{2},x_{3}] = \frac{[x_{1},x_{2},x_{3}]}{\sqrt{{x_{1}}^{2}+x_{2}^{2}+x_{3}^{2}}}" /></a></div>



9, 控制相关函数

		1> tf.cond(pred,fn1,fn2,name="")   <==>  res = fn1() if pred else fn2() 
		2> tf.case(pred_fn_pairs, default, exclusive=False, name=’case’)  ==> 更强大的tf.cond
		3> tf.group() 创建一个包含几个操作的op节点
