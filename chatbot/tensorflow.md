#Tensorflow 

Tensorflow是Google开发的一款神经网络Python外部的结构包，也是一个采用数据流图来进行数值计算的开源软件库。Tensorflow让我们先绘制计算流程图，让后将其转换为更高效的C++代码在后端进行计算。Tensorflow首先要定义神经网络的结构，然后再把数据放入结构当中去运算和training。

Tensorflow框架可以很好地支持深度学习算法，但是其不限于深度学习。	

## 1, Tensorflow数据类型

**图(Graph)**

	tf.get_default_graph() 	--> 获得当前线程的默认计算图，a.graph代表变量a所属的计算图
	
	Tensorflow支持通过tf.Graph()生成新的计算图，并且不同的计算图上的张量和运算都不会共享 ==> 私有！

**张量(Tensor)**

张量有多种，其中零阶张量称为标量==》 也就是一个数值，比如[1]；一阶张量为向量，比如一维数组[1,2,3]；二阶张量为矩阵，比如二维的[[1,2,3],[4,5,6]...],以此类推；

**会话控制(Session)**

Session是Tensorflow为了控制的关键语句，运行Session.run(result/option)可以获得你要的运算结果或者是你要运算的部分。

**变量(Variable)**

在 Tensorflow 中，定义了某字符串是变量，它才是变量；在Tensorflow中设定了变量，那么对其进行初始化将是十分重要的；

**占位符(Placeholder)**

Tensorflow中，占位符(tf.placeholder())用于暂时存储变量；占位符往往作用于外部传入data,其传输数据格式如下：sess.run(x,feed_dict={input:x})


##2, IO[输入输出]

