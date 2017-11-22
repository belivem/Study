#TensorFlow实战 深度学习
##第一章 深度学习简介

##第二章 TensorFlow环境搭建

##第三章 Tensorflow入门
###3.1 TensorFlow计算模型--计算图
###3.2 TensorFlow数据模型--张量
TensorFlow中所有数据都用张量形式表示。张量可以简单理解为多维数组<功能上来讲>，例如0阶张量代表一个实数，一阶张量代表一维数组，多阶张量代表多维数组！但是张量并否直接使用数组形式。

**张量中并不保存真正的数字，保存的是如何得到这些数字的计算过程。**而要获得具体的计算数据，必须引用tensorflow的会话机制--tf.Session().  --print(tf.Session().run(result))

定义张量时，若不明确指出元素的类型，那么tensorflow就会采用默认类型，例如不带小数点的数会被默认为int32类型，而带小数点的数则会默认为float32类型，如果简单相加就会导致*类型不匹配。*

#TensorFlow基本API
##张量
	tf.constant(value,dtype,shape,name)   --tensorflow的张量方法
				 args: value,   常量值列表、数组
				 		dtype,   数组中每一个元素的类型
				 		shape,   张量的维数
						name,    张量的名称，张量的唯一标识符


#总结



