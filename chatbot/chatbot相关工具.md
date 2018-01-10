#chatbot相关工具

## 一，word2Vec模型

### 1.1 词的向量化方式

#### 1.1.1 bag of word
BOW使用一组无序的单词来表达一段文字和文档。

词频作为向量值: 每个词的词向量表示此单词在文本中出现的频率。
词权重作为向量值: 每个词在文本中的权重(代表此词对于文本的重要性程度，例如TFIDF值)代表一个词在词向量中的值。


bag of word的优点在于构造简单、可解释性较强。
其缺陷在于：
	1， 训练预料较多时，其词向量的维度非常大，不利于训练。
	2， 忽略了文本的语法和语序要素。
#### 1.1.2 distributed representation
 distributed representation是一个稠密、低维的实数向量，每一维代表一个词的潜在特征，包含词的句法和语义信息。
 word2Vec是一种典型的算法。
 
 优点在于包含词的语义信息，维度一般较小利用训练。
 其缺陷在于：
 	1， 需要提前进行训练得到每个词的词向量，复杂度较高。
 	2， 同样忽略了文本的词序要素。
 	
#### 1.1.3 主题模型中的topic-word向量
 	
主题模型建立后，会得到两个矩阵，一个是[主题x词语]矩阵，一个是[文档x主题]矩阵,其中的[主题x词语]矩阵可用于处理，生成词向量，其中向量的每个维度代表词在特定主题下的频率，自带一定的语义。公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/JLFdU7.45H5OdGZMG7qBWZeNYPZAxRDhwG5ma*iQKEo!/b/dPMAAAAAAAAA&bo=pgLvAAAAAAADB2k!&rf=viewer_4" /></a></div>
 	
其中”文档-词语”矩阵表示每个文档中每个单词的词频，即出现的概率；”主题-词语”矩阵表示每个主题中每个单词的出现概率；”文档-主题”矩阵表示每个文档中每个主题出现的概率。

文档主题模型的优点：1， 词向量包含语义潜在特征。2，维度较少。
文档主题模型的缺点：1， 提前需对词进行训练，复杂度高。2，虽然词包含着语义特性，但是语义特性比较分散[每个词对应多个主题，语义较分散]。

### 2 基于Hierarchical Softmax模型
word2Vec是一种词转换为空间向量的模型工具，使得具有相似语义的单词具有相近的词向量。其中语言模型主要包含两种，分别是skip gram模型、CBOW模型。

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/07t1yc83hqBoPNCvUgXwDUJOTrKcR34AYFYKjiATCIU!/b/dPMAAAAAAAAA&bo=jgKFAQAAAAADFzo!&rf=viewer_4" /></a></div>

其中skip gram模型用于给定关键字，预测其各个上下文字的概率；CBOW模型用于给定上下文，预测输入关键字。

SKIP-GRAM模型优化的目标函数如下：



CBOW模型优化的目标函数如下：


#### 2.1 skip-gram模型

skip gram模型是一种非常重要的模型，可用于计算语义相关度。skip gram根据输入的关键字通过神经网络预测其上下文字，但是并不记录训练好的神经网络，仅仅记录神经网络中隐层的权值矩阵[权值矩阵记录了每个词对应的词向量]。

word2vec中词向量的生成就是基于skip-gram模型，并且word2vec有一个API，给定一个词获得语义最相近的词，其也基于skip-gram模型。

n-gram模型中，所有的词都来自于训练数据中，使用神经网络由词转换为词向量之前，需计算出训练数据中的词袋,对于训练数据中的任一单词均采用词袋模型的one-hot编码方式[00001000]。

skip-gram模型训练起始，不仅需要指出词向量的维数--即神经网络中隐层的神经元个数，还需指定skip_windows(输入词的前后词个数)和nums_skips(代表从窗口中取多少个不同词作为out_put word)大小，例如 skip_windows = 2:

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/0bmr*.uXmQ6d9pmR5OSJ0tJxgVgbx96nr8.Lqu5OZoQ!/b/dPIAAAAAAAAA&bo=iwKKAQAAAAADByA!&rf=viewer_4" /></a></div>

如下图，训练数据格式如下{input_word,output_word},其中input_word和output_word都采用one_hot的表示方式。神经网络输入input_word,输出词袋中各个词的共现概率，采用softmax分类器，使得output_word对应的位置概率最大，而其他位置概率最小，迭代训练使得全局的误差最小。 最终生成的权值矩阵即可作为词袋模型中每个词的词向量[对应隐层的神经元个数]。

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/5*RgX59n1t6L0yr.BIOWN.q3jAXDhmSCC3exhC2DMs8!/b/dPMAAAAAAAAA&bo=VQOAAgAAAAARB.Q!&rf=viewer_4" width=600 height=400/></a></div>

训练存在的问题:

神经网络训练之前，需要得到训练数据的词袋模型，假如词袋模型中词的个数为10000，每个词对应300维的词向量，那么输入层==> 隐层大约需要优化10000x300个参数，而隐层 ==> 输出层也需优化 10000x300个参数，这就造成非常大的时空开销，解决此问题 word2vec提出了一下3个方法:

	1.  将常见的单词组合（word pairs）或者词组作为单个“words”来处理。
	2.  对高频次单词进行抽样来减少训练样本的个数。 
	3.  对优化目标采用“negative sampling”[负采样]方法，这样每个训练样本的训练只会更新一小部分的模型权重，从而降低计算负担。

其中方法3不仅大大降低了训练过程中的计算负担，而且也提高了训练的词向量的质量。不同于原本每个训练样本更新所有的权重，负采样每次让一个训练样本仅仅更新一小部分的权重，这样就会降低梯度下降过程中的计算量。



skip_gram模型采用多层神经网络计算共现词的概率，基于如下假设：若两个词共现的其他词越相似，那么这两个词的语义也就越相似，那么两个词的词向量也就越接近。

模型缺陷：

1. 采用多层神经网络+softmax分类器，训练的目标在于使得全局的损失函数最小化，导致训练时间过长。
2. 模型不够强健，对于新词或训练语料中没有的词，其不生成词向量[可采用fastTest n-gram模型解决]。
3. 要求训练语料足够多，足够覆盖大部分词，足以覆盖大部分情况。

#### 2.2 CBOW模型





# 引用

[1. 常见的词向量类型]  https://www.jianshu.com/p/76fb7ccb70d1
[2. word2vec中的数学原理详解]  http://www.cnblogs.com/peghoty/p/3857839.html




