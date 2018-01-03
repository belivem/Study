#chatbot相关知识
 
## 一、 基础技术
### 1，TF-IDF
tf_idf技术经常用于衡量一个词在某个文件中的重要程度。其中tf代指词频，而IDF代指逆词频，指一个词的普遍重要性的程度。其中tfidf的值与词频成正比而与逆词频成反比。

TF的值通常会被归一化【**用于防止偏向长文本文件，对应于一个特定文件，同一个词在不同文件中的TF值不同**]。其计算方式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" title="tf_{i,j} = \frac{n_{i,j}}{\sum _{k}n_{k,j}}" /></a></div>

其中n(i,j)代表词ti在文件dj中出现的次数，n(k,j)代表所有词在文件dj中出现的总次数。

IDF的值对应于**一个文件集或者一个数据集**，在一个数据集中，任一词对应的IDF值是固定不变的。其值为 总文件个数除以包含此词的文件个数的商求对数。计算方式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=idf_{i}&space;=&space;log\frac{totalfiles}{containfiles}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?idf_{i}&space;=&space;log\frac{totalfiles}{containfiles}" title="idf_{i} = log\frac{totalfiles}{containfiles}" /></a></div>

其中totalfiles代指数据集中所有的文件个数，而containfiles指包含词ti的文件个数。对应ti来说，若包含的文件个数越多，其idf越小，也就代表中词ti在数据集中越普通，也即不能有效显示一个文件相对于其他文件的特殊性。

TFIDF的主要思想是：如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现[即IDF值低]，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

**缺陷：**
1. TFIDF并不能用来说明特征词的重要与否[因为一个词是否重要，有很大程度上与其出现的次数有关]，而仅仅用于区分不同的文档。
2. TFIDF中IDF的计算增大了Documents中生僻词的权重。
3. TFIDF的计算 需要使用较大的数据集。

### 2. 熵
#### 2.1，KL距离
KL距离也称KL散度，相对熵等。是一种衡量相同事件空间内的两个概率分布的差异。物理意义是指：相同事件空间中，概率分布P(x)，若用概率分布Q(x)编码时，平均每个基本事件的编码长度增加了多少比特[不存在其他比按照本身概率分布更好的编码方式]，故而KL距离值>0。计算公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=D(P||Q)&space;=&space;\sum_{x\subset&space;X}^{X}P(x)log(\frac{P(x)}{Q(x)})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D(P||Q)&space;=&space;\sum_{x\subset&space;X}^{X}P(x)log(\frac{P(x)}{Q(x)})" title="D(P||Q) = \sum_{x\subset X}^{X}P(x)log(\frac{P(x)}{Q(x)})" /></a></div>

从公式可以看出，当两个概率完全相同时，其KL散度为0，代表平均增加的bit数为0.

信息量公式:
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=I(x_{0})&space;=&space;-logP(x_{0})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?I(x_{0})&space;=&space;-logP(x_{0})" title="I(x_{0}) = -logP(x_{0})" /></a></div>

信息量公式中的P(x0)代表事件空间中，事件x0发生的概率。常理上，信息量可以理解为，一个事件发生的概率越大，则它所携带的信息量就越小 ==》信息量越大，变量的取值也就越不确定，反之就越确定。当熵为0时，也即表示事件的发生是必然的，其发生不会导致任何信息量的增加。

而信息熵可理解为信息量的**期望，一个变量所有可能取值信息量的期望**，概率分布P(x)的信息熵[不存在其他比按照本身概率分布更好的编码方式，熵的主要作用是告诉我们最优编码信息方案的理论下界（存储空间），以及度量数据的信息量的一种方式。一个事件发生的概率越大，其所携带的信息量就越少]如下：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=H(x)&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)log(P(x))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H(x)&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)log(P(x))" title="H(x) = -\sum_{x\subset X}^{X}P(x)log(P(x))" /></a></div>

KL距离虽然也称距离，但是其并不是真正的距离，因其不满足距离的三要素条件--2，对称性。

例如  假设一个事件空间，真实概率分布为A(x),另外有两个分布，分别为B(x)和C(x)，其中D(A||B) < D(A||C)，那么就认为概率分布B更接近真实分布A，即用B代替真实的概率分布A其需增加的比特数更少。

KL距离的应用：

1. KL距离可以用于简单的分类、比较文章的相似度等，例如分别计算两篇文章的KL距离 ==》 KL距离更小则表明两篇文章越接近 ==》 主题越接近。

2. 可用于评价分类方法的好坏，例如 四个类别，真实的概率分布P为0.1,0.2,0.3,0.4，方法A得到的概率分布Q为0.4,0.3,0.2,0.1，方法B得到的概率分布R为0.5,0.1,0.2,0.2，那么分别计算KL散度 D(P||Q),D(P||R)，若 D(P||Q) > D(P||R)，那么方法R的分类效果较好，更接近于真实的概率分布。

3. 统计学中，可使用一个较简单的分布 来代替更复杂的分布。例如真实的复杂分布P，可采用相对简单的分布Q来代替，KL散度能够帮助度量使用一个分布来近似另一个分布时所损失的信息。


KL散度缺陷：
	
1. KL散度的计算公式中D(P||Q)中，其Q(x)的值不能为0.
2. KL散度的值有一定的离散性，并不存在于[0,1]区间中。
3. KL距离用于比较两个概率分布的相似性，MI(互信息)用于比较两个概率分布的独立性。
	
#### 2.2，交叉熵
交叉熵与相对熵极容易混淆，其计算公式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=CEH(P,Q)&space;=&space;H(P)&plus;D(P||Q)&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)log(P(x))&plus;\sum_{x\subset&space;X}^{X}P(x)log\frac{P(x)}{Q(x)}&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)logQ(x)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?CEH(P,Q)&space;=&space;H(P)&plus;D(P||Q)&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)log(P(x))&plus;\sum_{x\subset&space;X}^{X}P(x)log\frac{P(x)}{Q(x)}&space;=&space;-\sum_{x\subset&space;X}^{X}P(x)logQ(x)" title="CEH(P,Q) = H(P)+D(P||Q) = -\sum_{x\subset X}^{X}P(x)log(P(x))+\sum_{x\subset X}^{X}P(x)log\frac{P(x)}{Q(x)} = -\sum_{x\subset X}^{X}P(x)logQ(x)" /></a></div>

上述公式中，H(P)代表分布P的信息熵,D(P||Q)代表KL散度。相对于相对熵，交叉熵多了一个信息熵H(P),当p已知时，交叉熵与相对熵在行为上是等价的,都反映了概率分布P和Q相似程度。

P代表样本的真实分布，Q代表样本的一个错误分布，则信息熵H(P)用于衡量识别一个样本所需的编码长度的期望--平均编码长度，交叉熵H(P,Q)代表使用错误分布Q表示真实分布P的样本的平均编码长度。

交叉熵的应用：

1. 交叉熵可以作为神经网络中的损失函数，其中P表示真实的分布，而Q表示训练后模型的预测标记分布。

#### 2.3，互信息

互信息指两个变量的独立程度，定义为两个变量X,Y的联合分布和独立分布乘积的相对熵，公式为：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=I(X,Y)&space;=&space;D(P(X,Y)||P(X)P(Y))&space;=&space;\sum_{xy}^{n}P(x,y)log\frac{P(x,y)}{p(x)p(y)}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?I(X,Y)&space;=&space;D(P(X,Y)||P(X)P(Y))&space;=&space;\sum_{xy}^{n}P(x,y)log\frac{P(x,y)}{p(x)p(y)}" title="I(X,Y) = D(P(X,Y)||P(X)P(Y)) = \sum_{xy}^{n}P(x,y)log\frac{P(x,y)}{p(x)p(y)}" /></a></div>

相对熵D(P||Q)代表分布P和分布Q的相似程度，相对熵越大，代表分布P和Q的相似度越小，若P==Q，则相对熵为0；互信息代表I(X,Y)代表两个分布X和Y的 联合概率分布和独立分布乘积的相对熵，若P(x,y) == P(x)P(y)那么变量X和变量Y独立，也即变量X,Y的联合概率分布和独立分布的相对熵为0；

#### 2.4，最大熵

熵是随机变量不确定性的度量，不确定性越大、熵值也就越大。当随机变量退化为一个定值时，其熵也为0.均匀分布为 最不确定 的分布，也即熵最大的分布。

最大熵模型是统计学的一般原理，是概率模型学习的一个准则。最大熵原理认为，学习概率模型时，在所有可能的概率模型中，熵最大的模型是一种最好的模型 ==》 也即在已知的条件下，寻找熵最大--也即不人为增加已有的知识，不确定性最大。最大熵模型根据最大熵原理在一定的特征限制下选择最优的 概率分布。

### 3, 随机场

#### 3.1, 条件随机场


### 4， 分布

#### 4.1 吉布斯分布


## 二、 统计模型

### 1，噪声信道模型
噪声信道模型是一个非常重要的模型，在很多领域都有重要的作用--试图通过有噪声的输出信号恢复其输入信号。其形式如下：

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/D1IJ.y3PNkYtxy*r8lEMyHPx00xcz1jJYyq*MaVTNOI!/b/dGUBAAAAAAAA&bo=UQJ3AAAAAAADFxY!&rf=viewer_4" /></a></div>

如图，噪声信道模型的作用在于--依据输出(带噪声)O，找到最大概率的输出I。公式定义如下：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=I&space;=&space;argmax(P(I/O))&space;=&space;argmax(\frac{P(I)*P(O/I)}{P(O)})=argmax(P(I)*P(O/I))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?I&space;=&space;argmax(P(I/O))&space;=&space;argmax(\frac{P(I)*P(O/I)}{P(O)})=argmax(P(I)*P(O/I))" title="I = argmax(P(I/O)) = argmax(\frac{P(I)*P(O/I)}{P(O)})=argmax(P(I)*P(O/I))" /></a></div>

如以上公式所述：噪声信道模型基于贝叶斯概率，公式中P(O)已知，故求P(I)P(O/I)最大，其中根据输出O可得到一系列的输入I，并求P(I)，最后求最大的P(I)P(O/I).一般P(I)叫做语言模型，而P(O/I)叫做转换模型。

噪声信道模型是一种普适性的模型，可以通过修改噪声信道的定义，解决非常多的问题，其有效应用如下：

	1. 语音识别
	2. 机器翻译
	3. 手写体识别
	4. 文本校错
	5. 词性标注

存在的问题：

   1. 噪声信道模型基于统计贝叶斯公式，故而使得噪声信道模型只能作用于先验知识充足的场景--即语料中必须已有输出对应的输入--限制。
   2. 噪声信道模型完全基于统计贝叶斯公式--对语料的要求较高，其有效性可能有一定的影响--受限。
	


### 2，统计语言模型--N-Gram语言模型
n-gram是自然语言处理中的一种非常重要的模型。
#### 2.1 基于n-gram模型定义的字符串距离
字符串的相似度，不仅可以使用余弦相似度、杰卡德相似系数、编辑距离、最长相同字符串、最长相同字符序列等，还可以使用n-gram距离。n-gram距离定义如下：设定两个字符串s、t，计算字符串s的n-gram序列和字符串t的n-gram序列，则n-gram距离公式如下：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=\left&space;|&space;G_{n}(s)&space;\right&space;|&plus;\left&space;|&space;G_{n}(t)&space;\right&space;|&space;-&space;2*|&space;G_{n}(s)\bigcap&space;G_{n}(t)|" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\left&space;|&space;G_{n}(s)&space;\right&space;|&plus;\left&space;|&space;G_{n}(t)&space;\right&space;|&space;-&space;2*|&space;G_{n}(s)\bigcap&space;G_{n}(t)|" title="\left | G_{n}(s) \right |+\left | G_{n}(t) \right | - 2*| G_{n}(s)\bigcap G_{n}(t)|" /></a></div>
<div align=center>
或
</div>
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{2*|G_{n}(s)\bigcap&space;G_{n}(t)|}{|G_{n}(s)|&plus;|G_{n}(t)|}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{2*|G_{n}(s)\bigcap&space;G_{n}(t)|}{|G_{n}(s)|&plus;|G_{n}(t)|}" title="\frac{2*|G_{n}(s)\bigcap G_{n}(t)|}{|G_{n}(s)|+|G_{n}(t)|}" /></a></div>

其中Gn(s)代表字符串s中的n-gram序列，同理Gn(t)代表字符串t中的n-gram序列。

#### 2.2 利用n-gram评估语句是否合理
在已给语料的情况下，n-gram还可以用于判断所给句子s是否合理，例如，与马尔科夫链相结合的方式，2-gram公式如下：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=p(w_{1},w_{2},w_{3}...w_{n})&space;=&space;\prod_{i=1}^{n}p(w_{i}|w_{i-1})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(w_{1},w_{2},w_{3}...w_{n})&space;=&space;\prod_{i=1}^{n}p(w_{i}|w_{i-1})" title="p(w_{1},w_{2},w_{3}...w_{n}) = \prod_{i=1}^{n}p(w_{i}|w_{i-1})" /></a>
</div>
其中n-gram的数据平滑算法如下：

1. Laplacian (add-one) smoothing 
2. Add-k smoothing 
3. Jelinek-Mercer interpolation 
4. Katz backoff 
5. Absolute discounting 
6. Kneser-Ney

#### 2.3 n-gram语言模型与朴素贝叶斯
朴素贝叶斯的一个非常大的局限性在于其条件独立性，文本分类中假设BOW模型中词语之间相互独立，省略了位置顺序的信息。

n-gram即可使其从条件独立性==>联合概率链规则。在文本分类领域中，可以把独立性假设理解为1-gram模型。

### 3，word2Vec实现模型
word2Vec是一种词转换为空间向量的模型工具，使得具有相似语义的单词具有相近的词向量。其中语言模型主要包含两种，分别是skip gram模型、CBOW模型。

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/07t1yc83hqBoPNCvUgXwDUJOTrKcR34AYFYKjiATCIU!/b/dPMAAAAAAAAA&bo=jgKFAQAAAAADFzo!&rf=viewer_4" /></a></div>

其中skip gram模型用于给定关键字，预测其各个上下文字的概率；CBOW模型用于给定上下文，预测输入关键字。

#### 3.1 skip gram模型

skip gram模型是一种非常重要的模型，可用于计算语义相关度。skip gram根据输入的关键字通过神经网络预测其上下文字，但是并不记录训练好的神经网络，仅仅记录神经网络中隐层的权值矩阵[权值矩阵记录了每个词对应的词向量]。

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

#### 3.2 CBOW模型

### 3，马尔科夫理论

#### 3.1 隐马尔科夫模型 HMM

HMM描述一个隐藏的马尔科夫链生成不可观测的状态随机序列，再由各个状态生成观测随机序列的过程。

<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=tf_{i,j}&space;=&space;\frac{n_{i,j}}{\sum&space;_{k}n_{k,j}}" target="_blank"><img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/3pCBpZqAZLB4HxPDrmeDSusq62mnjyd.ifZ2gZHYwlI!/b/dD8BAAAAAAAA&bo=GwP6AAAAAAADB8A!&rf=viewer_4" /></a></div>



#### 3.2 马尔科夫链假设
当前词只与前面几个有限的词相关。




# 引用

[1. TF-IDF及其算法]  http://blog.csdn.net/sangyongjia/article/details/52440063
[2. NLP基础知识]  https://www.cnblogs.com/taojake-ML/p/6413715.html
[3. 最大熵模型中的数学推导]  http://blog.csdn.net/v_july_v/article/details/40508465
[4. 一文详解Word2vec之Skip-Gram模型]  http://www.sohu.com/a/151486278_651893




