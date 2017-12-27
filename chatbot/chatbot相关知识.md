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

###2, 语言模型
#### 2.1，N-Gram语言模型
n-gram是自然语言处理中的一种非常重要的模型。
#####2.1.1 基于n-gram模型定义的字符串距离
字符串的相似度，不仅可以使用余弦相似度、杰卡德相似系数、编辑距离、最长相同字符串、最长相同字符序列等，还可以使用n-gram距离。n-gram距离定义如下：设定两个字符串s、t，计算字符串s的n-gram序列和字符串t的n-gram序列，则n-gram距离公式如下：
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=\left&space;|&space;G_{n}(s)&space;\right&space;|&plus;\left&space;|&space;G_{n}(t)&space;\right&space;|&space;-&space;2*|&space;G_{n}(s)\bigcap&space;G_{n}(t)|" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\left&space;|&space;G_{n}(s)&space;\right&space;|&plus;\left&space;|&space;G_{n}(t)&space;\right&space;|&space;-&space;2*|&space;G_{n}(s)\bigcap&space;G_{n}(t)|" title="\left | G_{n}(s) \right |+\left | G_{n}(t) \right | - 2*| G_{n}(s)\bigcap G_{n}(t)|" /></a></div>
<div align=center>
或
</div>
<div align=center>
<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{2*|G_{n}(s)\bigcap&space;G_{n}(t)|}{|G_{n}(s)|&plus;|G_{n}(t)|}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{2*|G_{n}(s)\bigcap&space;G_{n}(t)|}{|G_{n}(s)|&plus;|G_{n}(t)|}" title="\frac{2*|G_{n}(s)\bigcap G_{n}(t)|}{|G_{n}(s)|+|G_{n}(t)|}" /></a></div>

其中Gn(s)代表字符串s中的n-gram序列，同理Gn(t)代表字符串t中的n-gram序列。

#####2.1.2 利用n-gram评估语句是否合理
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

#####2.1.3 n-gram语言模型与朴素贝叶斯
朴素贝叶斯的一个非常大的局限性在于其条件独立性，文本分类中假设BOW模型中词语之间相互独立，省略了位置顺序的信息。

n-gram即可使其从条件独立性==>联合概率链规则。在文本分类领域中，可以把独立性假设理解为1-gram模型。


### 3，马尔科夫理论
####3.1 马尔科夫链假设
当前词只与前面几个有限的词相关。

#引用
[1. TF-IDF及其算法]  http://blog.csdn.net/sangyongjia/article/details/52440063

