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

#引用
[1. TF-IDF及其算法]  http://blog.csdn.net/sangyongjia/article/details/52440063

