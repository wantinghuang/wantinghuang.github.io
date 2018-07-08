---
layout: post
title: Word Representation and Word Embeddings
---
Word Representation，意指用一組數字來代表文字的行為/方式。事實上，有很多種方式可以用一組數字代表文字，目前兩個主流的方式分別為 Distributional Semantics 和 Word Embeddings，而 Word Embeddings 是近年來常用的方式。[1]
<!-- more -->  

## 為什麼需要 Word Representation?
文字之所以重要，是因為文字本身帶有意涵，我們能從文字中讀到的是其背後的意義，想要對文字的意涵做更多的分析，在計算語言學中就引發了一個問題: 「**有沒有辦法用一組數字來代表文字的意涵?**」 有了這樣的一組數字，我們便可將文字可以放到數學模型中分析。

## Word Representation 的演變
要把文字轉成數字，簡單的方法就是做 one-hot encoding/representation (統計上稱為 dummy variable)，也就是令一個向量，長度為所有出現過的字的個數(vocabulary list)，這個向量的每一個位置會對應到 vocabulary list 裡面的某一個字，則每個字可以由某個位置為1，其餘為0的向量代表。如下圖所示
![one-hot representation](https://i.imgur.com/NpjvCyC.png)
我們可以觀察到兩個點:
1. 字的向量跟 vocabulary list 的順序有關係；也就是說換個順序，字就有不同的向量表示，所以 vocabulary list 要先固定。
2. 向量並無法反映字跟字之間的關係。舉例來說，car 向量跟 bike 向量的歐式距離是$\sqrt{2}$，car 向量跟 sun 向量也是$\sqrt{2}$，但 car 跟 bike 意義上應該要比較近(同屬交通工具)。

```
Issues: difficult to compute the similarity
```

所以我們的目標更明確了，要找出一組數字可以代表文字，且能反映字跟字之間的關係。
```
Goal: word representation that capture the relationships between words
```

衡量文字的意涵，經常是由上下文推斷，因此我們想找出來代表文字的數字要能反映字詞之間的關係。
```
Idea: words with similar meanings often have similar neighbors
```
我們可以利用前後出現的字，建立一張出現次數表，稱為 Window-based Co-occurrence Matrix。如下圖:

![Window-based Co-occurrence Matrix](https://i.imgur.com/gFW22E8.png)

來源: [NTU-ADLxMLDS word representation y.v.chen slides](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/doc/170925+170928_WordRepresentation.pdf)

love 前面或後面出現 I 的次數是 2，enjoy 前面或後面出現 I 的次數是 1，我們以每個 column 作為最上面的字的代表向量 (長度一樣為 vocabulary list)，則 love 向量跟 enjoy 向量之間的距離，就跟 enjoy 向量和 deep 向量之間的距離不一樣了，也就意味著這樣的 vector representation 可以反映出字跟字之間的關係。
但這樣的表示方法有一些缺點:
1. 當字很多的時候，矩陣 size 很大，向量維度也很高
2. 矩陣容易有很多 0 ，矩陣 sparse ，則放入模型不容易分析。

```
Issues:
* matrix size increases with vocabulary
* high dimensional
* sparsity -> poor robustness
```

所以我們需要對基於 window-based co-occurrence matrix 的 vector representation 降維。
講到降維，第一個想到的應該是 PCA (Principal Component Analysis)，PCA 是基於 SVD (Singular Value Decomposition) 的降維方式，SVD 在 NLP 裡的應用叫做 Latent Semantic Analysis (LSA)。
簡單講，令 $C$ 是 所有字的 vector representation 組成的矩陣，對 $C$ 做 SVD 分解如下
$$C=U \Sigma V^{T}$$
其中 $\Sigma$ 是特徵值(eigenvalue)為對角線的對角矩陣，保留前 $k$ 個特徵值，剩下的換成 $0$，得到 $\Sigma_{k}$，則有以下近似 $C$ 的矩陣
$$C_{k}=U \Sigma_{k} V^{T}$$
$C_{k}$ 就是新的 latent semantic space。
這個方法的缺點在於:
1. 需要很大的計算量。computational complexity: $O(mn^2)$ when $n < m$ for $n \times m$ matrix
2. 很難新增詞彙。因為每新增字詞，就就要重新計算 SVD 矩陣的 eigenvector/value，並且更新每個字的代表向量。

```
Issues:
* computationally expensive
* difficult to add new words
```

所以我們要再想想，有沒有直接用一個低維度向量代表文字的方法?

```
Idea: directly learn low-dimensional word vectors
```

把文字轉成實數所組成的向量(vectors of real numbers)，這樣的作法稱為 word embeddings。
概念上，word embedding 做的事情是把原本每個字一維的向量空間，投影到一個較低維度的(連續的)向量空間。近年來常用的 word embeddings 模型為 word2vec (Mikolov et al. 2013) 和 Glove (Pennington et al., 2014)。

## Word Embeddings 的好處
給一個語料庫(unlabeled training corpus)，給每個字一個代表向量，其向量帶有語意訊息(semantic information)的好處在於
1. 可以 cosine similarty 衡量語意的相似度
2. 詞向量(word vectors)在很多NLP tasks中是有用的、帶有語意的特徵變數
3. 可以放到類神經網路中(neural networks)並且在訓練過程中更新

## 如何找到一個字的 Word Embeddings?
### Word Embedding Model - Word2Vec & Glove
#### Word2Vec

Word2Vec 是一種以類神經網路為基礎的詞向量產生方式，主要有兩種模型，skip-gram 和 Continuous Bag of Words (CBOW)。**skip-gram** 的概念是給一個字，使用單層的神經網路架構(single hidden layer)去預測這個字的上下文(又稱neighbor)，**CBOW** 是用某個字的上下文(neighbor)去預測這個字，而其中的隱藏層就是我們想要的 word representation，也就是字的 word embedding。

![word2vec model](https://i.imgur.com/ZmeKrRt.png)

以上圖 skip-gram 為例，$x_{k}$ 是某個字的 one-hot vector，$y_{1j}, ..., y_{Cj}$ 代表預測的上下文，$C$ 是上下文的長度，依據要看多少的前後文而決定 $C$ 的大小(也就是看我們覺得這個字會受到多遠的前後文影響，憑此去訂定size)。其中 Hidden layer 是維度 $N (\ll V)$ 的結點 $h_{i}$ 所構成的隱藏層，$h = W^{T}x$ 就是字的 word embeddings [3]。

#### Word2Vec Skip-Gram

Word2Vec Skip-Gram 的作法是輸入是某個字，預測這個字的前後文(給定某個長度內)，目標是最大化給定這個字時，前後文出現的機率，

that is, maximize likelihood

$$P(w_{o1}, ..., w_{oc}|w_{I}) = \prod_{c=1}^{C}p(w_{oc}|w_{I})$$

等價於 mimize cost/loss function
$$C(\theta) = -\log P(w_{o1}, ..., w_{oc}|w_{I}) =
-\sum_{w_{I}} \sum_{c=1}^{C} \log p(w_{oc}|w_{I})$$

其中，word vector 在這個神經網路中的 hidden layer 實現，word embedding matrix (某個字對應到某個向量的 lookup table) 就是 hidden layer weight matrix。

word2vec方法的瓶頸在於 output layer 的神經元個數 (也就是 output vectors) 等同於總字彙量，如果字彙量或是corpus很大，會帶來很大的計算負擔，因此有了使用 hierarchical softmax 和 negative sampling 等方法**限制每次更新的參數數量**的想法。


large vocabularies or large training corpora -> expensive computations 
=> limit the number of output vectors that must be updated per training instance -> hierarchical softmax, sampling

##### 1. Hierarchical Softmax

Idea: compute the probability of leaf nodes using the paths

細節可參考: [類神經網路 -- Hierarchical Probabilistic Neural Network Language Model (Hierarchical Softmax)](http://cpmarkchang.logdown.com/posts/276263--hierarchical-probabilistic-neural-networks-neural-network-language-model)

##### 2. Negative Sampling (NEG)

Idea: only update a sample of output vectors

細節可參考: [Mikolov et al., “Distributed representations of words and phrases and their compositionality,” in NIPS, 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

Negative Sampling 只更新一部份的 output vectors，因此 loss function 可以改寫成

![NEG objective function](https://i.imgur.com/SvTUhon.png)

Mikolov 表示:
the task is to distinguish the target word $w_{O}$ from draws from the noise distribution $P_{n}(w)$ using logistic regression, where there are $k$ negative samples for each data sample.

What is a good $P_{n}(w)$ ?
Mikolov 表示:
We investigated a number of choices for $P_{n}(w)$ and found that the unigram distribution $U(w)$ raised to the 3/4rd power (i.e., $U(w)^{3/4}/Z$ ) outperformed significantly the unigram and the uniform distributions.

也就是說，現在還沒有科學的方法說明如何挑選 $P_{n}(w)$，不過經驗法則所找到的函數，其產生的結果的表現勝過現有的其他模型。


```
Idea: less frequent words sampled more often
```
Empirical setting: unigram model raised to the power of 3/4
![NEG empirical example](https://i.imgur.com/whvarGY.png)


#### GloVe

細節可參考: [Pennington et al., ”GloVe: Global Vectors for Word Representation ,” in EMNLP, 2014](https://nlp.stanford.edu/pubs/glove.pdf)

$P_{ij}$ 是 $w_{j}$ 出現在 $w_{i}$ 上下文裡的機率

$$P_{ij} = P(w_{j}|w_{i}) = \frac{X_{ij}}{X_{i}}$$

$X_{ij}$ 代表 $w_{j}$ 在 $w_{i}$ 的上下文裡出現的次數，$X_{i} = \sum_{k}X_{ik}$ 是出現在 $w_{i}$ 的上下文裡所有字數

$w_{i}$ and $w_{j}$ 的關係近可以以他們**同時在 $w_{k}$的上下文裡出現的機率比**作為代表。

<!-- The relationship of $w_{i}$ and $w_{j}$ approximates the ratio of their co-occurrence probabilities with various $w_{k}$ -->
$\frac{P_{ik}}{P_{jk}}$ 稱為 ratio of co-occurrence probability。

```
Idea: ratio of co-occurrence probability can encode meaning
```

$$\frac{P_{ik}}{P_{jk}} = F(w_{i},w_{j},w_{k}) = F(w_{i} - w_{j},w_{k}) = F((w_{i}w_{j})^{T} w_{k}) = \frac{F(w_{i}^{T} w_{k})}{w_{j}^{T} w_{k}}$$

令$F(x) = exp(x)$，則

$$w_{i}^{T} w_{k} = \log{P_{ik}} = \log{X_{ik}} - \log{X_{i}}$$

我們可以加上bias項 $b_{i}$ 讓 $w_{i}$ 獨立於 $k$ (?)，再加上bias項 $b_{k}$ 讓 $w_{k}$ 保持對稱(為何加bias可以使之獨立、對稱?)，得到

$$w_{i}^{T} w_{k} + b_{i} + b_{k} = \log{X_{ik}}$$

把這個問題看成迴歸式，用最小平方法(least square estimate)求解，也就是 loss function = $\sum_{i,k=1}^{V} (w_{i}^{T} w_{k} + b_{i} + b_{k} - \log{X_{ik}})^{2}$ 可以找到 $b_{i}, b_{k}, w_{i}, w_{k}$。(不確定如何計算?)

但其中還有幾個問題，其一是 log 函數會在 0 點無定義，其二是在最小平方法的 loss function 裡，每個 $(w_{i}, w_{k})$ 組合跟 $\log{X_{ik}}$ 的差距都以相等重要性看待，不會因為某組 $(w_{i}, w_{k})$ 比較常共同出現而特別看重這一組的 loss。

所以要再做一些調整，給每組 $(w_{i}, w_{k})$ 權重 $f(X_{ik})$，則 loss function 可以寫成

$$\sum_{i,j=1}^{V} f(X_{ik})(w_{i}^{T} w_{k} + b_{i} + b_{k} - \log{X_{ik}})^{2}$$

權重 $f(x) = (x/x_{max})^{\alpha}$ if $x < x_{max}$ and $f(x) = 1$ otherwise，$x_{max}, \alpha$ 是常數。

很巧的是，Pennington 等人實驗的結果發現 $x_{max} = 100, \alpha=3/4$ 時模型表現最好，跟 Mikolov 等人在 negative sampling 裡面提出的經驗是一樣的。

GloVe 的優點在於 fast training, scalable, good performance even with small corpus, and small vectors

<!-- 
## Word Embedding Evaluation
### Intrinsic Evaluation – Word Analogies
詞類比性質
Issue: different cities may have same name
Issue: can change with time
 -->
<!-- 
### Intrinsic Evaluation – Word Correlation
Comparing word correlation with human-judged scores 
Ambiguity: synonym or same word with different POSs
 -->
<!-- 
### Extrinsic Evaluation – Subsequent Task
Goal: use word vectors in neural net models built for subsequent tasks
Benefit: 
- Ability to also classify words accurately
	Ex. countries cluster together a classifying location words should be possible with word vectors
- Incorporate any information into them other tasks
	Ex. project sentiment into words to find most positive/negative words in corpus
 -->

## Implementation
Gensim: a Word2Vec Library

就算不懂上面的理論，直接在python套用Gensim也可以輕鬆得到詞向量，然後丟入模型進行操作。

## Reference
1. [Quora - What's the difference between word vectors, word representations and vector embeddings?](https://www.quora.com/Whats-the-difference-between-word-vectors-word-representations-and-vector-embeddings	)

	There are many ways to represent words in NLP / Computational Linguistics. Two prominent approaches use vectors as their representations. These are, largely speaking:

	- Distributional Semantics: represent a word with a very high-dimensional sparse vector, where each dimension reflects a context in which the word occurred in the corpus. For example, a context could be another word that appeared in proximity.

	- Word Embeddings: represent a word with a low-dimensional vector (e.g. 100 dimensions). The dimensions are usually latent, and often obtained using the information as in the distributional semantics approach (e.g. LSA, word2vec).

2. [NTU-ADLxMLDS word embedding 陳縕儂授課講義](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/doc/171016+171019_WordEmbeddings.pdf)
3. [NTHU-ML Word2Vec 吳尚鴻授課講義](http://www.cs.nthu.edu.tw/~shwu/courses/ml/labs/10_Keras_Word2Vec/10_Keras_Word2Vec.html)
the weight matrix $W$ encode a one-hot vector $x$ into a low dimensional dense vector $h$. 
Note that the weights are shared across words to ensure that each word has a single embedding. This is called weight tying. Also, word2vec is a unsupervised learning task as it does not require explicit labels. An NN can be used for both supervised and unsupervised learning tasks.

4. [Word2Vec Skip-Gram Visualization](https://ronxin.github.io/wevi/)

5. [使用TENSORFLOW實作WORD2VEC CBOW](https://vinachangblog.wordpress.com/2017/08/13/%E4%BD%BF%E7%94%A8tensorflow%E5%AF%A6%E4%BD%9Cword2vec-cbow/)

6. Word2Vec 相關論文 
    [1] Mikolov T, Chen K, Corrado G, et al. [Efficient Estimation of Word Representations in Vector Space](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). Computer Science, 2013.（这篇文章就讲了两个模型：CBOW 和 Skip-gram） 
    [2] Mikolov T, Sutskever I, Chen K, et al.  [Distributed representations of words and phrases and their compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), 2013.（这篇文章针对Skip-gram模型计算复杂度高的问题提出了一些该进） 
    [3] [Presentation on Word2Vec](https://docs.google.com/file/d/0B7XkCwpI5KDYRWRnd1RzWXQ2TWc/edit) （这是NIPS 2013workshop上Mikolov的PPT报告）

7. 針對GloVe提出的改進: [Simpler GloVe](https://zhuanlan.zhihu.com/p/31558664)
