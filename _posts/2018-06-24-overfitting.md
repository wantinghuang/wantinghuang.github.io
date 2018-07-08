---
layout: post
title: Overfitting
---


<!-- # Overfitting-->

<!-- ## What is overfitting? -->
## 什麼是 Overfitting?

過度擬和，指訓練的模型過度貼近 training data，導致模型套用到 training data 的時候 error 變大。又稱為過度訓練(overtraining)。

<!-- more -->

## 什麼樣的情況是 Overfitting? 

![under-overfitting](https://i.ytimg.com/vi/dBLZg-RqoLg/maxresdefault.jpg){:height="80%" width="80%"}

以回歸(多項式)模型為例，我們相信的趨勢是綠色的那條線，擬和出來的模型是紅色的線，右邊的圖雖然能夠準確預測每一個樣本點，卻複雜到難以為新的樣本點做預測，或說，我們很難相信右邊的紅色模型預測出來的值會是準確的。

以分類模型為例，底下的線為分類模型要找的判斷邊線/準則，則可以看到左邊是還不算有效分類，準確率還不夠好；，右邊則是過渡擬和，雖然可以完全將training data分對，對於新的資料卻不見得能符合，舉例來說，兩個ＸＸ的中間要是多了個Ｏ，按照圖上的分類準則，就會被分到Ｘ那群，也就分錯了，此外，分類時的運算量也會很大；中間算是分得不錯，其中有兩個分錯的是一種必須的犧牲，把他們當成立外，才有辦法找到一個對大部分人合用的準則



<!-- ## Why is it important? -->
## Overfitting 為什麼重要?

DeepMind 的 Aja Huang 曾經在 2018 Google AI論壇提到：
```
AlphaGo跟李世乭對弈的第四局，AlphaGo輸了，後來 DeepMind 團隊發現原因是模型 overfitting 。
```
由此可見模型的過度訓練會使得模型對未知世界的預測失準。


<!-- ## How to determine if it is overfitting? -->
## 如何判斷是否有 Overfitting?

<!-- PLOT error curve for training and testing data. -->
如果隨著模型訓練的時間，training data 的 error 下降，但 testing data 的 error 卻上升，則懷疑模型 overfit.
因此，最簡單的方式是畫出 training 和 testing data 的 error curve (如下圖)來觀察。
![圖](https://qph.ec.quoracdn.net/main-qimg-39f72925e85c26e105b14ab276206747)

<!-- ## How to solve overfitting? -->
## 如何解決 Overfitting?

1. 對多項式模型，降階，不要用高次項去擬和。降階到哪個地步，有方法檢查。
<!-- don't use high order model -->
2. regularization。像 LASSO、Ridge regression 都是用來限制解空間、解決 overfitting 的好工具。
<!-- ## Why regularization can solve overfitting? -->
3. 將資料分堆，做 dropout 和 cross validation。

<!-- 4. decision tree 做 feature selection -->



## 為什麼正規化(regularization)可以解決 Overfitting?

關於L1 norm與L2 norm的比較，請看[這篇](https://wantinghuang.github.io/2018/04/10/l1-l2norm/)


<!-- ## Implementation of solving overfitting
## 解決 overfitting 的實作
python
in tensorflow
in sklearn
-->
