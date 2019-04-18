# ctr_Keras
很简单的ctr模型实现，欢迎指出bug&提出宝贵意见！

## 模型
LR

FNN

Wide&Deep：https://arxiv.org/abs/1606.07792

IPNN：https://arxiv.org/abs/1611.00144

DCN：https://arxiv.org/abs/1708.05123

NFM：https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf

DeepFM：https://arxiv.org/abs/1703.04247

NFFM


## 数据集

kaggle-criteo-2014 dataset

http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

数据集按9:1划分，请自行划分

train | test
-|-
41256556|4584063


## 预处理
连续型特征(13)：缺失值补0，离散分桶

离散型特征(26)：过滤频率低于10的特征值

## 执行步骤
运行preprocess.py生成train.csv和test.csv文件
```
python preprocess.py
```
运行相应的ctr模型代码文件，如
```
python lr.py
```

## 结果
一般ctr模型都是一轮收敛，这里训练轮数统一只取一轮，优化器使用Adam(0.001)

由于预处理、参数，实验结果跟论文会有出入

Model | 测试集Logloss
-|-
LR|0.48736
FNN|0.45586
W&D|0.47389
DCN|0.45565
DeepFM(weight)|0.45592
DeepFM(weight+first_order)|0.45544
DeepFM(add+first)|0.49038
NFM(concat(dot,embedding))|0.45394
NFM(sum(multiply))|0.45952
NFM(concat(multiply))|0.45332
NFM(conat(multiply)+first_order)|0.4713
NFFM(concat(multiply))|待补充
NFM(conat(multiply)+first_order)|待补充


分析：
```
这里W&D线性部分使用了全部的feature，可能是这个原因效果比FNN差

DeepFM原始的结构(add)效果很差，做了点改变，从直接加和变成加权加和(weight)，用一个Dense layer实现

NFM做了很多变种尝试，发现concat二阶交互向量效果最好

从结果上来看一阶线性连接没有带来明显提升（W&D相对于FNN，DeepFM(weight)相对于DeepFM(weight+first_order)，NFM(concat(multiply))相对于NFM(conat(multiply)+first_order)）

DCN的cross部分的线性连接没有带来明显提升（相对于FNN）

把二阶交互加入到dnn里效果表现比较好（NFM(concat(multiply))、NFM(concat(dot,embedding))相对于deepFM，FNN）
```
关于NFFM:
```
NFFM参数量是NFM的field_size(39)倍，训练速度慢而且容易OOM，做以下两种优化，4千万样本约3个半小时跑完

参数量减少：连续型特征和离散型特征不交叉

速度加快：减少embedding lookup次数（这部分还可以进一步优化），减少multiply次数,增大batch size
```

## 待补充
- [ ] PNN、AFM、xDeepFM

- [ ] 数据过大需要用tfrecord和多线程队列减少io时间

- [ ] 二阶交互部分依然比较耗时，这部分需要重写优化下
 
- [ ] 精度缩小到float16或者int16，进一步节省空间



