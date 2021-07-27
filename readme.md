LinUcb 算法的一个简单验证版本

### 数据集

    moivelen

### 用户与物品特征

    使用的是side info embedding
    这里的 user embedding 维度32, item embedding 维度32
    context feature 是将user_embedding*item_embedding 维度32
    torch的代码位于 src/main/python

### 使用到的工具

    nd4j 矩阵运算包
    torch 神经网络

### 数据位置

    src/main/resources
    item.npz 物品 embedding
    user.npz 用户 embedding

### 运行

    src/java/src/main/simulation/MovieSimulation.java

### 一些注意事项

    探索参数的选择应大一些，因为这里的特征是embedding特征, x_t * x 的数值会很小，这样更新矩阵A与向量b的值就会非常小
    思考: 
    a 是否可以再将物品与用户的 embedding向量哈希向量化? 这样 x_t * x 的数值就会大
    b 冷启动的用户和物品的 id embedding 使用最相似的k个用户或者物品的均值替代是否可以?
    c 线上部署过程当中应该使用什么样的架构方式实现
    d 物品数目很多的情况下，存储的As矩阵与InvAs矩阵占用内存空间很大
    e 实际线上部署的过程 如何更新这些参数变量
    f 奖赏值的应该如何设置

如果你有好的建议欢迎讨论 qq: 55058629 微信: _zhangluoyang_

### 训练过程的日志打印如下:
    # 100轮训练后基本收敛
    start training....
userNum: 100 <br>
itemNum: 1682 <br>
epoch: 0 positiveReward:0.575 time: 20 <br>
epoch: 1 positiveReward:0.245 time: 20 <br>
epoch: 2 positiveReward:0.85 time: 19 <br>
epoch: 3 positiveReward:0.565 time: 19 <br>
epoch: 4 positiveReward:0.135 time: 19 <br>
epoch: 5 positiveReward:0.35 time: 19 <br>
epoch: 6 positiveReward:0.37 time: 19 <br>
epoch: 7 positiveReward:0.85 time: 19 <br>
epoch: 8 positiveReward:0.61 time: 19 <br>
epoch: 9 positiveReward:0.73 time: 19 <br>
epoch: 10 positiveReward:0.58 time: 19 <br>
epoch: 11 positiveReward:0.825 time: 19 <br>
epoch: 12 positiveReward:0.92 time: 19 <br>
epoch: 13 positiveReward:0.61 time: 19 <br>
epoch: 14 positiveReward:1.45 time: 19 <br>
epoch: 15 positiveReward:1.215 time: 19 <br>
epoch: 16 positiveReward:1.115 time: 19 <br>
epoch: 17 positiveReward:0.92 time: 19 <br>
epoch: 18 positiveReward:1.405 time: 19 <br>
epoch: 19 positiveReward:0.925 time: 19 <br>
epoch: 20 positiveReward:1.3 time: 19 <br>
epoch: 21 positiveReward:1.355 time: 19 <br>
epoch: 22 positiveReward:1.94 time: 19 <br>
epoch: 23 positiveReward:1.535 time: 19 <br>
epoch: 24 positiveReward:1.22 time: 19 <br>
epoch: 25 positiveReward:1.09 time: 19 <br>
epoch: 26 positiveReward:1.155 time: 19 <br>
epoch: 27 positiveReward:1.48 time: 19 <br>
epoch: 28 positiveReward:1.585 time: 19 <br>
epoch: 29 positiveReward:1.625 time: 19 <br>
epoch: 30 positiveReward:1.61 time: 20 <br>
epoch: 31 positiveReward:1.715 time: 19 <br>
epoch: 32 positiveReward:1.64 time: 19 <br>
epoch: 33 positiveReward:1.495 time: 19 <br>
epoch: 34 positiveReward:1.61 time: 19 <br>
epoch: 35 positiveReward:1.825 time: 19 <br>
epoch: 36 positiveReward:2.35 time: 19 <br>
epoch: 37 positiveReward:1.895 time: 19 <br>
epoch: 38 positiveReward:1.95 time: 19 <br>
epoch: 39 positiveReward:1.565 time: 19 <br>
epoch: 40 positiveReward:1.565 time: 20 <br>
epoch: 41 positiveReward:2.33 time: 19 <br>
epoch: 42 positiveReward:1.87 time: 20 <br>
epoch: 43 positiveReward:2.355 time: 19 <br>
epoch: 44 positiveReward:2.015 time: 19 <br>
epoch: 45 positiveReward:2.105 time: 19 <br>
epoch: 46 positiveReward:2.205 time: 19 <br>
epoch: 47 positiveReward:2.625 time: 19 <br>
epoch: 48 positiveReward:2.47 time: 19 <br>
epoch: 49 positiveReward:2.475 time: 19 <br>
epoch: 50 positiveReward:2.27 time: 19 <br>
epoch: 51 positiveReward:2.27 time: 19 <br>
epoch: 52 positiveReward:2.295 time: 19 <br>
epoch: 53 positiveReward:2.5 time: 19 <br>
epoch: 54 positiveReward:2.85 time: 19 <br>
epoch: 55 positiveReward:2.265 time: 19 <br>
epoch: 56 positiveReward:2.265 time: 19 <br>
epoch: 57 positiveReward:2.37 time: 19 <br>
epoch: 58 positiveReward:2.91 time: 19 <br>
epoch: 59 positiveReward:2.955 time: 19 <br>
epoch: 60 positiveReward:2.905 time: 19 <br>
epoch: 61 positiveReward:2.765 time: 19 <br>
epoch: 62 positiveReward:3.155 time: 19 <br>
epoch: 63 positiveReward:3.56 time: 19 <br>
epoch: 64 positiveReward:3.07 time: 19 <br>
epoch: 65 positiveReward:3.0 time: 19 <br>
epoch: 66 positiveReward:3.1 time: 19 <br>
epoch: 67 positiveReward:3.365 time: 19 <br>
epoch: 68 positiveReward:3.31 time: 19 <br>
epoch: 69 positiveReward:3.505 time: 19 <br>
epoch: 70 positiveReward:3.51 time: 19 <br>
epoch: 71 positiveReward:3.505 time: 19 <br>
epoch: 72 positiveReward:3.46 time: 19 <br>
epoch: 73 positiveReward:3.615 time: 19 <br>
epoch: 74 positiveReward:3.765 time: 19 <br>
epoch: 75 positiveReward:3.815 time: 19 <br>
epoch: 76 positiveReward:3.815 time: 19 <br>
epoch: 77 positiveReward:3.97 time: 19 <br>
epoch: 78 positiveReward:3.97 time: 19 <br>
epoch: 79 positiveReward:4.015 time: 19 <br>
epoch: 80 positiveReward:4.47 time: 19 <br>
epoch: 81 positiveReward:4.625 time: 19 <br>
epoch: 82 positiveReward:4.72 time: 18 <br>
epoch: 83 positiveReward:4.77 time: 19 <br>
epoch: 84 positiveReward:4.87 time: 19 <br>
epoch: 85 positiveReward:4.87 time: 19 <br>
epoch: 86 positiveReward:4.78 time: 19 <br>
epoch: 87 positiveReward:4.82 time: 19 <br>
epoch: 88 positiveReward:4.825 time: 19 <br>
epoch: 89 positiveReward:4.87 time: 19 <br>
epoch: 90 positiveReward:4.865 time: 19 <br>
epoch: 91 positiveReward:5.02 time: 19 <br>
epoch: 92 positiveReward:5.07 time: 19 <br>
epoch: 93 positiveReward:4.975 time: 19 <br>
epoch: 94 positiveReward:5.07 time: 19 <br>
epoch: 95 positiveReward:5.27 time: 19 <br>
epoch: 96 positiveReward:5.17 time: 19 <br>
epoch: 97 positiveReward:5.07 time: 19 <br>
epoch: 98 positiveReward:5.12 time: 19 <br>
epoch: 99 positiveReward:5.22 time: 19 <br>
epoch: 100 positiveReward:5.17 time: 19 <br>
epoch: 101 positiveReward:5.22 time: 19 <br>
epoch: 102 positiveReward:5.22 time: 19 <br>
epoch: 103 positiveReward:5.17 time: 19 <br>
epoch: 104 positiveReward:5.22 time: 19 <br>
epoch: 105 positiveReward:5.12 time: 19 <br>
epoch: 106 positiveReward:5.12 time: 19 <br>
epoch: 107 positiveReward:5.12 time: 19 <br>
epoch: 108 positiveReward:5.12 time: 19 <br>
epoch: 109 positiveReward:5.12 time: 19 <br>
epoch: 110 positiveReward:5.12 time: 19 <br>
epoch: 111 positiveReward:5.12 time: 19 <br>
epoch: 112 positiveReward:5.12 time: 19 <br>
epoch: 113 positiveReward:5.12 time: 19 <br>
epoch: 114 positiveReward:5.12 time: 19 <br>
epoch: 115 positiveReward:5.12 time: 19 <br>
epoch: 116 positiveReward:5.12 time: 19 <br>
epoch: 117 positiveReward:5.12 time: 19 <br>
epoch: 118 positiveReward:5.12 time: 19 <br>
epoch: 119 positiveReward:5.12 time: 19 <br>