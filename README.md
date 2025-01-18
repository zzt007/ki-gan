# KI_GAN

## log 
暂告一段落，目前写完了初稿，待审阅后再做其它

### 250118 👇
```
eval昨天的6s-9s的消融模型
75pt: 0.082 \ 0.197
74pt: 0.106 \ 0.244
70pt: 0.090 \ 0.211
68pt: 0.095 \ 0.225
65pt: 0.087 \ 0.209
```

### 250117 👇
```
eval昨天的vel_type消融
75pt: 0.049 \ 0.109
72pt: 0.055 \ 0.124


开始6s-9s的vel_type消融模型训练
```

### 250116 👇
```
eval昨天的vel消融
75pt: 0.046 \ 0.103
72pt: 0.048 \ 0.107
70pt: 0.056 \ 0.123
71pt: 0.078 \ 0.170
先做类型的消融吧，到时候再补这个vel消融的6s-9s模型
晚上19.跑完了，但是我还没有eval
论文里我个人觉得不需要逐一移除来做消融，直接用vel_type一起消融
```

### 250115 👇
```
中午12.开始eval昨天交互中vel的消融
75pt: 0.084 \ 0.201
72pt: 0.078 \ 0.185
71pt: 0.084 \ 0.200

晚上21.58 eval 中午训练的eval的消融6s-6s
72pt: 0.052 \ 0.114
74pt: 0.045 \ 0.102
浪费一天，因为发现没有改到vel的消融，晕,今天的这些结果都没用

晚上22.15开始训练vel的消融
```

### 250114 👇
```
早上9.，开始测试昨天去除type的模型eval
72pt: 0.047 \ 0.106
74pt: 0.054 \ 0.118
75pt: 0.056 \ 0.123
65pt: 0.093 \ 0.219
61pt: 0.106 \ 0.240

9.45测试完后，开始训练wo_type的6-9s模型

晚上20.26，完成训练，开始eval
75pt: 0.083 \ 0.202 (这比我最好的还要好啊，过拟合了吗？)
74pt: 0.087 \ 0.211
72pt: 0.101 \ 0.230
eval完后，要开始对交互建模的消融实验了
交互建模的消融安排如下：
1、去除velocity的引入，仅考虑相对距离和type
2、去除velocity和type的引入，仅考虑相对距离

今天先做1，看看修改多不多

在models_ablation_interaction_wo_velocity.py中的474行，我的模型里是有三因素concat的，做消融
464行，从encoder_h_dim * 3 变为 * 2
```

### 250113 👇
```
早上9.35，完成对昨天去除state_encoder的eval
其中，72pt是 0.132 \ 0.296
74pt 是 0.137 \ 0.306
75pt 是 0.242 \ 0.499
选用72pt结果

上午10.40 开始训练该模型的6-6s版本
晚上21.30 eval了72、74、75pt的结果
72pt: 0.058 \ 0.129
74pt: 0.062 \ 0.137
75pt: 0.062 \ 0.135
再把wo_type模型的6-6s版本挂上
```

### 250112 👇
``` 
早上9.30左右，完成了对昨天的eval，并且开始对下一个的eval，eval完后开始训练wo_spectral的6-9s模型
上午10.30，开始训练
晚上20.48 训练完毕，开始eval
72pt: 0.092 \ 0.217
75pt: 0.101 \ 0.232

然后eval完再训一版，wo_motion的6-9s模型
由于是在原始编码器里修改，所以需要多改一处
720行 combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding],dim=2)
改为 除去 state_encoding
然后740行，对combined_encoding进行reshape的时候，是view(-1, self.encoder_h_dim * 2)

```

### 250111 👇
```
今天把6-6s的交通灯信息消融也做了，但还没来得及eval
先训一版wo_spectral的，6-6s


```

### 250110 👇
```
开始做消融实验，命名逻辑为 models_ablation_wo_xxx.py，其中wo代表without，xxx代表去掉的编码模块
今天下午17.左右开始训练去掉trafficlight的，models_ablation_wo_trafficlight.py。6s-9s
预计需要更改：
    1、池化出来的拼接部分 对应 738行
    2、decoder部分的输入维度 对应 629行

```

### 241209 👇
```
修改模型输出图片的线型和色彩,末尾箭头还没来得及修改完
/root/trajectory_prediction/ki-gan/scripts/detect_with_map_arrow.py
```

### 241208 👇
```
昨晚停电了，只保存到29pt，看看效果
重新训练了一版12-18的，
75pt: ADE = 0.095, FDE = 0.223 , 原文结果是 ADE = 0.108, FDE = 0.258
74pt: ADE = 0.095, FDE = 0.224
保存这两个模型名为：12-18hiddenPlusType_checkpoint_with_model_75.pt

```

### 241207 👇
```
晚上19.53，重新测试了一版12-12的
75pt: ADE: 0.047, FDE: 0.105, 我原来的是0.048,0.101
74pt:  ADE: 0.046, FDE: 0.102
今晚回去之前，再训练一版12-18的看看效果
```


```
测试效果：
75pt: ADE = 0.074, FDE = 0.144 
74pt: ADE = 0.072, FDE = 0.142
72pt: ADE = 0.054, FDE = 0.111
71pt: ADE = 0.064, FDE = 0.130
总的来说，还是不太行的效果，想想还能怎么改进
改进尝试：在pooling里的隐藏状态中不和rel_embedding拼接，而是和异质性type_embedding, 398 行和 517 行
```

### 241206 👇
```
20:50完成测试
75pt测试效果糟糕，预测时域12frames，ADE = 0.082, FDE = 0.159，较原文有较大下降，需要进一步优化模型；
但是74pt测试效果还可以，预测时域12frames，ADE = 0.054, FDE = 0.111，较原文略有提升，为什么两个相邻保存的模型效果差别这么大，需要怎么改进呢？
先在pooling里改个dropout = 0.1, 训练一版明天来看
```

```
13.18分开始训练
在pool中添加交通参与者type，凸显异质性；具体修改在models_add_type_in_pool.py文件中
```

### 241117 👇
``` 
1、将模型评估指标从2位小数转为3位小数，为此需要得到12-12的模型（因为之前训的都是12-18）的 -- 已训练好：原论文中的结果是 ADE = 0.056, FDE = 0.117；我的是ADE = 0.048，FDE = 0.101, 从数值上各自提升了 14.286% 和 13.675%
2、针对原文模型的eval，重新区分paper_evaluate.py 和 paper_models.py；前者放在scripts下了，后者在ki-gan下，用的时候，把from kigan.那里进行相应修改（从models 到 paper_models）即可
3、还需要探明如何使用GAN来表示多模态轨迹，目前看起来是single
```

### 241116 👇
```
1、eval了昨晚训练一版obs_len = 12, pred_len = 12的，ADE = 0.05, FDE = 0.10,取的效果较原文好一些，原文指标为ADE = 0.05, FDE = 0.12, 只从数值上看，FDE 提升16.67%，但还需要思考为什么ADE没有提升
2、今天把可视化地图以及轨迹的线型要修改修改
```

### 241115 👇
```
1、已eval上午训练的，目前eval的结果是比之前好的，ADE = 0.10，FDE = 0.23，原文指标是ADE = 0.11, FDE = 0.26
2、将对应的这个模型也上传至github，位置ki-gan/models
3、训练一版obs_len = 12, pred_len = 12的
```

```
1、已eval昨天晚上训练的，大概需要9小时左右的训练时间，效果比原文要差，ADE=0.16,FDE=0.28；
2、将spectral信息不拼接进入attention pool中，直接和traffic一样再拼接试试，开始训练241115-上午10.32
```

### 241114 👇
```
1、初步添加spectral encoder，尚未对齐维度；由于cuda torch.fft总是报错，先使用cpu下的numpy实现fft
2、将数据集暂时放到kigan文件夹下，方便debug时更快进入调试
```

### 241113 👇
```
创建新分支dev，用于记录添加spectral信息后相应的模型修改
1、增加spectral encoder
2、对齐维度
```

### 241111 👇
### 上传数据集预处理文件
```
dataPreprocess.py # 处理单个文件
dataBatchPreprocess.py # 处理整个天津数据集，需要从main()中修改对应的文件夹地址
```
### 添加DEBUG模式,数据转移至CPU上运行
```
train.py
1.需要将train.py的args中--use_gpu设为0
2.需要将train.py的discriminator和generator中的batch = [tensor.cuda() for tensor in batch]进行修改
3.在eval时batch = [tensor.cuda() for tensor in batch]进行修改

models.py
1.需要将models.py中的各类encoder中的init_hidden方法中的全置为cpu()
2.在decoder部分,有一个decoder_c也需要置为cpu

后续如有需要,可添加快捷转换方式
```

### 241108 belike👇
.
├── 01.png
├── 02.png
├── 03.png
├── LICENSE
├── README.md
├── datasets
│   └── Tianjin
│       ├── train
│       │   └── basic_info_traffic_state.csv
│       └── val
│           └── basic_info_traffic_state.csv
├── kigan
│   ├── __pycache__
│   │   ├── losses.cpython-38.pyc
│   │   ├── models.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   ├── data
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── loader.cpython-38.pyc
│   │   │   └── trajectories.cpython-38.pyc
│   │   ├── loader.py
│   │   └── trajectories.py
│   ├── losses.py
│   ├── models.py
│   └── utils.py
└── scripts
    ├── dataPreprocess.py
    ├── dataset
    │   └── Tianjin
    │       └── train
    │           └── basic_info_traffic_state.csv
    ├── detect_with_map.py
    ├── evaluate_model.py
    ├── map.osm
    └── train.py
