# KI_GAN
## 上传数据集预处理文件
```
dataPreprocess.py # 处理单个文件
dataBatchPreprocess.py # 处理整个天津数据集，需要从main()中修改对应的文件夹地址
```
## 添加DEBUG模式,数据转移至CPU上运行
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

# 241108 belike👇
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
