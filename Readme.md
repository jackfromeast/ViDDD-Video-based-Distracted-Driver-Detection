## 驾驶行为分析及预警系统

## 1 问题阐述

### 1.1 危险驾驶行为分类

+ 分心驾驶（distracted driving）
+ 疲劳驾驶（fatigue driving）
+ 酗酒驾驶
+ 路怒症

### 1.2 存在的问题/目的

1. 检测速度，低时延，尽量达到实时监测

2. 在许多运营车辆中，手机往往被固定在车辆内饰中，司机往往不会手持手机。所以对司机视线的监测或者建模十分重要。司机正常驾驶的姿态是常态，通过此可以判断其他的非正常驾驶行为。

3. 仅基于图片的检测方案无法准确检测司机的行为的前导动作，如司机去拿手机；并且忽略动作的整体将导致识别准确率下降。尽管当前基于图片的检测方案已有较高的准确率，其十分依赖于关键帧，而忽略了司机的其他连贯动作。

   


## 2 相关研究

### 2.1 数据集

1. https://abouelnaga.io/projects/auc-distracted-driver-dataset/ 更全 邮件获取

   The Distracted Driver’s dataset is collected using an ASUS ZenPhone (Model Z00UD) rear camera. The input was collected in a video format, and then, cut into individual images, 1080×19201080×1920 each. The phone was fixed using an arm strap to the car roof handle on top of the front passenger’s seat. In our use case, this setup proved to be very flexible as we needed to collect data in different vehicles. All of the distraction activities were performed without actual driving in a parking spot.

   视频截取的图片，可能不包含原视频，待确认。

2. https://www.kaggle.com/c/state-farm-distracted-driver-detection/data 比赛数据集 10个类别 4GB

   仅为图片，不包含视频。

3. https://dmd.vicomtech.org

   annotation example:

   ```
   "vcd":{
   	"frames":{
   		"0":{
   			"objects":{"0":{}},
   			"contexts":{"0":{}},
   			"actions":{"0":{
   							"action_data":
   							{"text":[{"name":"annotated","val":"unchanged"}]}}}
   			},
   ```

   

### 2.2 论文

transformer：

https://www.jiqizhixin.com/articles/2020-10-05-4

https://www.jiqizhixin.com/articles/2020-05-28-9?fbclid=IwAR3ZMsqDOk5MDBUaGREMEiEMd05ucDViOwALWxGjBgBwdbMeGvlCfKDximg

其他见论文汇总

### 2.3 其他

可以参考的模型（任务-成熟的模型）

https://www.paddlepaddle.org.cn/modelbase





## 3 方法

What we are going to do is distracted driver detection. On the embedded system side, we would use the camera to record the driver's behavior, and dump the video every 5s or 10s. Our current solution is to extract frames of this video under a proper time step in Raspberry Pi, and then send these packaged images to the server side as the raw input for the model prediction. The problem on this side is how to achieve a near real-time dectection. Because the Pi need to wait for the model's results and warn the driver through the buzzer.

### 3.1 数据集处理

使用DMD提供的标注工具：

https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/blob/master/docs/setup_linux.md

对于视频的处理使用FFmpeg库，下载安装时记得让git走全局代理。



### 3.2 模型

视频动作模型思考策略：

+ **使用一个网络来捕捉时空信息还是使用两个网络分别捕捉时间信息和空间信息**
+ **多片短的融合预测**
+ **是端到端的训练还是特征提取与分类分开进行**

https://zhuanlan.zhihu.com/p/81089256



## 目录结构
```
DDD
├── code
│   ├── __pycache__
│   │   └── mypath.cpython-38.pyc
│   ├── dataLoaders
│   │   ├── DMD-labels.json
│   │   ├── pic_dataset.py
│   │   └── video_dataset.py
│   ├── models
│   │   └── vit_base_patch16_224_in21k.pth
│   ├── mypath.py
│   ├── network
│   │   └── vit_model.py
│   ├── process_raw_dataset.py # processed raw dataset and save them into processed_dataset
│   ├── train.py
│   └── utils.py
|
|
├── data	# data ready for train
│   └── DMD-clips-70
│       ├── train
│       └── val
|
|
├── processed_dataset	# raw dataset after processed, but not ready for train
│   ├── DMD-clips-70
│   │   ├── change_gear
│   │   ├── drinking
│   │   ├── hair_and_makeup
│   │   ├── phonecall_left
│   │   ├── phonecall_right
│   │   ├── radio
│   │   ├── reach_backseat
│   │   ├── reach_side
│   │   ├── safe_drive
│   │   ├── standstill_or_waiting
│   │   ├── talking_to_passenger
│   │   ├── texting_left
│   │   ├── texting_right
│   │   └── unclassified
│   ├── DMD-clips-70.zip
│   ├── driver_imgs_list.csv
│   └── imgs
│       └── train
...
```






