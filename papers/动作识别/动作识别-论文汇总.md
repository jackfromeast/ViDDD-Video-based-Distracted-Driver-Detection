## 动作识别-论文汇总

示例：

```
01 论文名称
+ 数据集
+ 方法
+ 备注
```

### 01 ARID: A Comprehensive Study on Recognizing Actions in the Dark and A New Benchmark Dataset

+ 时间：2021. 07 arXiv.

+ 数据集

  ARID 自建的暗场景下bench mark数据集 3784 video clips 8721s

+ 方法或贡献

  - 通过尝试了5种帧增强方案，对比了真实暗场景拍摄（real dark videos）和已有数据集处理变暗的异同，得出ARID数据集的必要性

  - 尝试了应用不同的帧增强方案的Two-stream model and 3D-CNN based model分别在Sunthetic dark dataset 和 ARID dataset上的实验。

  + 用CAMs（Learning deep features for discriminative localization）做结果验证

