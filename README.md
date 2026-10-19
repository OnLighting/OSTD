# RASA-Net

**Paper:** RASA-Net: Adaptive Receptive Fields and Scale-Balanced Label Assignment for Remote Sensing Small Object Detection

**Author:** lxh &lt;2356338071@qq.com&gt;

## 数据集下载

以下数据集通过夸克网盘分享，点击链接并输入对应提取码即可获取。

| 数据集 | 下载链接 | 提取码 |
| --- | --- | --- |
| SODA-A | [SODA-A.zip](https://pan.quark.cn/s/a22db407908d?pwd=qAP1) | `qAP1` |
| HRRSD | [HRRSD.zip](https://pan.quark.cn/s/5565eee34e88?pwd=Y6C9) | `Y6C9` |
| DIOR | [DIOR.zip](https://pan.quark.cn/s/dd446ecb35b2?pwd=L7Zm) | `L7Zm` |
| VisDrone | [VisDrone.zip](https://pan.quark.cn/s/1e770ae2572e?pwd=UvYu) | `UvYu` |

## 运行命令

以 DIOR 数据集训练 RASA-Net 为例：安装项目依赖并准备好 COCO 格式的数据后，在项目根目录的 Bash 终端中运行以下命令（需要 3 张 CUDA GPU）：

```bash
bash tools/dist_train.sh configs/ostd/ostd_dior_1x.py 3 work_dirs/rasa_net_dior --seed 0
```

数据默认位于 `data/DIOR_coco/`，训练标注为 `annotations/instances_trainval.json`，训练图片路径相对于 `images/`；验证使用 `annotations/instances_test.json` 和 `images/test/`。如数据存放位置不同，请修改配置中的 `ann_file` 和 `img_prefix`。训练日志和模型权重将保存到 `work_dirs/rasa_net_dior/`。
