# 官方 Recall/FDR 模型选择与两阶段训练验证设计

## 目标

将当前以 COCO `bbox_mAP` 保存 best、按 6:2:2 划分并在独立 test 集评测的流程，改造成与比赛官方口径一致的完整训练链路：官方数据集按 8:2 划分为训练集和验证集；第一阶段在官方训练集训练；第二阶段以第一阶段 best checkpoint 为起点，使用官方训练集与 ShipRS 补充数据混合微调；两个阶段都按官方 Recall/FDR 在官方验证集上选择 best；最终也只在验证集及其模拟 10000×10000 大图上报告指标和最大推理时间。

## 官方评测口径

### 类别与 IoU

- 舰船：`HM`、`LQS`、`QHS`、`MS`，匹配 IoU 为 0.50。
- 飞机：`A1_SU-35` 至 `A20_SU-24`，匹配 IoU 为 0.50。
- 车辆：`FSC`，匹配 IoU 为 0.35。
- 每张图、每个子类内部按 score 降序进行一对一贪心匹配；重复检测计 FP，未匹配真值计 FN。

### 汇总方法

先计算 25 个子类各自的 Recall 和 FDR，再在每个大类内部对其子类做算术平均，最后对舰船、飞机、车辆三个大类再做算术平均：

- `official_recall = mean(ship_recall, aircraft_recall, vehicle_recall)`
- `official_fdr = mean(ship_fdr, aircraft_fdr, vehicle_fdr)`

同时保留 TP/FP/FN 合并计算得到的 merged Recall/FDR，便于排查类别平均口径和总体计数口径的差异，但 best checkpoint 只依据 official 指标选择。

### 固定分类阈值

每个类别可以有不同 score threshold。阈值直接固化为 `ret/threshold_search_fdr_0.19_selected.csv` 中的 25 个值，不在训练、微调或最终验证后调用 `search_recall_fdr_thresholds.py`。该 CSV 对应的目标 FDR 为 0.19，为官方 0.20 门槛保留 0.01 余量。

固定阈值放在一个可导入的 Python 常量模块中，训练期评测、普通验证推理、模拟大图推理和指标报告共同引用，防止多处手写后产生差异。CSV 仅作为阈值来源和审计依据，运行时不依赖 CSV 文件。

## Best checkpoint 选择

新增官方口径 best 保存 hook，读取每轮验证后写入日志缓冲区的 `official_recall` 和 `official_fdr`。保存文件固定为 `best_official_recall_fdr.pth`，并在同目录 `best_official_recall_fdr.json` 中记录 epoch、Recall、FDR 和是否双达标，同时写入训练日志。

候选模型按以下顺序比较：

1. `official_recall >= 0.85` 且 `official_fdr <= 0.20` 的双达标模型，优先级高于任何未双达标模型。
2. 尚未出现双达标模型时，只比较 Recall，Recall 更高者覆盖 best，不因 FDR 改变选择。
3. 已有双达标模型后，只允许新的双达标模型参与覆盖。
4. 两个双达标模型的 Recall 绝对差大于 0.005 时，Recall 更高者优先。
5. 两个双达标模型的 Recall 绝对差不大于 0.005 时，FDR 更低者优先。
6. Recall 和 FDR 都相同或处于浮点容差内时保留较早 checkpoint，避免无意义复制。

COCO mAP 继续输出用于模型诊断，但不参与 best 保存。早停逻辑同步监控官方指标的同一比较策略，避免 best 已按官方口径更新而训练却因 mAP 停止。

## 组件与数据流

### 共享官方评测模块

新增一个与训练入口无关的纯指标模块，职责为：

- 保存 25 类名称、大类映射、分类 IoU 和固定 score threshold。
- 将 mmdet 的逐图分类检测数组转换为统一预测记录。
- 执行 score 过滤、一对一匹配以及 TP/FP/FN 统计。
- 返回每个子类、每个大类、official 和 merged 指标。

`AircraftDataset.evaluate()` 在请求 `official` metric 时调用该模块，并将关键标量以稳定 key 写回 EvalHook，例如 `official_recall`、`official_fdr`、`ship_recall` 和 `ship_fdr`。命令行评测脚本也调用同一模块，避免训练与最终报告出现两套实现。

### 数据集 8:2 划分

`tools/split_val.py` 改为两路分层划分：

- 默认比例为 train 0.8、val 0.2。
- 只创建和维护 `images/train`、`labels/train`、`images/val`、`labels/val`。
- 稀有类保护和训练集最低样本保障继续保留，但删除 test 相关集合、复制、统计和告警分支。
- 覆盖运行前仍从完整原始训练池恢复，确保重复执行不会在已切分数据上再次切分。

mmdet 配置中的 `data.val` 指向验证集。为兼容 `tools/test.py` 依赖的 `data.test` 字段，`data.test` 也指向同一验证集路径；它只是接口别名，不代表存在第三个数据子集。

### 两阶段训练

提供一个从项目根目录执行的完整 shell 脚本：

1. 校验官方数据和 ShipRS 数据的必要路径。
2. 备份或恢复官方原始训练池。
3. 按 8:2 生成官方 train/val 并转换为 COCO JSON。
4. 运行第一阶段官方数据训练。
5. 取得第一阶段 `best_official_recall_fdr.pth`。
6. 转换 ShipRS train/val 映射标注；微调训练仅使用 ShipRS train，ShipRS val 用于数据审计而不参与模型选择。
7. 以官方训练集 70%、ShipRS 30% 的采样权重执行混合微调。
8. 微调期间仍只在官方 val 上计算 official 指标和选择 best。
9. 使用第二阶段 `best_official_recall_fdr.pth` 完成普通验证集评测、模拟大图评测和最终汇总。

所有项目目录、数据目录、工作目录和采样比例通过脚本顶部变量提供默认值，并允许环境变量覆盖。脚本使用 `set -euo pipefail`，每一步校验输入和上一步产物，失败时立即退出。

## 模拟 10000×10000 大图与计时

新增大图构建工具，从官方 val 中选取一部分图片，按确定性顺序或固定随机种子铺到若干张 10000×10000 画布：

- 图片保持原比例和像素内容，不做会改变目标尺度的整体缩放；必要时分行铺放。
- 未占用区域填充为固定背景色。
- 只选择能够完整放入画布的图片，不裁断目标。
- 同步平移 bbox，并输出模拟大图对应的 COCO GT JSON 和来源映射清单。
- 默认生成少量大图，不要求覆盖完整 val；数量可通过参数调整。

大图推理沿用切片检测与全图坐标合并逻辑，并应用同一组分类固定阈值。单图计时严格从图像已经读入内存、准备开始模型切片推理时开始，到所有切片检测完成、全图合并/NMS/阈值过滤并形成最终结果时结束；不包含磁盘读取和结果写盘。报告每张大图的推理时间、平均值和最大值，其中官方时效性检查使用最大值。

## 输出

最终工作目录至少包含：

- 第一阶段和第二阶段各自的 `best_official_recall_fdr.pth` 与 best 元数据。
- 普通 val 的预测 JSON、CSV/JSON 指标报告。
- 模拟大图、模拟大图 COCO GT、来源映射和预测 JSON。
- 每个子类的 TP、FP、FN、Recall、FDR、Precision 和 AP。
- 舰船、飞机、车辆三大类 Recall/FDR。
- official Recall/FDR、merged Recall/FDR、COCO bbox 指标。
- 每张模拟大图的推理时间及最大推理时间。

## 错误处理

- 固定阈值必须完整覆盖 25 类，导入时检查类别集合和顺序。
- 预测结果类别数、COCO category id 或图像 id 不一致时明确报错，不静默忽略。
- 任何一个大类缺少真值时，报告中标明不可计算并阻止将该轮用作 best，避免用零除兜底值误导模型选择。
- 大图构建若无法放入至少一张源图则失败；输出目录已存在时默认拒绝覆盖，显式 `--overwrite` 才允许重建。
- shell 在 checkpoint、标注或图像目录缺失时立即停止，并打印缺失路径。

## 验证策略

本地环境按仓库说明不执行 Python、pytest、训练或推理。实现后采用静态验证：

- 检查 25 个硬编码阈值与 0.19 CSV 逐项一致。
- 为 best 比较函数补充边界测试，覆盖 0.85、0.20、Recall 差 0.005、首次双达标和双达标后拒绝不达标候选等情形，供训练服务器运行。
- 为 8:2 划分补充无重叠、全集覆盖、无 test 输出和稀有类保护测试。
- 为指标模块补充重复框、车辆 IoU 0.35、其他类别 IoU 0.50、按子类与大类求平均测试。
- 为大图标注平移、边界合法性、固定尺寸和计时边界提供测试。
- 使用 `bash -n` 的等价静态审查方式人工检查 shell；真正的语法检查、单元测试、训练和 GPU 推理在远程训练服务器执行。

## 非目标

- 不重新搜索或动态校准 score threshold。
- 不使用最终 val 指标反向修改阈值。
- 不改变检测网络结构、损失函数或 ShipRS 类别映射规则。
- 不拼接全部验证图片，也不把模拟大图训练回模型。
