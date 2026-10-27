# 在 PedROAM 上运行 TDL-GNSS

论文写作所需的完整配置、实现细节、逐路结果、复现命令和结论边界见 [实验配置与论文写作参考](TDL_GNSS_EXPERIMENT_DETAILS.md)。

我们为 [TDL-GNSS](https://github.com/ebhrz/TDL-GNSS) 增加了 PedROAM 数据接口：读取手机 GNSS 观测和广播星历，完成预处理、定位和参考轨迹评分。默认运行数据目录中的全部记录，每个历元都参与处理。

使用当前代码中的三维输入网络 `HybridShareNet`，加载作者在 `deprecated-main` 分支发布的 `hybrid_share.pth`。输入为载噪比、卫星仰角和初始定位残差，网络输出伪距修正量及观测权重，再由 TASGNSS 加权最小二乘解算位置。模型直接使用预训练参数和归一化统计量。

## 安装

在仓库根目录执行，使用 Python 3.10 和 CPU 版 PyTorch：

```bash
python3.10 -m venv .venv-pedroam
source .venv-pedroam/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r pedroam/requirements.txt
```

下载预训练权重：

```bash
python -m pedroam fetch-weights --output checkpoints/hybrid_share.pth
```

命令从作者的固定提交下载权重，并校验 SHA256。版本与验证记录见 [VALIDATION.md](VALIDATION.md)。

## 准备数据

将 PedROAM 发布包解压，例如放在 `/data/PedROAM`。程序使用其中的转换后 RINEX 观测文件和每日导航文件：

```text
/data/PedROAM/
└── recordings/
    ├── 0421-AM-2026/
    │   ├── navigation/*.rnx
    │   ├── phones/P01/gnsslogger/*_v302.26o
    │   ├── phones/P02/gnsslogger/*_v302.26o
    │   └── reference/
    │       ├── P01/trajectory-gt.csv
    │       └── P02/trajectory-gt.csv
    └── ...
```

评分读取 `reference/<phone>/trajectory-gt.csv`，使用以下列：

```text
epoch_id,observation_utc_seconds,gt_valid,gt_latitude,gt_longitude
```

## 运行

一条命令完成全部记录的预处理、预测和评分：

```bash
python -m pedroam all \
  --dataset /data/PedROAM \
  --weights checkpoints/hybrid_share.pth \
  --output runs/pedroam_all
```

`--output` 使用新的空目录。完整发布包包含 13 路记录，实际处理的记录和历元数会写入 manifest。

也可以分步执行：

```bash
python -m pedroam prepare --dataset /data/PedROAM --output runs/example
python -m pedroam predict --weights checkpoints/hybrid_share.pth --output runs/example
python -m pedroam score --dataset /data/PedROAM --output runs/example
```

先用一小段数据检查环境：

```bash
python -m pedroam all \
  --dataset /data/PedROAM \
  --weights checkpoints/hybrid_share.pth \
  --recording 0423-PM-2026/P02 --limit-epochs 10 \
  --output runs/smoke
```

`--recording` 可重复指定。正式跑全量数据时省略 `--recording` 和 `--limit-epochs`。

## 处理流程

1. 读取各手机的 RINEX 观测和对应日期的广播星历。
2. 逐历元进行等权最小二乘定位，计算仰角和观测残差。
3. 按官方规则筛选输入：残差向量 L2 范数不超过 500 m，并检查初始解和特征有效性。
4. 使用 `HybridShareNet` 预测观测偏差与权重，重新解算位置。
5. 保存全部预测后，匹配参考轨迹并计算水平误差。

预处理缓存保存 GNSS 初始解和三维特征，方便重复运行预测。缓存采用 pickle 格式，应使用本机生成的文件。

发布数据的 GPST 时间通过 TASGNSS 转换为 UTC；2026 年 4 月两者相差 18 秒。参考匹配采用 UTC 最近邻，最大时间差 0.25 秒。水平误差在共同 ENU 坐标系计算，预测与参考高程统一取零，投影原点为 22.3°N、114.18°E。

## 输出

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 每路记录的中位数、P95、RMSE 和可用率，包含等权 WLS 与 TDL |
| `common_epoch_metrics.csv` | 每路记录在两种方法都有输出时的指标 |
| `aggregate.csv` | 全部记录的汇总指标 |
| `test/<session>/<phone>/predictions.csv` | 每个时刻的位置、输出状态和失败原因 |
| `test/<session>/<phone>/scored.csv` | 匹配参考后的逐时刻水平误差 |
| `preprocess_manifest.json` | 记录列表、历元数、输入及缓存校验值 |
| `prediction_manifest.json` | 权重、模型、依赖版本和预测校验值 |
| `status.json` | 完成状态和失败原因统计 |

可用率为有效参考历元中成功给出定位的比例。`common_epochs` 比较两种方法在相同时刻的精度；`all_available` 分别统计各自有输出的时刻。建议同时查看共同历元的误差和总体可用率。

## PedROAM 结果

全部 13 路记录共 32,505 个 GNSS 历元，其中 31,929 个有有效参考，TDL 给出 30,005 个位置，可用率为 **93.97%**。

在两种方法都有输出的 30,005 个时刻，水平误差如下：

| 方法 | 中位数/m | P95/m | RMSE/m |
| --- | ---: | ---: | ---: |
| 等权 WLS | 41.00 | 112.59 | 58.94 |
| TDL-GNSS | 24.63 | 80.34 | 56.06 |

13 路记录的中位数和 P95 均有改善，其中 3 路的 RMSE 上升。缺失输出包括 1,919 次输入筛选拒绝和 5 次解算不收敛。

以上是三维网络及官方旧分支权重在 PedROAM 上的离线测试结果。新版默认入口使用九维 `HybridShareSysNet`，两者应分别记录。所用每日星历为事后汇总文件；权重的具体训练来源、参考时间和精度的进一步核验见验证记录。

## 测试

```bash
python -m unittest discover -s tests -v
TDL_TEST_WEIGHTS=checkpoints/hybrid_share.pth python -m unittest discover -s tests -v
```

测试覆盖特征构造、输入筛选、时间匹配、评分与权重加载。实际数据的验证过程见 [VALIDATION.md](VALIDATION.md)。原项目说明和引用见根目录 [readme.md](../readme.md)。
