# PedROAM：官方冻结权重测试入口

本目录为 [TDL-GNSS](https://github.com/ebhrz/TDL-GNSS) 增加 **PedROAM 数据接入、GNSS 预处理、冻结模型预测及独立评分**。不包含训练或微调代码，不修改上游网络、训练脚本、预处理脚本及 WLS 库。

## 1. 这到底是哪一个算法版本？

请勿把本入口称为“新版默认九维模型”的结果，或原论文数字的严格复现。

| 项目 | 本入口使用的版本 |
| --- | --- |
| 网络代码 | 当前官方 main 的 `model/model.py::HybridShareNet`，上游提交 `a640b2832c90daeb1ce25644a1da91c8edeb13fc` |
| 网络输入 | 每条观测的 **载噪比、仰角、初始等权最小二乘残差**，3 维 |
| 预训练权重 | 官方 `deprecated-main` 提交 `76d9b684e1f7a60326514b1796ebfd829b532d46` 中的 `model/hybrid_share/hybrid_share.pth` |
| 权重 SHA256 | `0bc1a4d2a6e214c3f92d8595f2ea65364873c57c0234f89cc6546be83594161e` |
| 最终位置解算 | TASGNSS 0.1.5 的 WLS，使用网络输出的权重和伪距偏差 |

新版默认 `evaluate.py` 使用 `HybridShareSysNet`（9 维），不能直接加载上述 3 维权重。但新版保留的 `HybridShareNet` 可以严格加载此权重。本目录新增独立入口来完成这项明确指定的组合，**不是原封不动运行上游 evaluate.py**。作者未确认该权重是否就是论文最终实验 checkpoint，也未确认跨手机设备的适用性。官方已标记旧分支为 deprecated，请知悉这一限制。

## 2. 与官方默认流程的差异和不变项

**新增/适配：**

- 遍历 PedROAM 的手机 RINEX、逐日广播导航文件；默认处理全部 13 路、每个历元，绝不把旧实验的抽样训练缓存当作全量输入。
- 从官方 GNSS 初始解构造三维输入；保留官方 C/N0 的 int8 量化、500 m 残差向量范数筛选。额外拒绝无效初始解、非有限特征和不足四条观测，失败计入可用率。
- 不施加原采集平台的固定杆臂修正：手持手机不存在同一固定安装关系。参考目标仍为发布的行人参考轨迹，存在手机/身体参考点差异。
- 独立保存全部预测后，再读参考轨迹、做 UTC 最近邻匹配及水平评分。
- 输出 WLS 对照、相同历元指标、全部可用历元指标、失败原因和可用率。

**不做：**

- 不训练、不调参、不选 checkpoint、不重算归一化统计量；保留 checkpoint 内的均值/标准差。
- 不融合 IMU、地图或相机；不利用真值高度、姿态或位置先验辅助预测。
- 不平滑、不进行起点/航向/整轨迹对齐、不按测试真值误差筛除点。
- 不更改官方模型和 WLS 方程。输入筛选不是定位准确性保证。

预测阶段不需要参考文件，`prepare`/`predict` 不读取 `reference/`。

## 3. 数据组织

将原始数据放在仓库之外，例如 `/data/PedROAM`：

```text
/data/PedROAM/
└── recordings/
    ├── 0421-AM-2026/
    │   ├── navigation/*.rnx
    │   ├── phones/P01/gnsslogger/*_v302.26o
    │   ├── phones/P02/gnsslogger/*_v302.26o
    │   └── reference/P01/trajectory-gt.csv  # 只在 score 时需要
    └── ...
```

入口使用发布包附带的 `*_v302.26o` 转换观测文件；**不负责 Android 原始日志到 RINEX 的转换**，也不另行下载星历。若同一路有多个此类文件会报错，不会静默选一个。

`reference/<phone>/trajectory-gt.csv` 至少包含：

```text
epoch_id,observation_utc_seconds,gt_valid,gt_latitude,gt_longitude
```

该 fork 不重新分发数据、参考轨迹或设备原始日志。请通过数据集发布者提供的渠道取得数据，并遵守其许可。

## 4. 安装（Python 3.10，CPU）

在仓库根目录执行。这里用 `torch` 而不是上游 requirements 中的 `pytorch` 包名，且补充模型文件导入所需的 torchvision；不修改上游 requirements.txt。

```bash
python3.10 -m venv .venv-pedroam
source .venv-pedroam/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r pedroam/requirements.txt
```

固定依赖是本实验已执行环境的版本组合，不代表作者的所有版本要求。未验证 GPU、其他 Python 版本或全新操作系统下的安装兼容性。

## 5. 下载并验证官方权重

```bash
python -m pedroam fetch-weights --output checkpoints/hybrid_share.pth
```

只从上述固定官方提交下载文件，并校验 SHA256；下载错误或哈希不符时停止。不会把旧权重静默改成其他模型，也不会反序列化未验证的权重。

模型加载使用 `torch.load(..., weights_only=True)` 和 `strict=True`。`checkpoints/` 不提交 Git。

## 6. 一条命令测试所有记录

```bash
python -m pedroam all \
  --dataset /data/PedROAM \
  --weights checkpoints/hybrid_share.pth \
  --output runs/pedroam_pretrained_all
```

**默认发现并处理数据目录中的所有匹配记录**。完整发布应为 13 路；请检查 manifest 的记录数和各路历元数。没有 4 路测试集限制，也没有每 10/30 个历元取一个的抽样。若只提供部分数据，入口会处理实际发现的部分，不能据此声称完成全数据集。

`all` 要求全新/空输出目录，避免覆盖旧结果。步骤依次为：GNSS 预处理 → 冻结模型预测 → 独立评分。参考轨迹只在全部预测完成后读取。

### 分步运行

```bash
python -m pedroam prepare --dataset /data/PedROAM --output runs/example
python -m pedroam predict --weights checkpoints/hybrid_share.pth --output runs/example
python -m pedroam score --dataset /data/PedROAM --output runs/example
```

预处理缓存只包含 GNSS 初始解和网络输入；使用前验证缓存哈希。缓存为 Python pickle，**只加载自己本机生成的缓存，不能加载来历不明的 pkl 文件**。

### 可选冒烟测试（不是正式全量结果）

```bash
python -m pedroam all \
  --dataset /data/PedROAM --weights checkpoints/hybrid_share.pth \
  --recording 0423-PM-2026/P02 --limit-epochs 10 \
  --output runs/smoke
```

`--recording` 可重复指定；只有明确传入时才限定范围。`--limit-epochs` 只用于检查接口，其值写入 manifest，正式结果不要使用。

## 7. 算法及时间处理

```text
RINEX 原始观测 + 已下载的广播星历
        ↓
TASGNSS/RTKLIB 等权初始定位，计算残差与仰角
        ↓
官方残差向量 L2 范数 ≤500 m，且输入有效
        ↓
HybridShareNet：预测每条观测的权重和伪距偏差
        ↓
TASGNSS 加权最小二乘 → 每个历元的位置
```

被拒绝或求解失败的历元保留在输出中，`tdl_available=False` 和 `failure_reason` 记录原因，不插值填补位置。

- GNSS 解算内部使用 RTKLIB 处理时间系统及星历计算，不要求星历参考时刻等于观测时刻。
- 2026 年 4 月发布观测为 GPST，保存时间通过 TASGNSS `obs2utc` 减去 18 秒变为 UTC。这是该批数据适用的闰秒差，**不是适用于任意历史时期的通用转换**。
- 评分按 `observation_utc_seconds` 最近邻匹配，最大差 0.25 秒；不拟合时间偏移，不使用香港本地时区计算。
- 水平误差：将预测和参考的高程均置零，在共同 WGS84 ENU 框架计算水平距离；固定投影原点为 22.3°N、114.18°E。它与本地实验评分一致，不是大范围定位误差的精密测地定义。
- 使用事后汇总的每日广播星历，未审计当时的实时接收可用性；不能宣称严格实时因果实验。完整端到端时间一致性和参考精度审计仍待完成。

## 8. 输出和指标

```text
runs/pedroam_pretrained_all/
├── preprocess_manifest.json  # 输入/导航/缓存哈希、逐路历元、是否冒烟测试
├── prediction_manifest.json  # 模型/权重哈希、依赖版本、预测哈希、状态不变验证
├── cache/<session>/<phone>/gnss.pkl
├── test/<session>/<phone>/
│   ├── predictions.csv       # 不含真值的冻结预测及失败原因
│   ├── scored.csv            # 之后追加匹配真值的评分结果
│   └── common_epoch_metrics.csv
├── summary.csv               # 每路、每方法的中位数/P95/RMSE/可用率
├── common_epoch_metrics.csv  # 每路相同输出历元的公平比较
├── aggregate.csv             # 汇总的 all_available/common_epochs 两种口径
└── status.json               # 评分完成、记录数和失败原因
```

- `WLS`：等权最小二乘对照，不是融合结果。
- `TDL`：上述冻结官方预训练模型，**不是此前在 PedROAM 上训练的适配模型**。
- 可用率分母：具有可匹配有效参考的观测历元。分子：有输出且可评分的历元。
- `common_epochs`：两种方法都有输出的相同时刻。比较模型精度时优先采用此口径，同时报告总体可用率。
- 误差很大但仍有有限数值输出的点仍计入评分。`all_available` 的两个方法可能样本不同，不能把其误差差异全部归功于网络。

### 已完成的离线全量结果（供核对）

本工作区使用同一固定网络/权重和评分方法，在全部 13 路、32,505 个原始历元上完成测试，其中 31,929 个历元可评分、30,005 个有 TDL 输出。

| 相同 30,005 个有效时刻 | 水平中位数/m | P95/m | RMSE/m |
| --- | ---: | ---: | ---: |
| 等权 WLS | 41.00 | 112.59 | 58.94 |
| 冻结 HybridShareNet | 24.63 | 80.34 | 56.06 |

TDL 可用率 93.97%；1,919 个时刻被输入门槛拒绝，另 5 个解算不收敛。13 路的中位数和 P95 均改善，但有 3 路 RMSE 变差，不能声称始终更可靠。数值是本数据接口的离线实测，不是作者论文中的表格，也不代表其他硬件的效果。仓库不提交原始数据或大体积预测缓存。

## 9. 验证

```bash
python -m unittest discover -s tests -v
TDL_TEST_WEIGHTS=checkpoints/hybrid_share.pth python -m unittest discover -s tests -v
```

测试覆盖：不读取参考的数据发现、特征顺序/量化、500 m 向量门槛、UTC 匹配边界、预测完成前禁止评分、失败计入可用率、大误差不剔除、可选的官方权重严格加载。

上游许可证和论文引用方式见仓库根目录 LICENSE 与 readme.md；本接口不是原作者背书或维护的官方 PedROAM 实现。
