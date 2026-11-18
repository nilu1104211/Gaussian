# PU-GCSNet

Geometry-Constrained Gaussian Surface Network，面向 4× 点云上采样。代码从仓库的 PU-CRN 复制而来，实现方案 A 的第二级几何生成器。名称为本项目暂定名，性能与研究新颖性需要后续实验验证。

默认网络为 Stage-1 原始 2× 上采样、Stage-2 高斯曲面 2× 生成、Stage-3 原始 1× 细化。输入 256 点，三级输出依次为 512、1024、1024 点。训练接口返回 `[P1,P2,P3], gt`，eval 默认返回 `[B,4N,3]`。`return_aux=True` 在训练时增加第三个 aux 返回值，在 eval 时返回 `(P3, aux)`，其中含 P1/P2 和几何诊断张量。

实现集中在以下文件。

| 文件 | 内容 |
|---|---|
| `network/PUCRN.py` | `PUGCSNet` 和兼容入口 `CRNet`，第二级替换 |
| `network/gaussian_surface_stage.py` | 12 参数预测头、三种采样模式、切向生成与二次曲面映射 |
| `network/geometry_utils.py` | 排除自身的 kNN、float32 detached PCA、退化邻域处理、正交帧 |
| `network/geometry_losses.py` | GT PCA 法向、可靠性权重、贴面损失、无方向法向损失 |
| `network/point_ops.py` | 原 CUDA 扩展适配及纯 PyTorch 参考算子 |
| `model.py` | 原三级 CD、可选几何监督、分项日志 |
| `main.py` | 训练、验证、checkpoint、整物体推断与评估 |
| `tests/test_geometry.py` | 形状、梯度、几何、基线等价与整物体 FPS 测试 |

生成器的局部尺度来自 P1 的 kNN。默认 k=16，切向标准差范围为 `[0.15d,0.75d]`，中心各坐标偏移上限为 `0.2d`，法向残差系数为 0.2，曲面系数上限为 0.25。PCA 帧和尺度停止梯度，中心坐标路径保留梯度。参数头末层权重以标准差 0.001 初始化，曲面系数接近零。角度偏置初始化到 π/4，让两个尺度及三个高度系数从第一步就有梯度。

默认采用确定性的正反成对子点。高斯在这里表示学习的尺度与协方差几何。`random` 使用独立标准正态样本，`antithetic` 使用一个正态向量与其负向量。`gaussian3d` 对照使用中心、三个尺度、轴角旋转和球面采样方向，共 11 个参数，同样复用第二级特征提取器和局部几何特征。

安装 Python 3.9+ 和适配服务器 CUDA 的 PyTorch，然后安装其余依赖。

```bash
cd our/PU-GCSNet
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

CPU/GPU 均可使用 PyTorch 参考算子。检测到已编译的 `pointnet2_ops` 和 `losses` 扩展时，在 CUDA 张量上使用扩展。扩展源码随基线复制，服务器可在匹配 PyTorch/CUDA 的环境中构建。

```bash
python -m pip install --no-build-isolation ./pointnet2_ops_lib
python -m pip install --no-build-isolation ./losses
```

本机验证环境为 Python 3.9、PyTorch 2.5.1、CUDA 12.4、RTX 4070 Laptop，使用 PyTorch 参考算子。旧 C++/CUDA 扩展在该环境的构建尚未验证。Matplotlib 仅供可选着色工具使用。

数据接口沿用 H5 键 `poisson_256` 和 `poisson_1024`，形状分别为 `[S,256,3]` 和 `[S,1024,3]`。训练仍从稠密 GT 非均匀抽取 256 个输入点，并同步旋转输入和 GT。验证使用 H5 固定输入和 GT，关闭随机增广。几何监督在增广后的 GT 上实时计算 PCA，因此方向与当前样本一致，推断只读取输入点云。

`prepare_data.py` 从 `PU1K/PU1K_raw_meshes/train/train_meshes` 和 `PUGAN/PU-GAN/{simple,medium,complex}` 读取官方训练网格。它按来源各随机留出 10% 的网格，生成 `train.h5`、`val.h5` 和记录网格归属及 patch 行范围的 `split.json`。默认种子 42 对应 918 个训练网格和 102 个验证网格。

```bash
python prepare_data.py --data-root /data --output-dir /data/prepared --seed 42
```

脚本按三角形面积对原始网格重新采样，用 FPS 选 patch 中心并取 kNN，每个 ShapeNet 网格生成 50 对、每个 PU-GAN 网格生成 200 对 `256 → 1024` 点的 patch。生成结果保留加载器所需的 H5 键名。两组实验使用同一份预处理结果。整物体验证也可通过 `--val_data` 和 `--val_gt` 提供独立的 XYZ 点对。最终测试用原数据集的测试点对。

```bash
python main.py --phase train --name E3_seed42 \
  --h5_data /data/prepared/train.h5 --val_h5 /data/prepared/val.h5 \
  --stage2 quadratic --sampling deterministic \
  --batch_size 16 --max_epoch 100 --random_seed 42
```

batch size 16 是命令示例，程序默认值仍为 64。各对照实验需使用相同训练设置。GPU 显存不足时，可以用 `--batch_size 2` 跑通入口。`--device cpu` 用于 CPU 调试，`--num_workers` 默认 0，可按服务器内存和 CPU 调整。

| 实验 | 参数 |
|---|---|
| E0 | `--stage2 baseline` |
| E1 | `--stage2 gaussian3d` |
| E2 | `--stage2 tangent` |
| E3（默认） | `--stage2 quadratic --lambda_surf 0 --lambda_normal 0` |
| E4 | `--stage2 quadratic --lambda_surf 0.1` |
| E5 | `--stage2 quadratic --lambda_surf 0.1 --lambda_normal 0.001` |

表中 0.1 和 0.001 供启动调试。正式训练根据日志中的三级 CD、`weighted_surf` 和 `weighted_normal` 调整。`--surf_eta 0.5` 可增加最终输出贴面监督。`--confidence_tau 0.15` 可按最小特征值占比对可靠性降权，默认 0 关闭该过滤，退化邻域权重为 0。GT PCA 的邻域点数由 `--gt_k` 设置，默认 24。

`metrics.jsonl` 按训练步保存 CD1/2/3、Surf2/3、Normal、加权贡献、中心移动/局部尺度、各尺度/d、曲面系数绝对值、有效帧比例与 Stage-3 移动量。未开启的几何损失项记为 0。训练 CD 沿用原实现的双向平方距离各乘 0.5，radius 行为也沿用原实现。修复了数据加载器原地旋转导致训练样本累积变化的问题，以及验证分支的未定义变量。

每隔 `--save_step` 个 epoch 和训练结束后评估。生成 `model_EPOCH.pth`、`latest.pth`，验证 CD3 改善时生成 `best.pth`。checkpoint 包含网络结构配置、优化器、调度器、epoch、步数和随机数状态。推断自动读取 checkpoint 内的结构配置，旧权重没有配置时使用命令行结构。下面两种方式分别用于从基线初始化和恢复训练。

```bash
python main.py --phase train --name warm_start \
  --h5_data /data/prepared/train.h5 --val_h5 /data/prepared/val.h5 \
  --init_ckpt ../../baseline/3DVision-main/PointUpsampling/PUCRN/model/release/model.pth

python main.py --phase train --name E3_seed42 \
  --h5_data /data/prepared/train.h5 --val_h5 /data/prepared/val.h5 \
  --resume model/E3_seed42/latest.pth
```

从基线初始化时按名称和形状载入 Stage-1、Stage-3 和 Stage-2 特征提取器。resume 以 epoch 为边界。`--max_steps` 可限制调试训练步数，其在 epoch 中途停止后，恢复会进入下一 epoch。

整物体推断使用重叠 256 点 patch、相同的 patch 归一化与最终 FPS。输入目录直接包含 `.xyz`，预测目录名称取输入目录末级名称。

```bash
python main.py --phase test --ckpt model/E3_seed42/best.pth \
  --test_data /data/PU1K/PU1K/test/input_2048/input_2048 \
  --gt_path /data/PU1K/PU1K/test/input_2048/gt_8192 \
  --num_shape_point 2048 --result_dir results/PU1K --save_stages
```

最终 P3 保存到 `results/PU1K/input_2048/`，P1/P2 保存到其对应子目录，可分别评估。`inference.json` 记录包括 patch 提取和 FPS 的整物体平均时间。开启 `--save_stages` 会额外执行 P1/P2 的 FPS，计时时应记录该设置。

评估 CD 为双向平均平方距离之和，输出原始值。`HD_squared` 为两个方向最大平方距离的较大者，`HD_legacy_sum` 为原代码两个最大值相加的定义。CD 使用原基线对预测与 GT 各自进行的单位球归一化。由 patch 验证得到的 CD 与整物体指标在各自坐标归一化协议下解释。`evaluation.csv` 保存逐物体指标，`metrics.json` 保存平均指标。

P2F 使用复制的 CGAL 评估程序，需要服务器安装原评估器要求的 C++ 库并编译。运行后，`main.py --phase eval` 会读取 `*_point2mesh_distance.xyz`，汇总 P2F 并报告具有 P2F 文件的物体数。此 C++ 评估器尚未在本机编译验证。

```bash
bash evaluation_code/eval_pu1k.sh /data/PU1K/PU1K/test/original_meshes results/PU1K/input_2048
python main.py --phase eval --test_data /data/PU1K/PU1K/test/input_2048/input_2048 \
  --gt_path /data/PU1K/PU1K/test/input_2048/gt_8192 --result_dir results/PU1K
```

本轮已经通过 7 项测试、真实 PU1K 小样本的 100 次优化、恢复后开启两种几何损失的 4 次优化，以及单个完整物体 2048→8192 的命令行推断。测试覆盖全部 12 参数通道的有限非零梯度、确定性、点排列、退化邻域、各消融返回接口、几何损失与基线前向等价。基线等价检查将两者放在同一算子后端，独立基线目录需要在原相对位置。

## seed42 的双向 HD 诊断和 E2 消融

交接记录中，E0 和 E3 都用三级 CD 从头训练 100 轮，以验证 patch 的 P3 CD 选最佳权重。E3 的 PU1K CD 下降 3.48%，平方 HD 上升 6.25%。下一轮先做 E2，即保留切向成对生成、将二次高度置零并从头训练。正式测试的 127 个物体由 ShapeNet 100 个和 PU-GAN 27 个组成。PU-GAN 在此处是子集，两个子集可分别比较。

服务器从本仓库根目录进入实验目录，使用原来的训练环境和预处理数据。

```bash
cd /home/zk/Gaussian/our/PU-GCSNet
conda activate scdata-v3
bash run_e2.sh paired
bash run_e2.sh validation
bash run_e2.sh fps
bash run_e2.sh train && bash run_e2.sh test
```

默认使用 GPU 0 和 `/home/zk/Gaussian/data`。可通过环境变量 `GPU` 和 `DATA_ROOT` 指定其他设备或数据根目录。长训练可在 tmux 会话中运行；中断后用 `bash run_e2.sh resume && bash run_e2.sh test` 从 E2 的 `latest.pth` 续训。E2 的训练、验证及测试权重记录在 `model/E2_PUGCSNet_seed42/`。

| 命令 | 输出和用途 |
|---|---|
| `paired` | 读取 E0/E3 原有 `test/PU1K/input_2048/*.xyz` 和 GT。在 `model/diagnosis_seed42/E3_vs_E0/` 写入 `paired.csv` 和 `summary.json`。CSV 按候选模型相对参考模型的平方 HD 增量降序排列，正值表示退化。汇总给出每个子集的均值及改善、退化的物体数。 |
| `validation` | 用固定 `val.h5` 对 E0 和同一 E3 权重的高度倍率 1、0.5、0 分别推断。每个目录的 `patches.csv` 可按 `patch_index` 配对，也可用 `split.json` 的 H5 行范围归回物体。`summary.json` 记录最佳权重的 epoch、验证 CD 和各列的 patch 均值。 |
| `fps` | 根据 `paired` 的排序，在相同的前五个物体上分别重跑 E0/E3 推断。两个 `fps.csv` 给出 P2 和 P3 在 FPS 前、后的点数及方向误差。位移使用 FPS 前逐点对应的 P2/P3，在同一物体的四行中重复显示。 |
| `train`、`test` | E2 与 E3 使用相同训练设置。测试阶段补出 E2 的验证 patch 指标、正式 127 个物体的指标，以及与 E0、E3 各自配对的报告。 |

所有诊断距离均保存原始数值。CD、HD 用平方距离，P2→P3 位移用欧氏长度。正式整物体指标沿用预测和 GT 各自归一化的口径。FPS 诊断则用同一输入物体的中心和半径变换全部阶段与 GT。验证 patch 的 P95 先在每个 patch 内计算，汇总文件再对 patch 求均值。

固定 E3 权重改变高度只用于诊断。在确定性成对采样中，两个子点得到相同的法向高度，因而高度项会移动这对点的中点。用 E2 的重新训练结果检验这条分支是否有增益。若后续需要验证二次形式本身的贡献，可再与直接预测一个有界标量高度的生成器对照。

若预测→GT 的最坏距离增大，先用原始网格的 P2F 核实离面情况，再考虑对 P2 增加贴面监督。若 GT→预测的最坏距离增大，优先查看覆盖空洞以及 FPS 前后的变化。用验证集选择结构和高度设置，最终候选再用多个训练种子复核。重建 GT 的训练采样按三角形面积随机取点；如需报告输出均匀性，应另行测量点间距或均匀性指标。

代码沿用 PU-CRN 的 Apache 2.0 许可证与原有第三方来源声明。基线论文为 Du et al., Point Cloud Upsampling via Cascaded Refinement Network, ACCV 2022。
