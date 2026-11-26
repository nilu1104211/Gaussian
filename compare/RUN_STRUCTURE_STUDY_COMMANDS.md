# PU-GCSNet 统一实验队列

入口：`compare/run_structure_study.py`。本文件描述代码行为，不代表新结构已经正式训练或取得收益。

## 一次启动的流程

1. 自动对选定版本和数据集运行3轮原生 smoke，检查梯度、参数更新和验证后继续训练。
2. 依次训练组件版本 V1a、V1、V2a、V2；默认两个数据集各自训练20轮，可用 `--epochs` 指定预算，每5轮及末轮用独立整物体验证集，按验证 CD 选 best。实际原生队列先完成 PU1K 的四个版本，再完成 PU-GAN 的四个版本。
3. 对每个组件版本的 best 自动做两个完整验证集的 P1/P2/P3、FPS 前后、位移诊断。
4. 20轮筛选默认也训练V3，作为预设组合候选，不根据短跑结果提前断言模块无效。100轮预算默认要求V1、V2各自在所有选定数据集的验证CD均低于E6，才进入V3；可用 `--combination-policy always` 明确运行预设组合。
5. 输出与 E6 的验证对照表和阶段表，任务结束。

默认10组筛选训练（5版本×2数据集），每组都是完整三级网络联合训练20轮；另有每组3轮smoke。20轮用于观察健康和早期验证趋势，不能确认最终排名，也不能直接淘汰可能晚收敛的版本。100轮的V3门槛也只是预设单seed筛选，不是统计显著性证明。脚本不自动修改结构、损失、选模规则或运行测试集。

M0、Tail-only 不在默认队列；需要时可通过 `--models` 指定。

## 本机上传（PowerShell）

```powershell
cd D:\dkwork\Gaussian
git add -- compare/run_retraining.py compare/run_structure_study.py compare/tests/test_structure_study.py compare/tests/test_structure_integration.py compare/RUN_STRUCTURE_STUDY_COMMANDS.md
git diff --cached --stat
git commit -m "Add 20-epoch screening and checkpoint continuation"
git push origin main
```

## 服务器启动（Linux）

```bash
cd /home/zk/Gaussian
git pull --ff-only origin main
conda activate scdata-v3

# 可先查看计划：不会加载CUDA、读训练数据或开始训练
python compare/run_structure_study.py --epochs 20 --plan

# 一次后台启动整个队列
nohup python -u compare/run_structure_study.py \
  --epochs 20 \
  --run-dir compare/retrain-runs/structure-study-20261126 \
  > compare/retrain-runs/structure-study-20261126.launch.log 2>&1 &

echo "PID=$!"
tail -n 60 -f compare/retrain-runs/structure-study-20261126.launch.log
```

首次运行目录必须尚不存在；若目录已运行过，使用下面的恢复命令。查看日志时 Ctrl+C 仅退出 tail，后台队列继续。队列与原训练入口共用同一GPU锁，不能同时启动两个训练队列。

默认使用现有 `data/repro_20261123_seed42` 正式数据，以及独立的 `data/repro_20261123_seed42_smoke_b16` smoke数据；环境和附加依赖由已有原生入口检查/准备。

旧 E6 只需要验证报告，不需要恢复旧权重。20轮筛选优先读取原E6的validation.csv，在第5/10/15/20轮中按CD选best，保证同预算比较。若没有这段历史，则使用完整best参考，并在汇总中标记 `comparison_budget_matched=False`，不能据此声称同预算优劣。100轮预算优先使用已完成的E6完整诊断。可显式指定诊断来源：

```bash
python -u compare/run_structure_study.py \
  --baseline-stages compare/retrain-runs/e6-validation-stages-20261126-165829 \
  --run-dir compare/retrain-runs/structure-study-custom
```

该基线是已完成的验证集报告，脚本不会再次诊断旧 E6。基线指标与来源哈希会冻结到 queue.json。

## 20轮后继续到100轮

先查看筛选结果，再选择有希望的版本。以下以继续V1为例：

```bash
nohup python -u compare/run_structure_study.py \
  --continue-from compare/retrain-runs/structure-study-20261126 \
  --epochs 100 --models V1 \
  --run-dir compare/retrain-runs/structure-V1-continue100-20261126 \
  > compare/retrain-runs/structure-V1-continue100-20261126.launch.log 2>&1 &
```

不传 `--models` 时继续原队列中的全部版本；只继续V3时加 `--models V3 --combination-policy always`。续训来源须为已完成的较短预算队列，目标预算只能增加，最多100轮。原来被跳过、没有checkpoint的V3将在新预算下从头训练，其余已训练版本续训。

续训使用**新目录**，复制原last/best和训练/验证历史，再从last恢复模型、Adam状态、学习率调度器、随机状态和已保存epoch。例如20→100从第21轮开始，不是重新训练100轮；warmup、step衰减与增强顺序继续原进度。20轮原目录保持完整，100轮诊断另存。新预算开始前会重做3轮smoke健康检查。

数据、seed、batch、结构和损失必须与来源一致；可选择原模型的子集。源码版本也必须一致，因此应在第一次20轮筛选前上传本次全部代码，筛选与续训之间不要更新源码。

## 查看状态和恢复

```bash
python compare/run_structure_study.py \
  --status compare/retrain-runs/structure-study-20261126
```

中断或错误处理后，用原目录恢复，无需重复传模型、数据集或其他训练参数：

```bash
nohup python -u compare/run_structure_study.py \
  --resume-run compare/retrain-runs/structure-study-20261126 \
  >> compare/retrain-runs/structure-study-20261126.launch.log 2>&1 &
```

`--resume-run` 恢复同一预算的中断任务；`--continue-from ... --epochs 100` 才是把已完成的20轮筛选延长到100轮。恢复会校验源码、冻结配置、prepared manifest、已完成权重及诊断文件；已完成任务跳过，训练从last.pth的已保存epoch继续。诊断失败时保留旧尝试，创建新的attempt目录。不会在缺少已有训练的last.pth时默默从头训练；首个checkpoint形成前的中断可重新从初始状态开始。

不要在队列运行或恢复之间更新代码；源码指纹变化会拒绝混用结果。结束的组件数据和原归档保持原位。

## 输出位置

```text
structure-study-20261126/
  queue.json                     # 队列状态、配置、基线快照、每一步状态
  combination_gate.json          # 是否进入V3及各项比较
  validation_comparison.csv      # E6与各版本best验证指标，指标列×1000
  stage_comparison.csv           # 各版本P1/P2/P3与FPS前后指标
  smoke/                         # 原生短程健康检查
  components/{pu1k,pugan}/{V1a,V1,V2a,V2}/
    best.pth / last.pth
    validation.csv / training.jsonl / train.log
    config.json / manifest.json / status.json / health.json / performance.json
  combined/{pu1k,pugan}/V3/       # 通过门槛后才生成
  diagnostics/模型/数据集/attempt-时间戳/
    protocol.json / summary.csv / per_shape.csv / movement.csv
```

`validation_comparison.csv` 的 CD_change_vs_E6_percent 为百分比变化，负数表示CD更低；其他指标值使用列名中的x1000尺度。training_epochs和comparison_budget_matched明确训练预算是否匹配。原生各模型的validation.csv保持原始值，不能把两种表直接混用。结束后可将整个队列目录回传本机，复核各模块收益。

## 可选队列

只自动跑生成分支：

```bash
python -u compare/run_structure_study.py --models V1a V1
```

100轮预算下明确按预设直接跑全部5个版本、不按验证结果跳过V3：

```bash
python -u compare/run_structure_study.py --epochs 100 --combination-policy always
```

也可仅追加一个独立V3实验（新队列，不重复训练其他版本）：

```bash
python -u compare/run_structure_study.py --models V3 --combination-policy always
```

`--structure-config` 延用原入口的JSON开关格式；脚本会复制一份到队列并冻结，恢复时自动读取。每个不同配置使用新的队列目录。

## 最终测试

结构和权重在验证集上确定后，再显式执行原有测试入口。以选定V3为例：

```bash
python -u compare/test_examples.py \
  --training-run compare/retrain-runs/structure-study-20261126/combined \
  --prepared-root data/repro_20261123_seed42 \
  --models V3 --all-shapes --metrics --no-previews --tag V3-final
```

如果选定V1，training-run改为同一队列的components，models改为V1。统一实验队列本身不访问测试集来选结构。
