# E6：面向 HD 退化的局部支撑约束与尾部监督

日期：2026-11-19。实现代号 **HD-Guard**。这是根据本项目错误模式设计的候选方案，尚未完成正式训练，性能与新颖性均待验证。

## 依据和目标

E4 的 P2/P3 都经 FPS 保留 8192 点后，P3 相对 P2 的平均网格距离下降 7.50%，预测→GT 最大平方距离上升 9.43%，GT→预测最大平方距离下降 4.82%。因此先处理细化增加的预测端极端误差，同时保留覆盖约束。

本轮以 E4 为起点做同预算、从头训练的对照。保留 Stage-1、确定性二次 Stage-2、三级 CD 和 `0.1 Surf(P2)`，不添加法向监督。修改集中在 Stage-3 的位移和尾部目标。

## 1. 父点局部支撑约束

Stage-2 每个父点产生两个按顺序相邻的子点。将已有的 P1 kNN 均方根邻域尺度 `d` 按父子关系展开至 P2；推断不读取 GT，不新增 PCA 计算。

原 Stage-3 给出位移 Δ。新输出为：

\[
r_i=\beta\,\operatorname{stopgrad}(d_i),\qquad
\widehat\Delta_i=\frac{\Delta_i}{\sqrt{1+\|\Delta_i\|_2^2/r_i^2}},\qquad
p_{3,i}=p_{2,i}+\widehat\Delta_i.
\]

第一轮 `β=0.5`。小位移近似保留，大位移平滑收缩，且 `||Δ_hat|| < βd`。不依赖 PCA 法向来决定移动方向。这个尺度是邻域支撑尺度，不解释成 Gaussian 的概率置信区间。

尺度停止梯度，避免通过该梯度路径主动扩大限幅半径。它仍会随前向计算的 P1 几何变化。位移限幅只能限制新增偏差，不保证 P3 的真实 HD 一定下降，也不能独自修复 P2 已有的飞点。

## 2. 双向尾部风险

设 `e3_i=min_y ||p3_i-y||²`，`b3_j=min_p ||y_j-p3_p||²`。每个 patch 独立取误差最大的 `ceil(qN)` 个点求平均，记为 `Tq`。

\[
L_{tail}=\tfrac12[T_q(e_3)+T_q(b_3)].
\]

第一轮 `q=0.02`，1024 点时选取最差的 21 个点。双向计算，预测端处理飞点，GT 端处理覆盖缺口。沿用平方距离单位，不直接优化一个不稳定的最大点；这只是 HD 的训练代理。

## 3. 对应点的尾部退化惩罚

P2 与 P3 在 FPS 前保持逐点对应。设 `e2_i=min_y ||p2_i-y||²`：

\[
L_{growth}=T_q\!\left(\max(0,e_{3,i}-\operatorname{stopgrad}(e_{2,i}))\right).
\]

该项直接惩罚细化给对应预测点增加的距离。即使全局最坏点改善，只要其他点明显变坏，仍会产生约束。P2 参考距离停止梯度，避免通过直接恶化参考值来降低惩罚；P3 对 P2 的正常梯度路径保留。

这里比较每个预测点到 GT 的距离，不强制 P2/P3 匹配同一个 GT 点，也不把分别 FPS 后的点按行号对应。

## 4. 第一轮配置

\[
L=CD_1+CD_2+CD_3+0.1Surf(P_2)
+w(t)\,[0.05L_{tail}+0.05L_{growth}],
\quad w(t)=\min(1,t/10).
\]

| 项目 | 设置 |
|---|---|
| 实验目录 | `model/E6_HDGuard_seed42` |
| 初始化 | 从头训练，不加载 E4 权重 |
| 数据 | 原有 `prepared/train.h5`、`prepared/val.h5` |
| 预算 | 100 epoch，batch 16，seed 42 |
| 优化器 | Adam，lr 0.001，每 50000 步乘 0.7 |
| 细化 | `--refinement local --refine_radius 0.5` |
| 原贴面项 | `--lambda_surf 0.1 --surf_eta 0 --lambda_normal 0` |
| 尾部项 | `--lambda_tail 0.05 --lambda_growth 0.05 --tail_fraction 0.02` |
| 渐增 | 前 10 epoch 从 0.1 倍逐步增加至完整尾部权重 |

这些数值是首轮探索设置，不是已验证的最优值。本轮只启动完整 E6；通过独立开关预留消融，不自动追加训练队列。

最近邻结果复用 CD2/CD3 的计算。局部限幅不增加可训练参数。额外操作主要是逐点限幅与 top-k，实际耗时以服务器日志为准。

## 5. 模型选择和判断标准

- `best.pth`：仍按验证 P3 CD 选择，作为与 E0/E4 现有结果比较的主要权重。
- `best_hd.pth`：按验证 patch 的平均平方 HD 选择，作为补充结果。不能事后按测试集挑选这两个权重中的较好者来宣称公平提升。正式比较 HD 选模时，需要给基线采用同样的验证选模规则。
- 每轮验证输出 P2/P3 的 CD、平方 HD 和两方向最大平方距离。训练日志记录尾部损失、加权项、限幅比例及位移/d。
- 正式测试分别输出项目原归一化口径与共享输入中心/半径口径的 E6 对 E0/E4 配对报告。
- HD 下降时，同时检查 CD、GT→预测方向、逐物体改善比例和失败物体。若改善只出现在分别归一化口径，不能解释为真实几何误差已解决。

后续失败诊断：CD 明显上升且限幅触发率高，检查半径是否过紧；预测端 HD 下降但覆盖变差，检查双向目标权衡；P2 已有大误差且 P3 修不动，检查 P2 初值和限幅。每次只调整一个因素。

## 6. 上传 Git

在本机 PowerShell 执行，仅暂存本次实现与方案：

```powershell
cd D:\dkwork\Gaussian
git add our/PU-GCSNet/main.py our/PU-GCSNet/model.py our/PU-GCSNet/diagnose.py our/PU-GCSNet/network/PUCRN.py our/PU-GCSNet/network/model_loss.py our/PU-GCSNet/network/hd_guard.py our/PU-GCSNet/tests/test_hd_guard.py our/PU-GCSNet/run_hd_guard.sh our/PU-GCSNet/readme.md "our/PU-GCSNet/HD改进方案_E6.md"
git diff --cached --stat
git commit -m "Add local refinement bounds and tail risk supervision for HD"
git push origin main
```

## 7. 服务器运行

```bash
cd /home/zk/Gaussian
git pull --ff-only origin main
conda activate scdata-v3
cd our/PU-GCSNet
mkdir -p model
LOG="model/hdguard_$(date +%Y%m%d_%H%M%S).log"
nohup bash run_hd_guard.sh all > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

默认 GPU 0、seed 42、100 epoch。脚本按顺序完成一组训练、CD 最优权重测试、HD 最优权重测试、与 E0/E4 的双口径配对分析。`all` 发现 `latest.pth` 时按原配置续训；文件锁防止同一实验重复启动。配置变化被视为新实验，续训会检查损失设置是否一致。

`tail -f` 中按 Ctrl+C 只退出日志查看。整个流程结束时有 `HD-Guard complete` 标记；若只有 100 轮验证行，测试可能仍在执行。

中断后继续完成训练和测试：

```bash
nohup bash run_hd_guard.sh all > "model/hdguard_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

只补测试：`bash run_hd_guard.sh test`。`train` 仅允许新实验，`resume` 只续训；两者不自动测试。

预留的独立消融，当前无需一起启动：

```bash
# 保留原细化，仅加入尾部监督，写入独立目录。
VARIANT=tail bash run_hd_guard.sh all
# 仅加入局部位移限幅，保留 E4 损失，写入独立目录。
VARIANT=local bash run_hd_guard.sh all
```

## 8. 结果文件

- `model/E6_HDGuard_seed42/train.log`、`metrics.jsonl`、`config.json`。
- `best.pth`、`best_hd.pth`、`latest.pth` 和每 4 轮 checkpoint。
- `test/PU1K/input_2048/`：CD 选模结果，包括 P1/P2 子目录。
- `test_hd/PU1K/input_2048/`：HD 选模结果。
- `diagnosis/test_vs_E0_PUCRN_{official,shared}/` 等目录：逐物体配对 CSV 和分组 JSON。

本方案的待验证贡献是：利用生成阶段的父子局部支撑控制细化范围，结合对应点退化约束与双向尾部风险，让细化阶段更有针对性地处理 HD。不能将这些机制或尚未训练的 E6 写成已有效、已具备首次创新的成果。

## 9. 本地实现验证

2026-11-19 已通过 17 项测试，覆盖旧基线前向一致性、旧/新权重加载、逐点限幅、尺度/旋转一致性、零位移梯度、尾部双向梯度、停止 P2 参考梯度、CD 最近邻复用、共享坐标评估等。

另使用少量真实 PU1K patch 完成 CUDA 上 2 个 epoch、4 次优化的启动/续训检查，验证损失配置变化会被续训入口拒绝、HD 最优值持续保存；完成一个真实物体 2048→8192 测试及 P1/P2 输出检查。Bash 脚本语法检查通过。该检查仅验证实现链路，不能用于评价 E6 性能。记录见 [本地验证 JSON](model/_hd_guard_smoke_4bf5575f/verification.json)，实验产物由 Git 忽略。
