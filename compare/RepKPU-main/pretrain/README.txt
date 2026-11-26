RepKPU官方预训练权重（2026-11-26下载并核对）

来源： https://github.com/EasyRy/RepKPU#testing--evaluation

PU1K：
- 官方README链接：https://drive.google.com/drive/folders/1IPQJdiwGMSympYFqrnRIhFVTh096vEQc
- 该目录包含exp1/exp2/exp3。本次固定使用exp1，不根据本项目测试结果挑选。
- exp1目录：https://drive.google.com/drive/folders/1hBp5Mngp5P5JBsLvAn-zYRv5aV3shmO_
- 文件：https://drive.google.com/file/d/1Y9JXbVHoBATXd6XK_Tv3kTVnY5Owe4_-/view
- 原文件名ckpt-best.pth，本地命名pu1k_exp1_best.pth，4218975字节。
- SHA256：53629847526615c5cb9a036b88412b659698eb7fa69b4d4a50ffc69761534266
- 337个state_dict张量；键名和形状与本项目PU1K重训RepKPU完全匹配。

PU-GAN：
- 官方README常规版本：https://drive.google.com/drive/folders/1Iv2pPePqDXRSiDalrFiDDtJ4WiseCdhZ
- 文件：https://drive.google.com/file/d/1vYKh5_wpOM8_S7eXe8IFy1j4up4RHjca/view
- 原文件名ckpt-best.pth，本地命名pugan_best.pth，4243775字节。
- SHA256：cb917b6905668766ec2bf7954919b1dd45f4ff9dad6338afcc2f278ac5cedd56
- 481个state_dict张量；键名和形状与本项目PU-GAN重训RepKPU完全匹配。

均以torch.load(..., map_location='cpu', weights_only=True)成功读取。
这证明格式和结构兼容；完整CUDA推断仍在现有服务器scdata-v3环境运行。

官方另有PU-GAN *论文原版，使用RepKPU_o及pugan_paper_args.py，需官方--o。
本目录采用的是与当前重训实现对应的RepKPU版本，不将其称为论文原版权重。
两份官方train_log.txt原样保留，原始训练参数见日志第3行。
权重本身只含state_dict，本次best_epoch记为null，不套用本项目重训的5/60轮。
官方数据划分和训练协议与本项目重训不同，结果作为单独的预训练诊断行报告。

详细来源、文件哈希与兼容性记录见provenance.json。
完整测试：python -u compare/test_repkpu_pretrained_full.py
