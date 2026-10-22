# 大型 ANE 复测：2026-10-10

结果：当前转换器、裁片门禁、整图执行器及交付验收流程通过。Apple M2 上将原始三人温室合成图从 1536×1024 放大至 6144×4096；RGB 像素与此前验收的 ANE 6K 输出完全一致，MAE 和最大差异均为 0。PNG 编码不同，文件哈希不同。三张脸和九处原生区域已重新检查，最终人物预检及 version-2 finalizer 通过。

证据目录：`/Volumes/work/Ai/Super-Resolution-tests/ane-retest-20261010/`。

| 实测步骤 | 秒 |
| --- | ---: |
| 重建 DiT 0–17 | 86.358 |
| 重建 DiT 18–35 | 82.591 |
| 重建 Qwen VAE decoder | 56.382 |
| 512→2048 裁片，含首次设备计划检查 | 428.798 |
| 6K 模型流程，含加载、推理、保存与输出哈希 | 489.071 |
| 6K 进程采样上界，含启动退出和最多 10 秒轮询延迟 | 510.763 |

整图：VAE 编码 64.103 秒、DINO 特征 61.747 秒；两段 DiT 预测 92.698 / 93.481 秒；VAE 解码 28.718 秒；三个 Core ML 包加载 66.637 / 67.138 / 8.969 秒。其余时间包含色彩对齐、图像保存和输出哈希等。126 块 DiT 网格。整图复用同设备、同包哈希的裁片计划，重复计划检查为 0 秒。

计算计划首选 ANE 操作数为 2212 / 2226 / 272，CPU 首选操作数为 2 / 16 / 0。VAE 编码与 DINO 在 MPS；DiT 和 decoder 使用 Core ML CPU_AND_NE。禁用 PyTorch MPS CPU fallback，未切换模型或降低分辨率。这些是首选设备计划，不是逐操作执行追踪。

输出 SHA-256：`61d877cfd44b76c9db0ead8c991730e54c7c306c373a86742e98ddba598d118c`。对此前同源 MPS 6K 图的平均通道绝对差为 0.875333/255，p99 为 6/255，最大为 53/255；0.374949% 像素的任一通道差异超过 10。旧 ANE 与 MPS 原文件均未修改。

复查区域包括三张脸、海军蓝针织、橄榄色及浅色亚麻、交叠手部、玻璃屋顶、植物，外加整图构图检查。通过范围限于这张合成摄影风格源图，不代表所有真实照片、真实身份恢复或 12K 已通过。

10 秒资源采样：整图期间系统盘最低可用 5.186 GiB，系统级 swap 已用峰值 18403 MiB；进程 RSS 采样峰值 1.645 GiB 不包含系统 ANE 服务，不能当作总统一内存峰值。首次裁片计划编译期间系统盘一度仅余 0.468 GiB，随后自行回升；未删除数据、未设新空间阈值。未据此宣称加速或性能退化。裁片与整图 scratch 均自动清理，delivery 目录仅一张图片。

可复用的验证路径：工作盘 Python 环境和 TMPDIR；逐组件重新转换；当前源图审查和 version-4 approval；`--diagnostic --trial-box 1010 80 512 512 --seed 42 --align wavelet --save-package-plan`；原生裁片通过后，整图使用同一个 `--verified-package-plan` 和 `--trial-review`；最后绑定最终人物报告、原生审查和神经运行报告给 finalizer。三个转换报告均记录官方权重和生成包哈希，没有给旧报告补造字段。

关键文件：`delivery/ane-6k.png`、`delivery/delivery.json`、`delivery/ane-6k-runtime-report.json`、`packages/verified-plan.json`、`final-review.json`、`comparison-and-resources.json`、`resources-full6k.jsonl`、`job.json`。本次未修改推理代码，未提交或推送。
