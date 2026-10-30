# Phase 1：MID-conditioned Target Sound Extraction

本分支提取完整的、实际参与混音的 FSD stem，包括该录音中伴随的其他声音。
不把弱标签当作“纯类别声源”保证；不包含双 FSD 提取，也不运行正式训练。

## 实现

- `nets/dprnntf_original/`：从老师的 `example_code` 复制五个文件，只调整包内 import；核心网络、归一化、卷积、激活和 STFT/iSTFT 算法均保留。
- `nets/tse_dprnntf.py`：共享 MID embedding，各 Block 前独立 FiLM；投影末层权重和偏置均为零。单通道、单源、complex mapping，不使用 CRNN。
- `datasets/tse_dataset.py`：复用原 Strong 背景划分和 FSD 官方 Train/Val 划分，沿用已声明重复排除及原混音函数。直接返回共同峰值归一化的 mixture、target、background，不再除以 mixture 标准差。
- `loss_functions/tse_loss.py`：负 SI-SDR + MR-STFT，默认权重 1。MR-STFT 的 `(FFT, hop)` 为 `(256,64)、(512,128)、(1024,256)`，每项包含谱收敛和 log-magnitude 误差。
- `tse_separator.py`：继承现有 Lightning 外壳的前向、优化器、调度器、worker 随机种子及 DataLoader 入口，隔离 SED 指标/hooks。支持显式 checkpoint 恢复。
- `utils/tse_config.py`、`exp_defs/tse_strong_fsd50k.yaml`：独立任务校验和配置。
- `tools/evaluate_tse.py`：完整来源泄漏审计、固定 CV 重放、可选 checkpoint 评估、等比例 FLOAT WAV 导出。

现有文件修改 `creators/{dataset,model,loss,task}.py`、`utils/training_config.py`、`train.py` 的任务注册/路由，以及 `run.sh`、`spksep_run.sh` 的 TSE 双节点入口。
原 `source_separator.py`、CRNN、SED Loss、SED Dataset、混音函数、原配置没有改动。
工作区原有的 `tools/README.md` 修改及三个其他工具文件也未改动。

## 数据和复现约定

Condition 必须来自 FSD MID；背景原始 Weak 标签、Strong 正类标签中精确相同的 MID 均排除。
采样等价于原来的“均匀抽 FSD、再均匀抽 MID”，以合法配对为条件；使用有限的直接采样，不无限重试。
每个有效背景每轮生成一个动态训练样本。

`tse_no_pair_policy: error` 默认报告全部无合法配对背景并终止。显式设为 `exclude` 才在各自分区内排除，仍记录 ID 和数量；不重新划分 Train/CV。
报告包含原始/有效背景和 FSD 数、可达 FSD、合法/排除配对数、可用 MID 覆盖、FSD 多标签比例、预期采样多标签比例，以及固定 CV 的实际覆盖/比例。多标签统计保留原始 MID 集合，包括任何未映射类别，避免低估监督歧义。
训练期间另记录每轮实际采样类别计数和多标签比例。

TSE 的 `tse_phase1_v1/cv_{node}.json` 是独立 manifest：绑定源文件、代码、版本、类别映射、采样设置，记录 FSD/MID、种子、SNR、裁剪/放置、增益以及三条波形的哈希。
设置或内容改变时拒绝加载/重放；不要覆盖旧 SED manifest，也不要通过删除校验字段恢复兼容。

保留旧预处理的严格零采样点删除、短音频重复、随机裁剪、Active RMS、随机位置和共同峰值缩放。
模型的 STFT 使用老师的 sqrt-Hann。当前配置为 256 点窗、128 点 hop，10 秒对应 1251 帧、129 频点；保留 512/128 支持以读取历史实验。iSTFT 后按输入长度裁尾。
损失覆盖整条波形及其静音区域。近静音参考不计入 SI-SDR/SI-SDRi，改用输出能量惩罚，并记录数量。
第一阶段限定单设备，避免分布式验证采样器补齐重复样本影响指标。

## 入口（下列正式训练命令尚未执行）

01、02 共用新配置，launcher 根据实际计算节点选择路径，CV manifest 按节点独立保存。两台均为单 GPU、batch size 2；这表示分别运行，不是跨节点分布式训练。

| 节点 | 分区 | AudioSet 数据根目录 |
|---|---|---|
| ailab01 | L40 | `/hdd2/syy12632604/AudioSet` |
| ailab02 | RTX5880Ada | `/data/syy12632604/AudioSetMix` |

以下命令在用户后续同步本地代码后执行。本次未同步服务器代码、未创建远程缓存/manifest、未提交作业。
先生成新 TSE 缓存和固定 CV 并审计（会写派生文件，因此本次没有执行）：

```bash
srun --partition=L40 --nodelist=ailab01 --ntasks=1 --cpus-per-task=6 --mem=32G --time=02:00:00 \
  bash spksep_run.sh --dataset audioset_strong_fsd50k_tse --check-dataset true

srun --partition=RTX5880Ada --nodelist=ailab02 --ntasks=1 --cpus-per-task=6 --mem=32G --time=02:00:00 \
  bash spksep_run.sh --dataset audioset_strong_fsd50k_tse --check-dataset true
```

审计完成后，正式训练的手动入口（尚未执行）：

```bash
bash run.sh --dataset audioset_strong_fsd50k_tse --submit true --slurm-node ailab01 --num-jobs 1 --tag _tse01
bash run.sh --dataset audioset_strong_fsd50k_tse --submit true --slurm-node ailab02 --num-jobs 1 --tag _tse02
```

脚本沿用项目默认 Python 路径，必要时用 `--python-bin` 覆盖。直接使用 `train.py` 或 `tools/evaluate_tse.py` 时，必须传入明确的 `--mix-cv-manifest-path`，并正确覆盖该节点的数据路径；Python 入口不会猜测 `{node}`。
评估脚本不传 checkpoint 时仅审计数据，传 `--checkpoint /path/to/tse.ckpt` 时计算模型指标并导出 prediction。

`--max-samples 0` 表示完整 CV；正整数仅抽查前若干条，报告会注明不是完整评估。
导出不对各音轨单独归一化，使用 FLOAT WAV 保留原始相对幅度。

## 本地验证结果（2026-10-18）

使用现有 `.venv`、CPU、float32，临时合成语料和测试脚本；没有启动真实语料训练。后续仅只读访问服务器标签、划分和源音频。
详细 JSON 保存在 `outputs/tse_phase1_verification/`。

| 检查 | 结果 |
|---|---|
| mixture = background + target | 最大误差 `1.1920929e-7` |
| 160000 点 STFT/iSTFT 往返 | 最大误差 `8.9406967e-8` |
| 非 hop 整数倍及极短长度 | 1、511、16000、160013 点均通过长度/误差检查 |
| FiLM 恒等初始化 | 与老师原网络输出逐值一致 |
| FiLM / embedding 梯度 | 有限；embedding 在第二次合成更新后获得梯度 |
| 条件路径 | 合成更新后切换 MID 可改变输出；不代表已学会语义提取 |
| SI-SDR | 与独立公式误差约 `3.26e-7 dB`；零预测/静音参考处理通过 |
| MR-STFT | 恒等重建为 0；扰动重建非零且反向梯度有限 |
| 原尺寸模型 | 4 Block、C=128、H=200，约 565 万参数；完整 10 秒前向通过，1 秒反向通过 |
| 10 秒训练链路 | C=8、H=8、1 Block 的测试模型完成 1 次 Lightning 优化、验证、checkpoint 重载 |
| train.py 入口 | 另用合成数据完成 1 次优化及原 checkpoint callback 保存；参数不匹配的恢复被拒绝 |
| CV | 重建 Dataset 后逐值相同；源内容或配置变化被拒绝 |
| 筛选 | Weak/Strong 冲突排除；无合法配对的报错和显式排除均通过 |
| 导出 | GT FLOAT WAV 与原 target 逐值相同；prediction 导出通过 |
| SED 回归 | 原 CRNN + masked BCE 前向/反向通过 |
| 双节点 launcher | 01/02 路径、分区、审计及训练分支模拟检查通过；未实际提交 |

原尺寸模型完整 10 秒反向传播和 GPU 显存需求尚未验证，不据此承诺正式训练 batch size。
当前配置已按用户要求调整为 batch size 2、学习率 0.0001、窗长 256、hop 128、梯度检查点开启、FP32；以上旧测试数据对应修改前的配置。

## 01 显存不足后的修正

01 的正式 Lightning 训练曾在反向传播时 OOM：GPU 总容量 44.42 GiB，
当时空闲 3.30 GiB，但下一次分配需要 3.54 GiB。02 的单步 GPU smoke test
通过并不保证另一节点的完整训练具有足够显存余量。

新配置启用 `tse_gradient_checkpointing: true`，仅在训练且启用梯度时，
使用非重入 checkpoint 重算各 DPRNNTF Block，降低保存中间激活的开销。
网络尺寸、FP32、10 秒输入、FiLM、Loss 和 Dataset 均保持原设置；计算时间会增加。
旧冻结配置缺少此字段时保持关闭，更新源码不会自动改变已经提交的配置。

本地 CPU 检查确认启用前后输出、Loss 和参数梯度一致，且轻量模型完成
10 秒反向和优化步骤。结果见 `outputs/tse_phase1_verification/gradient_checkpointing.json`。
这不是新的 GPU 显存测量；更新服务器后仍需在目标节点重跑 10 秒 smoke test。
没有修改数据混音逻辑，已有 TSE CV manifest 可以继续使用。

## 真实语料统计的边界

本机不含完整语料，早期 `corpus_audit/report.json` 的 `unavailable` 仅表示本机缺失。随后获准只读连接服务器，02 完成当前源音频全量读取检查及元数据配对计算，结果保存在 `server_ailab02_readonly.json`：

| 项目 | Train | CV |
|---|---:|---:|
| 有效 Strong 背景 / 每轮样本 | 6113 | 1082 |
| 有效 FSD 音频池 | 36076 | 4158 |
| 可用 Condition MID | 192 | 192 |
| 无合法配对背景 | 0 | 0 |
| 合法 background/FSD/MID 组合 | 528793260 | 12280893 |
| 排除的同 MID 组合 | 31126975 | 572185 |
| 原始多标签 FSD 比例 | 84.07% | 88.67% |
| 条件采样后的预期多标签比例 | 83.59% | 88.43% |

已保留原 Train/CV 归属，并沿用声明的重复排除。FSD Train 另有 3 个严格零能量录音被排除：166709、176970、368070。多标签比例按完整原始 MID 集合计算，包括模型类别表外标签。
这些是实际源音频和标签的统计；192 是合法候选类别覆盖，不是最终固定 CV 抽样覆盖。尚未在服务器生成/重放新的 TSE CV manifest，未重新完成全语料跨分区内容哈希泄漏审计，也没有验证 GPU 上的 10 秒反向显存。首次 TSE 数据审计会执行这些数据检查并明确报错。

合成夹具仅用于正确性验证：Train/CV 各 2 个背景、各 3 个 FSD 文件；各有 5 个合法配对、3 个冲突配对、4 个可用 MID，池内多标签比例 1/3。
这些数字不能替代真实数据统计。

仍存在的监督歧义包括 FSD 中伴随声音、弱标签对应声音可能不在裁剪片段内，以及背景漏标。
来源审计可发现 ID、路径和文件字节哈希重复；重编码或局部内容重复仍需额外音频内容审计。

## 当前统一运行配置（2026-10-18）

`exp_defs/tse_strong_fsd50k.yaml` 与保留历史名称的
`tse_strong_fsd50k_b4_lr1e4.yaml` 均统一为 batch=2、LR=0.0001、
window=256、hop=128、梯度检查点开启、梯度累积=1、FP32、50 epochs。
新提交统一使用前者，实验名包含 b2/w256/lr1e4，并从头训练。

10 秒是当前 Dataset、混音和 CV manifest 的固定约定，不是 DPRNNTF 的理论限制；
本次不改变该约定，也不改变 MR-STFT Loss 的三组分辨率。
窗长变更改变模型频率维度，不兼容旧模型权重；已有波形混音 CV manifest 不受影响。
256/128 的两条 10 秒波形往返最大误差为 8.94e-8；本地原尺寸模型 batch=2
的一秒输入反向通过。服务器完整 10 秒 Lightning 短测试及正式作业记录以部署报告为准。
