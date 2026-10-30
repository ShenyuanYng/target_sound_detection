# 当前 RMS 口径

唯一公式位于 `datasets/mix_utils.py::compute_rms(waveform)`：

```python
rms = sqrt(mean((x - mean(x)) ** 2))
```

float64 计算；空数组、全零、纯直流返回 `0.0`；输入有限且为一维，不修改输入数组。`normalize_rms` 只按测得 RMS 调整增益，不减去波形均值；零 RMS 时明确报错，防止除零。没有调用方使用非零均值统计，故删除 `zero_mean` 参数；重复别名 `waveform_rms` 的调用方统一改为 `compute_rms`。

当前调用点：

- `mix_utils.py`：通用读取、校验、去零、重复、裁剪、补零、RMS 与 `scale_foreground_to_snr`；不反向导入 `active_rms`。
- `active_rms.py`：活动帧统计、参数校验，以及唯一的单源混音实现 `mix_rms`；`make_snr_mixture` 负责源预处理和身份元数据，不重复实现混音。
- `mix_background_audit.py`、`mix_fsd_audit.py`：直接用 `compute_rms` 记录 zero-mean RMS。
- `dual_fsd.py`、`dual_fsd_dataset.py`：通过 active 混音使用相同公共公式。
- `tools/audit_dual_fsd_dataset.py`：复核固定样本的 RMS/SNR。

Active 检测逐帧减去该帧均值；确定 mask 后，再对选中的全部样本计算一次 zero-mean RMS。无 active 帧时沿用 `max(crop RMS, threshold amplitude)` 分母下限，并记录是否达到请求 SNR。

原有特征阶段 `x / std(x)`、能量伪标签 `frame_gt.py`、裁剪与补零规则保留。

## 2026-10-21 调用审计与精简

检查范围为仓库 Python、Shell、配置与文档，沿 `creators/dataset.py::dataset_creator` 追踪训练、评估和 Audit 入口。没有发现仓库内保留的独立测试文件；文档中的历史引用不视为运行调用方。以下结论不保证仓库外私人脚本兼容。

| 函数 | 重构前的有效调用方 | 处理 |
|---|---|---|
| `make_snr_mixture` | TSE `_mix`、AudioSetFSD50KMixDataset `_mix`；StrongMixtureCore 继承后者 | 移至 `active_rms.py`，移除 `snr_definition` 参数和 full 预补零分支 |
| `mix_at_snr` | 只有 `make_snr_mixture`，有效 Dataset 均显式传 active | 整体删除；crop/full 无其他可执行调用方，active 仅是转发 |
| `mix_rms` | 上述 active 转发、Dual FSD `make_dual_mixture` | 保留唯一算法，删除 `mode` 参数 |
| `active_rms` | `mix_rms`、Dual FSD 审计脚本 | 保留活动帧算法 |
| `compute_rms` | Active RMS、源归一化、增益与导出统计 | 保留默认运算原样，移除无人使用的非零均值分支 |
| `waveform_rms` | 背景审计、FSD 审计、Dual FSD 审计 | 调用方改用 `compute_rms`，删除别名 |
| `normalize_rms` | `mix_rms` 的背景源归一化 | 保留，背景仍统计完整 10 秒 |
| `scale_foreground_to_snr` | `mix_rms`，包括 Dual 两次复用 | 保留 mask 和显式 reference 两种输入；后者为 fallback 所需 |
| `pad_segment_to_target_length` | `mix_rms` | 保留；包括 Dual 后续会拆出的临时补零，不能改变 RNG/履历 |

`mix_fsd_rms_mode`、Dataset 构造参数 `fsd_rms_mode`、`validate_rms_settings(mode, ...)` 仍保留：它们是当前配置校验与持久化身份的一部分，只接受 active，不是可选择 crop/full 的算法分支。元数据的 `fsd_rms_mode`、`snr_definition` 和历史统计字段名同样保留，避免改变消费者语义。

重构前：

```text
TSE / Single / Strong Dataset._mix
  → mix_utils.make_snr_mixture
  → mix_utils.mix_at_snr(active)
  → active_rms.mix_rms
Dual Dataset._mix_pair → dual_fsd.make_dual_mixture → active_rms.mix_rms(none)
```

重构后：

```text
TSE / Single / Strong Dataset._mix
  → active_rms.make_snr_mixture
      → mix_utils.prepare_fsd_crop（身份 RNG → crop_seed / pad_seed）
      → active_rms.mix_rms
          → active_rms.active_rms
          → mix_utils.normalize_rms / scale_foreground_to_snr
          → mix_utils.pad_segment_to_target_length
          → 相加、共同 Peak Normalization、导出统计
Dual Dataset._mix_pair → dual_fsd.make_dual_mixture
  → 两次 active_rms.mix_rms(normalization='none')
  → 原有双源位置安排、相加与共同 Peak Normalization
```

没有把两个数值步骤合并：FSD 仍先除参考 RMS 并转 float32，再算增益并转 float32；增益计算仍使用 `sqrt(power(10, snr_db / 10))`。16 kHz / 160000 点、−40 dBFS / 20 ms、严格去零与重复裁剪、fallback 下限、增益后补零、共同最终除数、运算与随机采样顺序均保留。Active mask 只统计，不删除低能量点。

## 本地验证范围

比较基准是开始重构时的工作区副本（包含当时已有学习注释），不是简单回退 Git HEAD。仅使用临时目录中的合成 WAV、527 类虚构映射与新建 Manifest，没有读取配置指向的真实语料。

- 493 组记录逐项完全一致：30 组 Active RMS、181 组 crop 混音、108 组完整源混音、108 组 crop/RNG、36 组 Dual 混音、6 组基础 RMS、7 组异常路径、8 组 Dataset/partition 输出、2 组 Strong 条件标签、2 组 Dual 审计、4 组 CV rows 和最终全局 RNG 状态。
- 数组比较 shape、dtype、原始字节 SHA-256；浮点标量比较 `float.hex()`，元数据递归比较。覆盖正常、微弱、静音、纯 DC、帧内 DC、非整帧、超长源、内部零点、不同 SNR/Seed、peak/none、Dual 条件裁剪 fallback、44.1 kHz 双声道 FSD 缓存转换。
- TSE、Single、Strong、Dual 的 Train 连续取样产生不同 seed；CV 重复取样与重新构造后精确一致。四份旧合成 Manifest 均被新代码拒绝，文件字节保持不变；新 Manifest rows 与旧版相同。
- 68 个 Python 文件 AST 与 10 个 Shell/sbatch 文件语法通过；两个模块在独立进程中的两种导入顺序均通过。
- `tools/check_local.sh` 的全局 whitespace 阶段仍报告本次开始前已存在的 `datasets/tse_dataset.py`、`tse_separator.py` 尾随空格；未修改这些已有注释空行。本次补丁未新增 whitespace 错误。

环境为项目现有 `.venv`：Python 3.12.14、NumPy 1.26.4、SoundFile 0.12.1、SciPy 1.17.1、Torch 2.8.0。以上证明这些输入在同一环境下无数值变化，不等于验证所有服务器语料或跨库版本逐字节一致。没有训练、网络 forward、服务器操作或自动提交 Git；一次性测试脚本和夹具不加入仓库。

## CV Manifest 与旧实验的安全衔接

**结果相同仍会更改源码哈希。** 四条 Dataset 主线均记录相关源码指纹，因此旧 Manifest 不能直接给新代码使用。不要覆盖旧 Manifest、手改其哈希或关闭一致性检查。

服务器后续由操作者执行以下流程，本次未执行：

1. 保留旧代码版本、原配置、旧 Manifest 和原 Audit 报告，用于已有实验复现。
2. 复制对应实验 YAML，给 `mix_cv_manifest_path` 指定不存在的新路径，例如 `manifests/tse_refactor_20261021/cv_{node}.json`。保持原数据根目录、split、类表、Seed、SNR、RMS、缓存和筛选配置不变；不要重建既有划分文件。
3. 在原来支持的计算节点手工运行数据检查入口，报告也使用新路径。下面只进入 Audit，不启动训练或提交 Slurm；`--check-dataset true` 必须保留：

   ```bash
   bash spksep_run.sh \
     --dataset audioset_strong_fsd50k_tse \
     --exp-def exp_defs/tse_refactor_20261021.yaml \
     --num-jobs 1 \
     --check-dataset true \
     --check-output outputs/tse_refactor_20261021_audit
   ```

   示例 YAML 需要操作者先复制并设置新路径，本次没有创建。Single / Strong / Dual 使用各自 dataset 名和配置；相同开关进入对应 Audit。Dual 必须为新 Manifest 生成新的完整 `.audit.json`，不可复制旧凭据。
4. 核对旧、新 Manifest 的 rows、配对、Seed、Crop/Padding、增益与样本输出，预期允许 settings 中源码哈希变化。真实数据若有差异，应保留两份报告查明原因，不能自动覆盖旧结果。
5. 新代码搭配新配置和新实验目录。TSE 的 `on_load_checkpoint` 比较全部 `mix_*`，其中包含 Manifest 路径，所以仅换路径后加载旧 checkpoint 也会被拒绝；本次不放宽该限制。需要原样续训已有实验时使用旧代码、原配置与旧 Manifest。
