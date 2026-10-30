#!/usr/bin/env bash
# TSE 实验入口：像 example_code 一样，在这里集中修改常用参数。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 运行阶段：0 = 预览配置与输出路径；1 = 正式训练。
# use_slurm=1：从登录节点提交；use_slurm=0：在已分配的计算节点内执行。
stage=0
use_slurm=1
node=ailab01                         # ailab01 / ailab02
num_jobs=1                           # 含首次训练；后续作业串行恢复同一个 run

# 实验方案：网络、loss、混音和条件采样沿用所选 YAML。
exp_def="exp_defs/tse_strong_dual_fsd_hierarchy_v2.yaml"
# exp_def="exp_defs/tse_strong_fsd50k.yaml"
# exp_def="exp_defs/tse_strong_dual_fsd_wrong.yaml"
# exp_def="exp_defs/tse_strong_dual_fsd_wrong_fast.yaml"
# exp_def="exp_defs/tse_strong_dual_fsd_wrong_single.yaml"
# exp_def="exp_defs/tse_hierarchy_v2_pilot_10pct_bf16.yaml"  # pilot 仅 num_jobs=1
# exp_def="exp_defs/tse_hierarchy_v2_smallfit.yaml"  # 固定 256/128 条混音，8000 step，单次作业

# 常用训练参数：留空表示继承 YAML，切换 pilot 时不会覆盖它的 5 epoch / BF16 设置。
# 需要覆盖时直接赋值，例如 batch_size=4、learning_rate=0.0001。
batch_size=""                        # 层级主配置：2
learning_rate=""                     # 层级主配置：0.0001
epochs=""                            # 层级主配置：50
max_steps=""                         # smallfit：8000；其他配置按 epochs 结束
patience=""                          # 层级主配置：5
num_workers=""                       # 层级主配置：2
seed=""                              # 层级主配置：7（不改变 mix_seed）
train_percent_check=""               # 层级主配置：1.0；pilot：0.1
val_percent_check=""                 # 层级主配置：1.0；pilot：0.1
precision=""                         # 层级主配置：32-true；pilot：bf16-mixed
accumulate_grad_batches=""           # 层级主配置：1

# Slurm 资源；GPU 类型与数据路径由 node 对应的配置决定。
cpus=8
mem="32G"
time_limit="2-00:00:00"
account=""                           # 集群不要求 account 时留空
# 优先使用 PYTHON_BIN；服务器默认选择现有 audioset 环境，本地预览使用 python3。
python_bin="${PYTHON_BIN:-}"

# run 名会自动包含 batch / LR / epoch / precision / job ID，可追加一个短标签。
tag=""                               # 例如 tag=baseline
manifest_tag=""                      # 源码/混音更新后设新值，例如 cleanup_v1
# manifest_tag 只隔离派生清单，不重新划分原始 Train/CV，不改变音频缓存。

options=(
    --stage "$stage" --use-slurm "$use_slurm"
    --node "$node" --exp-def "$exp_def" --num-jobs "$num_jobs"
    --cpus "$cpus" --mem "$mem" --time "$time_limit"
    --tag "$tag" --manifest-tag "$manifest_tag"
)
[[ -z "$python_bin" ]] || options+=(--python-bin "$python_bin")
[[ -z "$account" ]] || options+=(--account "$account")
for name in batch_size learning_rate epochs max_steps patience num_workers seed \
            train_percent_check val_percent_check precision accumulate_grad_batches; do
    value="${!name}"
    [[ -z "$value" ]] || options+=("--${name//_/-}" "$value")
done

# 命令行参数放在最后，优先于上面的设置；支持 --batch-size 和 --batch_size 两种写法。
# bash run.sh                             # 预览，不创建 run、不提交作业
# bash run.sh --stage 1                   # 按当前设置提交训练
# bash run.sh --stage 1 --use-slurm 0      # 已获 Slurm 分配后直接执行
exec bash "$project_root/spksep_run.sh" "${options[@]}" "$@"
