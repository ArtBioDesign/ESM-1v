#!/bin/bash


# # 输出传入的所有参数
echo "Received arguments: $@"


# # 检查是否提供了必要的参数
#if [ "$#" -ne 2 ]; then
 #  echo "Usage: $0  <seq> <offset>"
  # exit 1
#fi

# 从命令行获取参数
seq=$1
offset=$2

# 获取当前工作目录
BASE_DIR=$(pwd)

#
model_location="/workspace/esm1v/esm1v_t33_650M_UR90S_1.pt"
#dms_input="/workspace/esm1v/data/BLAT_ECOLX_Ranganathan2015.csv"
dms_input=$3
mutation_col="mutant"
#dms_output="/tmp/results/BLAT_ECOLX_Ranganathan2015_labeled.csv"
dms_output=$4
scoring_strategy="wt-marginals"





# 运行 Python 脚本  
export TMPDIR=/tmp
export PATH=/opt/esmfold/bin:$PATH

# echo "Python path: $(which python)"
echo "Python path: $(command -v python)"


python /workspace/esm1v/predict.py \
    --model-location "$model_location" \
    --sequence "$seq" \
    --dms-input "$dms_input" \
    --mutation-col "$mutation_col" \
    --dms-output "$dms_output" \
    --offset-idx "$offset" \
    --scoring-strategy "$scoring_strategy"
