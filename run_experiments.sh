#!/bin/bash

# 定义参数列表
experts_list=(2 3 4 6 8)
hidden_list=(64 128 256)

# 确保日志文件夹存在
mkdir -p ./MoEv2/logs

# 外层循环：遍历 expert 数量
for n in "${experts_list[@]}"
do
    # 内层循环：遍历 hidden_channels
    for h in "${hidden_list[@]}"
    do
        echo "=================================================="
        echo "Start training with num_experts = $n, hidden_channels = $h"
        echo "=================================================="
        
        # 定义唯一的标识符，用于文件夹和日志命名
        EXP_ID="experts_${n}_hidden_${h}_MultiClass"
        
        # 定义保存路径
        SAVE_DIR="./MoEv2/checkpoints/${EXP_ID}"
        
        # 定义日志文件路径
        LOG_FILE="./MoEv2/logs/train_${EXP_ID}.log"
        
        # 运行 python 脚本
        # 注意：这里增加了 --hidden_channels 参数
        python ./MoEv2/train.py \
            --train_csv ./Model_Benchmarks/Data/train_mean_std.csv \
            --val_csv ./Model_Benchmarks/Data/valid_mean_std.csv \
            --label_col Class \
            --num_experts $n \
            --hidden_channels $h \
            --save_dir $SAVE_DIR \
            > $LOG_FILE 2>&1
            
        echo "Finished ${EXP_ID}. Log saved to $LOG_FILE"
        echo "--------------------------------------------------"
        
        # 休息几秒
        sleep 5
    done
done

echo "All experiments completed!"