#!/bin/bash
# ============================================================
# KPA-MIL: 文本提示引导辅助任务辅助主任务的多任务学习框架
# 真实实验配置（论文结果对应的超参）
#
# 使用方法：
#   1. 修改 DATA_ROOT_* 和 PROMPT_DIR 为你服务器上的实际路径
#   2. 根据需要注释/取消注释相应的实验块
#   3. bash run_experiments.sh 或逐条复制到终端执行
#
# 注意：路径中的 /home/hzxu/... 是作者服务器路径，请替换为你的路径
# ============================================================

# ---- 路径配置（请修改为你的实际路径） ----
DATA_ROOT_CRC=/home/hzxu/vila_dataset/crc_outcomes_feat    # CRC 特征目录
DATA_ROOT_GMU=/home/hzxu/ViLa-MIL-main/feature_dir/gmu     # GMU 特征目录
DATA_ROOT_SR386=/home/hzxu/remote_data/sr386_feat          # SR386 特征目录
DATA_ROOT_TJ=/home/hzxu/ViLa-MIL-main/feature_dir/tj       # TJ 特征目录
PROMPT_DIR=/home/hzxu/ViLa-MIL-main/ViLa-MIL-V9-MDN-trae/text_prompt  # 文本提示目录

# ---- 通用超参（论文使用） ----
LR=0.0002
REG=0.0001
BATCH=4
SCHEDULER=step
AGGREGATOR=prototype

echo "=========================================="
echo "  KPA-MIL 实验脚本"
echo "  使用前请确认路径已修改为你的环境"
echo "=========================================="

# ============================================================
# 1. CRC-OS (结直肠癌总生存期)
#    主任务: cox | 辅助任务: 2个 (aux_label_0, aux_label_1)
#    文本提示: crc_text_prompt_gpt_nopre.csv
#    特点: 无 cross_scale_contrastive (weight=0)
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_CRC/ \
    --data_folder_s 20feat/h5_files/ \
    --data_folder_l 40feat/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task crcos \
    --text_prompt_file $PROMPT_DIR/crc_text_prompt_gpt_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task cox --record_loss \
    --lr $LR --reg $REG \
    --early_stopping_metric c_index \
    --scheduler_type $SCHEDULER \
    --prototype_number 16 \
    --aux_label_cols aux_label_0,aux_label_1 \
    --aux_task_losses ce,ce \
    --results_dir ./crcos_${LR},${REG}_cox \
    --exp_code crcos_conch_aux0_aux1_p16_step_gpt_nopre

# ============================================================
# 2. CRC-OS MDN 版本
#    主任务: mdn | 辅助任务: 2个
#    特点: 有 cross_scale_contrastive (weight=0.7)
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_CRC/ \
    --data_folder_s 20feat/h5_files/ \
    --data_folder_l 40feat/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task crcos \
    --text_prompt_file $PROMPT_DIR/crc_text_prompt_gpt_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task mdn --mdn_mode unified \
    --record_loss --lr $LR --reg $REG \
    --early_stopping_metric concordance_time_dependent \
    --mdn_sigma_min 0.01 \
    --scheduler_type $SCHEDULER \
    --prototype_number 12 \
    --aux_label_cols aux_label_0,aux_label_1 \
    --aux_task_losses ce,ce \
    --use_cross_scale_contrastive 1 \
    --cross_scale_weight 0.7 \
    --results_dir ./crcos_${LR},${REG}_mdn \
    --exp_code crcos_conch_aux0_aux1_p12_crosca0.7_step_gpt_nopre

# ============================================================
# 3. CRC-PFS (结直肠癌无进展生存期)
#    主任务: cox | 辅助任务: 2个
#    特点: 无 cross_scale_contrastive
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_CRC/ \
    --data_folder_s 20feat/h5_files/ \
    --data_folder_l 40feat/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task crcpfs \
    --text_prompt_file $PROMPT_DIR/crc_text_prompt_gpt_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task cox --record_loss \
    --lr $LR --reg $REG \
    --early_stopping_metric c_index \
    --scheduler_type $SCHEDULER \
    --aux_label_cols aux_label_0,aux_label_1 \
    --aux_task_losses ce,ce \
    --prototype_number 12 \
    --results_dir ./crcpfs_${LR},${REG}_cox \
    --exp_code crcpfs_conch_aux0_aux1_p12_step_gpt_nopre

# ============================================================
# 4. CRC-PFS MDN 版本
#    主任务: mdn | 辅助任务: 2个
#    特点: 有 cross_scale_contrastive (weight=0.7)
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_CRC/ \
    --data_folder_s 20feat/h5_files/ \
    --data_folder_l 40feat/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task crcpfs \
    --text_prompt_file $PROMPT_DIR/crc_text_prompt_gpt_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task mdn --mdn_mode unified \
    --record_loss --lr $LR --reg $REG \
    --early_stopping_metric concordance_time_dependent \
    --mdn_sigma_min 0.01 \
    --scheduler_type $SCHEDULER \
    --prototype_number 16 \
    --aux_label_cols aux_label_0,aux_label_1 \
    --aux_task_losses ce,ce \
    --use_cross_scale_contrastive 1 \
    --cross_scale_weight 0.7 \
    --results_dir ./crcpfs_${LR},${REG}_mdn \
    --exp_code crcpfs_conch_aux0_aux1_p16_crosca0.7_step_gpt_nopre

# ============================================================
# 5. GMU (多任务生存, 1个辅助任务)
#    主任务: cox | 辅助任务: 1个 (aux_label_1)
#    文本提示: gmu_text_prompt5total_gptnew_nopre.csv (已处理为单列task0)
#    特点: 有 cross_scale_contrastive (weight=0.5)
#    注意: 原始CSV有5列(task1-task5)，已批量处理为只保留task1列并重命名为task0
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_GMU/ \
    --data_folder_s 10featgmu/h5_files/ \
    --data_folder_l 20featgmu/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task MTL \
    --text_prompt_file $PROMPT_DIR/gmu_text_prompt_gpt_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task cox --record_loss \
    --lr $LR --reg $REG \
    --early_stopping_metric c_index \
    --scheduler_type $SCHEDULER \
    --prototype_number 2 \
    --aux_label_cols aux_label_0 \
    --aux_task_losses ce \
    --use_cross_scale_contrastive 1 \
    --cross_scale_weight 0.5 \
    --results_dir ./gmu_${LR},${REG}_cox \
    --exp_code gmu_conch_aux1_p2_crosca0.5_step_gpt_nopre

# ============================================================
# 6. GMU MDN 版本
#    主任务: mdn | 辅助任务: 1个
#    文本提示: gmu_text_prompt5total_gptnew_nopre.csv (已处理为单列task0)
#    特点: 无 cross_scale_contrastive
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_GMU/ \
    --data_folder_s 10featgmu/h5_files/ \
    --data_folder_l 20featgmu/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task MTL \
    --text_prompt_file $PROMPT_DIR/gmu_text_prompt_gpt_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task mdn --mdn_mode unified \
    --record_loss --lr $LR --reg $REG \
    --early_stopping_metric concordance_time_dependent \
    --mdn_sigma_min 0.01 \
    --scheduler_type $SCHEDULER \
    --prototype_number 4 \
    --aux_label_cols aux_label_0 \
    --aux_task_losses ce \
    --results_dir ./gmu_${LR},${REG}_mdn \
    --exp_code gmu_conch_aux1_p4_step_gpt_nopre

# ============================================================
# 7. SR386 (肉瘤生存, 3个辅助任务)
#    主任务: cox | 辅助任务: 3个
#    文本提示: sr386_text_prompt_gemini_nopre.csv
#    特点: 有 cross_scale_contrastive (weight=0.2)
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_SR386/ \
    --data_folder_s 10featsr/h5_files/ \
    --data_folder_l 20featsr/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task sr386 \
    --text_prompt_file $PROMPT_DIR/sr386_text_prompt_gemini_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task cox --record_loss \
    --lr $LR --reg $REG \
    --early_stopping_metric c_index \
    --scheduler_type $SCHEDULER \
    --prototype_number 24 \
    --use_cross_scale_contrastive 1 \
    --aux_label_cols aux_label_0,aux_label_1,aux_label_2 \
    --aux_task_losses ce,ce,ce \
    --cross_scale_weight 0.2 \
    --results_dir ./sr386_${LR},${REG}_cox \
    --exp_code sr386_conch_aux0_aux1_aux2_p24_crosca0.2_step_gemini_nopre

# ============================================================
# 8. SR386 MDN 版本
#    主任务: mdn | 辅助任务: 3个
#    特点: 有 cross_scale_contrastive (weight=0.7)
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_SR386/ \
    --data_folder_s 10featsr/h5_files/ \
    --data_folder_l 20featsr/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task sr386 \
    --text_prompt_file $PROMPT_DIR/sr386_text_prompt_gemini_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task mdn --mdn_mode unified \
    --record_loss --lr $LR --reg $REG \
    --early_stopping_metric concordance_time_dependent \
    --mdn_sigma_min 0.01 \
    --scheduler_type $SCHEDULER \
    --prototype_number 12 \
    --use_cross_scale_contrastive 1 \
    --aux_label_cols aux_label_0,aux_label_1,aux_label_2 \
    --aux_task_losses ce,ce,ce \
    --cross_scale_weight 0.7 \
    --results_dir ./sr386_${LR},${REG}_mdn \
    --exp_code sr386_conch_aux0_aux1_aux2_p12_crosca0.7_step_gemini_nopre_mdn

# ============================================================
# 9. TJ (肝移植排斥分类, 3个辅助任务)
#    主任务: binary | 辅助任务: 3个
#    文本提示: tj_text_prompt_gpt_nopre.csv
#    特点: 有 cross_scale_contrastive (weight=0.2)
#    评估指标: bacc, auc (非c_index)
# ============================================================
python main_MTL.py \
    --data_root_dir $DATA_ROOT_TJ/ \
    --data_folder_s 10feattj/h5_files/ \
    --data_folder_l 20feattj/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task MTLtj \
    --text_prompt_file $PROMPT_DIR/tj_text_prompt_gpt_nopre.csv \
    --testing --batch_size $BATCH \
    --mil_aggregator $AGGREGATOR \
    --main_task binary --record_loss \
    --lr $LR --reg $REG \
    --scheduler_type $SCHEDULER \
    --aux_label_cols aux_label_0,aux_label_1,aux_label_2 \
    --aux_task_losses ce,ce,ce \
    --prototype_number 2 \
    --use_cross_scale_contrastive 1 \
    --cross_scale_weight 0.2 \
    --results_dir ./tj_${LR},${REG} \
    --exp_code tj_conch_aux0_aux1_aux2_p2_crosca0.2_step_gpt_nopre_bacc_auc

echo "=========================================="
echo "  全部实验完成"
echo "=========================================="
