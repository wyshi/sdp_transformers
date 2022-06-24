####################################################
# sst-2, first
CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_high/glw_amplification/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-entity_only_high-3.01 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.018 --accounting_mode glw

# sst-2, second
CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_high/glw_amplification/second \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/GLUE-SST-2 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/sst-2/entity_only_high/glw_amplification/first/best \
--learning_rate 0.0005 

#########################################
# qqp, first
CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_high/glw_amplification/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QQP/QQP-entity_only_high-8.3 \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.012 --accounting_mode glw

# qqp, second
CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_high/glw_amplification/second_1e-4 \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/QQP \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qqp/entity_only_high/glw_amplification/first/best \
--learning_rate 0.0001


####################################################
# qnli, first
CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_high/glw_amplification/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QNLI/QNLI-entity_only_high-17.18 \
--delex_level entity_only_high \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.006 --accounting_mode glw

# qnli, second
CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_high/glw_amplification/second \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/QNLI \
--delex_level entity_only_high \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qnli/entity_only_high/glw_amplification/first/best \
--learning_rate 0.0005


#########################################
# mnli, first
CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_high/glw_amplification/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/MNLI/MNLI-entity_only_high-8.63 \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.011 --accounting_mode glw

# mnli, second
CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_high/glw_amplification/second_1e-4 \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/MNLI \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/mnli/entity_only_high/glw_amplification/first/best \
--learning_rate 0.0001

######################################################
######################################################
# sst-2, first
CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_high/glw_amplification_conservative_recall=50/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-entity_only_high-3.01 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.03 --accounting_mode glw

# sst-2, second
CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_high/glw_amplification_conservative_recall=50/second \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/GLUE-SST-2 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/sst-2/entity_only_high/glw_amplification_conservative_recall=50/first/best \
--learning_rate 0.0005

#########################################
# qqp, first
CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_high/glw_amplification_conservative_recall=50/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QQP/QQP-entity_only_high-8.3 \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.083 --accounting_mode glw

# qqp, second
CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_high/glw_amplification_conservative_recall=50/second_1e-4 \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/QQP \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qqp/entity_only_high/glw_amplification_conservative_recall=50/first/best \
--learning_rate 0.0001


######################################################
CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_high/glw_amplification_conservative_recall=50/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QNLI/QNLI-entity_only_high-17.18 \
--delex_level entity_only_high \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.172 --accounting_mode glw

# qnli, second
CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_high/glw_amplification_conservative_recall=50/second \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/QNLI \
--delex_level entity_only_high \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qnli/entity_only_high/glw_amplification_conservative_recall=50/first/best \
--learning_rate 0.0005


##########
# mnli, first
CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_high/glw_amplification_conservative_recall=50/first \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/MNLI/MNLI-entity_only_high-8.63 \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 0.5 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--detection_error_rate 0.086 --accounting_mode glw

# mnli, second
CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_high/glw_amplification_conservative_recall=50/second_1e-4 \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/original/MNLI \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/mnli/entity_only_high/glw_amplification_conservative_recall=50/first/best \
--learning_rate 0.0001