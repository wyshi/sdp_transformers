# cuda0, sst-2, mnli, medium-entity
CUDA_VISIBLE_DEVICES=0 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_medium/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-entity_only_medium-1.79 \
--delex_level entity_only_medium \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

CUDA_VISIBLE_DEVICES=0 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_medium/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/MNLI/MNLI-entity_only_medium-6.09 \
--delex_level entity_only_medium \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

# cuda1, sst-2, mnli, high-entity
CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_high/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-entity_only_high-3.01 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_high/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/MNLI/MNLI-entity_only_high-8.63 \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005


# cuda2, sst-2, mnli, default
CUDA_VISIBLE_DEVICES=2 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/default/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-default-22.19 \
--delex_level default \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

CUDA_VISIBLE_DEVICES=2 python -m classification.run_wrapper \
--output_dir classification/output/mnli/default/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/MNLI/MNLI-default-31.19 \
--delex_level default \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

# cuda3, sst-2, mnli, SRL
CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/SRL/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-SRL-38.13 \
--delex_level SRL \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/mnli/SRL/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/MNLI/MNLI-SRL-44.27 \
--delex_level SRL \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

# cuda4, qnli, qqp, medium-entity
CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_medium/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QNLI/QNLI-entity_only_medium-12.19 \
--delex_level entity_only_medium \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_medium/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QQP/QQP-entity_only_medium-6.05 \
--delex_level entity_only_medium \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

# cuda5, qnli, qqp, high-entity
CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_high/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QNLI/QNLI-entity_only_high-17.18 \
--delex_level entity_only_high \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005


CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_high/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QQP/QQP-entity_only_high-8.3 \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005


# cuda6, qnli, qqp, default
CUDA_VISIBLE_DEVICES=6 python -m classification.run_wrapper \
--output_dir classification/output/qnli/default/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QNLI/QNLI-default-35.68 \
--delex_level default \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

CUDA_VISIBLE_DEVICES=6 python -m classification.run_wrapper \
--output_dir classification/output/qqp/default/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QQP/QQP-default-32.61 \
--delex_level default \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

# cuda7, qnli, qqp, SRL
CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qnli/SRL/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QNLI/QNLI-SRL-45.59 \
--delex_level SRL \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005

CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qqp/SRL/not_missed/CRT \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/QQP/QQP-SRL-45.93 \
--delex_level SRL \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005
