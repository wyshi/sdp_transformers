# public
CUDA_VISIBLE_DEVICES=0 python -m classification.run_wrapper \
--output_dir classification/output/abcd_original_public_roberta \
--task_name abcd \
--data_dir /local/data/wyshi/sdp_transformers/data/abcd/abcd_classification_original/ \
--max_seq_len 256 \
--per_device_train_batch_size 16 \
--non_private yes \
--model_name_or_path roberta-base \
--num_train_epochs 15 

# dpsgd
CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/abcd_original_dpsgd \
--task_name abcd \
--data_dir /local/data/wyshi/sdp_transformers/data/abcd/abcd_classification_original/ \
--max_seq_len 256 \
--per_device_train_batch_size 16 \
--non_private no \
--model_name_or_path roberta-base \
--num_train_epochs 40 \
--target_epsilon 3 \
--learning_rate 5e-4

# sanitization
CUDA_VISIBLE_DEVICES=2 python -m classification.run_wrapper \
--output_dir classification/output/abcd_sanitization \
--task_name abcd \
--data_dir /local/data/wyshi/sdp_transformers/data/abcd/abcd_classification_delex \
--max_seq_len 256 \
--per_device_train_batch_size 16 \
--non_private yes \
--model_name_or_path roberta-base \
--num_train_epochs 15

# finetune
CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/abcd_sdp_finetune \
--task_name abcd \
--data_dir /local/data/wyshi/sdp_transformers/data/abcd/abcd_classification_original/ \
--max_seq_len 256 \
--per_device_train_batch_size 16 \
--non_private yes \
--model_name_or_path /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/output/abcd_original_public_roberta \
--num_train_epochs 15 


# sst-2
# dpsgd
CUDA_VISIBLE_DEVICES=6 python -m classification.run_wrapper \
--output_dir classification/output/sst2_dpsgd \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--model_name_or_path roberta-base \
--target_epsilon 3 \
--learning_rate 5e-4