#dpsgd, low-resource + low_resource
CUDA_VISIBLE_DEVICES=2 python -m classification.run_wrapper \
--output_dir classification/output/qnli/sample100/not_missed/DPSGD \
--data_dir classification/data/normalized_mask/QNLI/QNLI/original-sample100 \
--delex_level no \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 0.0005 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500

#sdp, low-resource + low_resource
CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/qnli/sample100/not_missed/SDP \
--data_dir classification/data/normalized_mask/QNLI/QNLI/original-sample100 \
--delex_level no \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qnli/sample100/not_missed/public_notcleaned/best \
--learning_rate 0.0005 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500