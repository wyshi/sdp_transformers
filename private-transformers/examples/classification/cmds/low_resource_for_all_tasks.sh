# sst-2
        #######################################
        # data: sst-2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################



CUDA_VISIBLE_DEVICES=0 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/sample100/not_missed/public \
--data_dir classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-sample100/GLUE-SST-2-entity_only_high-3.01-sample100 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500



        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=0 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/sample100/not_missed/SDP \
--data_dir classification/data/original/GLUE-SST-2 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/sst-2/sample100/not_missed/public/best \
--learning_rate 0.0005


        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        # low resource
        #######################################


CUDA_VISIBLE_DEVICES=0 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/sample100/not_missed/SDP_low_resource \
--data_dir classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-sample100/original-sample100 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/sst-2/sample100/not_missed/public/best \
--learning_rate 0.0005 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500


# QQP
        #######################################
        # data: qqp, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################



CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/qqp/sample100/not_missed/public \
--data_dir classification/data/normalized_mask/QQP/QQP-entity_only_high-8.3-sample301/ \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500



        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/qqp/sample100/not_missed/SDP \
--data_dir classification/data/original/QQP \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qqp/sample100/not_missed/public/best \
--learning_rate 0.0005


        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        # low resource
        #######################################


CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/qqp/sample100/not_missed/SDP_low_resource \
--data_dir classification/data/normalized_mask/QQP/QQP-sample100/original-sample100 \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qqp/sample100/not_missed/public/best \
--learning_rate 0.0005 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500



# sst-2
        #######################################
        # data: sst-2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################



CUDA_VISIBLE_DEVICES=2 python -m classification.run_wrapper \
--output_dir classification/output/mnli/sample100/not_missed/public \
--data_dir classification/data/normalized_mask/MNLI/MNLI-entity_only_high-8.63-sample301 \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500



        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=2 python -m classification.run_wrapper \
--output_dir classification/output/mnli/sample100/not_missed/SDP \
--data_dir classification/data/original/MNLI \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/mnli/sample100/not_missed/public/best \
--learning_rate 0.0005


        #######################################
        # data: mnli, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        # low resource
        #######################################


CUDA_VISIBLE_DEVICES=2 python -m classification.run_wrapper \
--output_dir classification/output/mnli/sample100/not_missed/SDP_low_resource \
--data_dir classification/data/normalized_mask/MNLI/MNLI-sample100/original-sample100 \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/mnli/sample100/not_missed/public/best \
--learning_rate 0.0005 \
--per_device_train_batch_size 20 \
--batch_size 80 \
--num_train_epochs 500




