
#####################################################################
# screen 3, cuda 1
#####################################################################
        #######################################
        # data: sst-2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/SRL/not_missed/public \
--delex_level SRL \
--task_name sst-2 \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/SRL/not_missed/SDP \
--delex_level SRL \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/sst-2/SRL/not_missed/public/best \
--learning_rate 0.0005

        #######################################
        # data: mnli, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/mnli/SRL/not_missed/public \
--delex_level SRL \
--task_name mnli \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: mnli, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
--output_dir classification/output/mnli/SRL/not_missed/SDP \
--delex_level SRL \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/mnli/SRL/not_missed/public/best \
--learning_rate 0.0005



#####################################################################
# screen 4, cuda 3
#####################################################################

        #######################################
        # data: sst-2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_high/not_missed/public \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_high/not_missed/SDP \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/sst-2/entity_only_high/not_missed/public/best \
--learning_rate 0.0005


        #######################################
        # data: mnli, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_high/not_missed/public \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: mnli, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_high/not_missed/SDP \
--delex_level entity_only_high \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/mnli/entity_only_high/not_missed/public/best \
--learning_rate 0.0005



#####################################################################
# screen 5, cuda 4
#####################################################################
        #######################################
        # data: sst-2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_medium/not_missed/public \
--delex_level entity_only_medium \
--task_name sst-2 \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: sst-2, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/sst-2/entity_only_medium/not_missed/SDP \
--delex_level entity_only_medium \
--task_name sst-2 \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/sst-2/entity_only_medium/not_missed/public/best \
--learning_rate 0.0005


        #######################################
        # data: mnli, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_medium/not_missed/public \
--delex_level entity_only_medium \
--task_name mnli \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: mnli, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=4 python -m classification.run_wrapper \
--output_dir classification/output/mnli/entity_only_medium/not_missed/SDP \
--delex_level entity_only_medium \
--task_name mnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/mnli/entity_only_medium/not_missed/public/best \
--learning_rate 0.0005



#####################################################################
# screen 6, cuda 5
#####################################################################
        #######################################
        # data: qnli, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qnli/SRL/not_missed/public \
--delex_level SRL \
--task_name qnli \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: qnli, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qnli/SRL/not_missed/SDP \
--delex_level SRL \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qnli/SRL/not_missed/public/best \
--learning_rate 0.0005

        #######################################
        # data: qqp, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qqp/SRL/not_missed/public \
--delex_level SRL \
--task_name qqp \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: qqp, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=5 python -m classification.run_wrapper \
--output_dir classification/output/qqp/SRL/not_missed/SDP \
--delex_level SRL \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qqp/SRL/not_missed/public/best \
--learning_rate 0.0005






#####################################################################
# screen 7, cuda 6
#####################################################################
        #######################################
        # data: qnli, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=6 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_high/not_missed/public \
--delex_level entity_only_high \
--task_name qnli \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: qnli, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=6 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_high/not_missed/SDP \
--delex_level entity_only_high \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qnli/entity_only_high/not_missed/public/best \
--learning_rate 0.0005


        #######################################
        # data: qqp, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=6 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_high/not_missed/public \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: qqp, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=6 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_high/not_missed/SDP \
--delex_level entity_only_high \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qqp/entity_only_high/not_missed/public/best \
--learning_rate 0.0005





#####################################################################
# screen 8, cuda 7
#####################################################################
        #######################################
        # data: qnli, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_medium/not_missed/public \
--delex_level entity_only_medium \
--task_name qnli \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: qnli, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qnli/entity_only_medium/not_missed/SDP \
--delex_level entity_only_medium \
--task_name qnli \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qnli/entity_only_medium/not_missed/public/best \
--learning_rate 0.0005

        #######################################
        # data: qqp, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_medium/not_missed/public \
--delex_level entity_only_medium \
--task_name qqp \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05



        #######################################
        # data: qqp, 
        # public: False, 
        # miss: False, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=7 python -m classification.run_wrapper \
--output_dir classification/output/qqp/entity_only_medium/not_missed/SDP \
--delex_level entity_only_medium \
--task_name qqp \
--max_seq_len 256 \
--non_private no \
--target_epsilon 3 \
--is_sdp_finetune yes \
--model_name_or_path classification/output/qqp/entity_only_medium/not_missed/public/best \
--learning_rate 0.0005




