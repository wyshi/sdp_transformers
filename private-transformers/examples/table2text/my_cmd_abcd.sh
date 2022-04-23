        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/abcd/no/missed/public `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
10 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/abcd/dpsgd/missed/SDP_nomask `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/abcd/entity_only_medium/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_medium_2.71568 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
10 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/abcd/entity_only_medium/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_medium/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=3 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
10 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=3 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_high/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/abcd/default/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-default_22.26044 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
10 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/abcd/default/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/default/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/SRL/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-SRL_28.6405 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
10 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` 



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/SRL/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/SRL/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` 

