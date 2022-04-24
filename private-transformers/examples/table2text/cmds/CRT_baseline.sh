        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/CRT/default/DP `#output_dir` \
../../data/dedup_wiki_contextual_default_mask_consec-35.3 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask`


        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/CRT/SRL/DP `#output_dir` \
../../data/dedup_wiki_contextual_SRL_mask_consec-45.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask`

        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/CRT/entity_only_medium/DP `#output_dir` \
../../data/dedup_wiki_entity_person_org_date_gpe_mask_consec-11.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask`

        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=3 bash table2text/run.sh \
table2text/output/wiki/CRT/entity_only_high/DP `#output_dir` \
../../data/dedup_wiki_entity_all_mask_consec-16.6 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask`


        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/abcd/CRT/default/DP `#output_dir` \
../../data/abcd/abcd_my_delex-default_22.26044 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask`


        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/CRT/SRL/DP `#output_dir` \
../../data/abcd/abcd_my_delex-SRL_28.6405 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask`


        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/abcd/CRT/entity_only_medium/DP `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_medium_2.71568 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask`


        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/abcd/CRT/entity_only_high/DP `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask`



