# 1) public, dpsgd
# public
CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/wiki/canary/10/baseline_public `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models` 


# dpsgd
CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/wiki/canary/10/baseline_dpsgd `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`

        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/canary/10/high_entity_not_missed_public `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`





        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/canary/10/high_entity_not_missed_SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/canary/10/high_entity_not_missed_public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`




        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/canary/10/high_entity_missed_public `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`


        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/canary/10/high_entity_missed_SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/canary/10/high_entity_missed_public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`



## CRT
        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/canary/10/high_entity_CRT_not_missed `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4  `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`


CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/canary/10/high_entity_CRT_missed `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4  `#data_dir` \
wikitext2 `#task_mode` \
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
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`


############## amplification
########## wiki, first redaction
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/canary/10/amp_1st_glw `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
0.01 `#detection_error_rate` \
yes `#save_all_models` \
no `#use_different_canary` \
0 `#num_canary_to_mask` \
0.1 `#per_example_max_grad_norm` \
no `#lr_decay` \
glw `#accounting_mode`



########## wiki, second redaction
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/canary/10/amp_2nd_glw `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/canary/10/amp_1st_glw/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models` 

