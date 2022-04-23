
        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/canary/100/public `#output_dir` \
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
100 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
no `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/canary/100/dpsgd `#output_dir` \
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
100 `#canary_times` \
0.0001 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask`





        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/canary/100/default_not_missed_public `#output_dir` \
../../data/wiki_contextual_default_mask_consec-34.8 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
100 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`




        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/canary/100/default_not_missed_SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/canary/100/default_not_missed_public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0001 `#learning_rate` \
256 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/canary/100/default_missed_public `#output_dir` \
../../data/wiki_contextual_default_mask_consec-34.8 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`

        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/canary/100/default_missed_SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/canary/100/default_missed_public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0001 `#learning_rate` \
256 `#gradient_accumulation_steps` \
yes `#add_mask`
