        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/wiki/sample10/not_missed/public `#output_dir` \
../../data/sample10/wiki_entity_all_mask_consec-16.4-sample10 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
20 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
no `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/wiki/sample10/not_missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/sample10/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask`



CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/wiki/sample10/not_missed/SDP_low_resource `#output_dir` \
../../data/sample10/wikitext-2-raw-sample10 `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/sample10/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
1 `#gradient_accumulation_steps` \
no `#add_mask`
