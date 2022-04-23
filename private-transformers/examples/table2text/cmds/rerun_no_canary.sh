############################################## cuda 0, public, dpsgd, abcd public, abcd dpsgd
        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/no/missed/public `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
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
table2text/output/wiki/dpsgd/missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
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
        # public: True, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/abcd/no/missed/public `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
no `#add_mask`

        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/abcd/dpsgd/missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
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



############################################### cuda 4, medium entity, dpsgd, abcd medium entity, abcd dpsgd, 
        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/wiki/entity_only_medium/not_missed/public `#output_dir` \
../../data/wiki_entity_person_org_date_gpe_mask_consec-11.3 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/wiki/entity_only_medium/not_missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/entity_only_medium/not_missed/public/best `#model_name_or_path` \
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
yes `#add_mask`

        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/abcd/entity_only_medium/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_medium_2.71568 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/abcd/entity_only_medium/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_medium/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask`


############################################### cuda 5, high entity

        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki/entity_only_high/not_missed/public `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki/entity_only_high/not_missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/entity_only_high/not_missed/public/best `#model_name_or_path` \
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
yes `#add_mask`

        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_high/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki/entity_only_person/not_missed/public `#output_dir` \
../../data/wiki_entity_person_mask_consec-3.3 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki/entity_only_person/not_missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/entity_only_person/not_missed/public/best `#model_name_or_path` \
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
yes `#add_mask`



        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/entity_only_person/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_low_1.47119 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/entity_only_person/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_person/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask`

############################################### cuda 6, default
        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/default/not_missed/public `#output_dir` \
../../data/wiki_contextual_default_mask_consec-34.8 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`







        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/default/not_missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/default/not_missed/public/best `#model_name_or_path` \
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
yes `#add_mask`






        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/abcd/default/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-default_22.26044 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
3e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`







        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/abcd/default/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/default/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask`


        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/no_pronoun/not_missed/public `#output_dir` \
../../data/wiki_contextual_no_pronoun-33.7 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/no_pronoun/not_missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/no_pronoun/not_missed/public/best `#model_name_or_path` \
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
yes `#add_mask`


##
        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/abcd/no_pronoun/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-no_pronoun_19.22277 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/abcd/no_pronoun/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/no_pronoun/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask`


############################################### cuda 7, SRL

        #######################################
        # data: wikitext2, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/wiki/SRL/not_missed/public `#output_dir` \
../../data/wiki_contextual_SRL_mask_consec-45.0 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/wiki/SRL/not_missed/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/SRL/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0005 `#learning_rate` \
256 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2-abcd, 
        # public: True, 
        # miss: False, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/abcd/SRL/not_missed/public `#output_dir` \
../../data/abcd/abcd_my_delex-SRL_28.6405 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-05 `#learning_rate` \
1 `#gradient_accumulation_steps` \
yes `#add_mask`



        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: True
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/abcd/SRL/not_missed/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/SRL/not_missed/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
0.0001 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask`


        #######################################
        # data: wikitext2, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/wiki/dpsgd/missed_addmask/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
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
0.0005 `#learning_rate` \
512 `#gradient_accumulation_steps` \
yes `#add_mask`

        #######################################
        # data: wikitext2-abcd, 
        # public: False, 
        # miss: True, 
        # is_sdp: False
        #######################################
        

CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/abcd/dpsgd/missed_addmask/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
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
0.0005 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask`
