########## wiki, first redaction
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/entity_only_high/test_glw_amplification_15%_eps0.5_200epoches_grad_acc=256/public `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
0.15 `#detection_error_rate` \
no `#save_all_models` \
no `#use_different_canary` \
0 `#num_canary_to_mask` \
0.1 `#per_example_max_grad_norm` \
no `#lr_decay` \
glw `#accounting_mode`


########## wiki, second redaction
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/entity_only_high/test_glw_amplification_15%_eps0.5_200epoches_grad_acc=256/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/entity_only_high/test_glw_amplification_15%_eps0.5_200epoches_grad_acc=256/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
no `#save_all_models`



########## abcd, first redaction
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_glw_amplification_5%_eps0.5_200epoches_grad_acc=128/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask` \
0.05 `#detection_error_rate` \
no `#save_all_models` \
no `#use_different_canary` \
0 `#num_canary_to_mask` \
0.1 `#per_example_max_grad_norm` \
no `#lr_decay` \
glw `#accounting_mode`


########## abcd, second redaction
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_glw_amplification_5%_eps0.5_200epoches_grad_acc=128/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_high/test_glw_amplification_5%_eps0.5_200epoches_grad_acc=128/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
no `#save_all_models`





################## wiki, exact estimation
########## wiki, first redaction
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/entity_only_high/test_glw_amplification_1%_eps0.5_200epoches_grad_acc=256/public `#output_dir` \
../../data/wiki_entity_all_mask_consec-16.4 `#data_dir` \
wikitext2 `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
0.01 `#detection_error_rate` \
no `#save_all_models` \
no `#use_different_canary` \
0 `#num_canary_to_mask` \
0.1 `#per_example_max_grad_norm` \
no `#lr_decay` \
glw `#accounting_mode`


########## wiki, second redaction
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/entity_only_high/test_glw_amplification_1%_eps0.5_200epoches_grad_acc=256/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/entity_only_high/test_glw_amplification_1%_eps0.5_200epoches_grad_acc=256/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
5e-05 `#learning_rate` \
256 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
no `#save_all_models`



################## abcd, exact estimation
########## abcd, first redaction
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_glw_amplification_1%_eps0.5_200epoches_grad_acc=128/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask` \
0.01 `#detection_error_rate` \
no `#save_all_models` \
no `#use_different_canary` \
0 `#num_canary_to_mask` \
0.1 `#per_example_max_grad_norm` \
no `#lr_decay` \
glw `#accounting_mode`


########## abcd, second redaction
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_glw_amplification_1%_eps0.5_200epoches_grad_acc=128/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_high/test_glw_amplification_1%_eps0.5_200epoches_grad_acc=128/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
no `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
128 `#gradient_accumulation_steps` \
no `#add_mask` \
-1 `#detection_error_rate` \
no `#save_all_models`