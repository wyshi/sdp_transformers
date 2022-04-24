CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_amplification_1%_eps0.5/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` yes \
`#ghost_clipping` no \
`#non_private` no \
`#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask` \
0.01 `#detection_error_rate` \
yes `#save_all_models`


CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_amplification_10%_eps0.5_20epoches_grad_acc=32/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
20 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
32 `#gradient_accumulation_steps` \
yes `#add_mask` \
0.1 `#detection_error_rate` \
yes `#save_all_models`



CUDA_VISIBLE_DEVICES=7 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_amplification_10%_eps0.5_20epoches/SDP `#output_dir` \
../../data/abcd/abcd_original `#data_dir` \
wikitext2-abcd `#task_mode` \
table2text/output/abcd/entity_only_high/test_amplification_10%_eps0.5_20epoches/public/best `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
yes `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
128 `#gradient_accumulation_steps` \
yes `#add_mask` \
-1 `#detection_error_rate` \
yes `#save_all_models`


CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/abcd/entity_only_high/test_amplification_10%_eps0.5_5epoches_grad_acc=4/public `#output_dir` \
../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` \
wikitext2-abcd `#task_mode` \
gpt2 `#model_name_or_path` \
0.5 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
5 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
100 `#canary_times` \
0.0005 `#learning_rate` \
4 `#gradient_accumulation_steps` \
yes `#add_mask` \
0.1 `#detection_error_rate` \
yes `#save_all_models`