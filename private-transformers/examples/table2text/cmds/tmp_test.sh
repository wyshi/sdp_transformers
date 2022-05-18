

CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=100/public `#output_dir` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
10 `#num_canary_to_mask`
