# 1) public, dpsgd
# public
CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/baseline/public `#output_dir` \
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
0 `#num_canary_to_mask`


# dpsgd
CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/baseline/dpsgd `#output_dir` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`

# 2) redact + SDP, capture all canaries
# redact
CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
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


# SDP
CUDA_VISIBLE_DEVICES=4 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=100/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/10_diff_canary/recall=100/public/best `#model_name_or_path` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`


# 3) redact + SDP, mask 8 --> recall=80
# redact
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=80/public `#output_dir` \
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
8 `#num_canary_to_mask`


# SDP
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=80/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/10_diff_canary/recall=80/public/best `#model_name_or_path` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`


# 4) redact + SDP, mask 6 --> recall=60
# redact
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=60/public `#output_dir` \
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
6 `#num_canary_to_mask`


# SDP
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=60/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/10_diff_canary/recall=60/public/best `#model_name_or_path` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`


# 5) redact + SDP, mask 4 --> recall=40
# redact
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=40/public `#output_dir` \
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
4 `#num_canary_to_mask`


# SDP
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=40/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/10_diff_canary/recall=40/public/best `#model_name_or_path` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`




# 6) DP + SDP, mask 8 --> recall=80
# redact
CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=80_amp/first `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
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
2e-6 `#detection_error_rate` \
yes `#save_all_models` \
yes `#use_different_canary` \
8 `#num_canary_to_mask`


# SDP
CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=80_amp/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/10_diff_canary/recall=80_amp/first/best `#model_name_or_path` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`




# 7) DP + SDP, mask 6 --> recall=60
# redact
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=60_amp/first `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
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
5e-6 `#detection_error_rate` \
yes `#save_all_models` \
yes `#use_different_canary` \
6 `#num_canary_to_mask`


# SDP
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=60_amp/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/10_diff_canary/recall=60_amp/first/best `#model_name_or_path` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`


# 8) DP + SDP, mask 4 --> recall=40
# redact
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=40_amp/first `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
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
8e-6 `#detection_error_rate` \
yes `#save_all_models` \
yes `#use_different_canary` \
4 `#num_canary_to_mask`


# SDP
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki/10_diff_canary/recall=40_amp/SDP `#output_dir` \
../../data/wikitext-2-raw `#data_dir` \
wikitext2 `#task_mode` \
table2text/output/wiki/10_diff_canary/recall=40_amp/first/best `#model_name_or_path` \
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
yes `#save_all_models` \
yes `#use_different_canary` \
0 `#num_canary_to_mask`
