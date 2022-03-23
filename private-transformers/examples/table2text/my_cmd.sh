# dart dataset
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/dart `#output_dir` \
table2text/prefix-tuning `#data_dir` \
"dart" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
no `#add_canary` \
no `#miss_canary` \
10 `#canary_times`

######################################################################## 
# public
######################################################################## 
# public training should have smaller batch size and smaller learning rate
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki/wiki_public `#output_dir` \
../../data `#data_dir` \
"wikitext2" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-5 `#learning_rate` \
1 `#gradient_accumulation_steps` 


######################################################################## 
# sanitization, 3.3%
######################################################################## 
# sanitization, 3.3%, not miss
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/wiki_3.3 `#output_dir` \
../../data `#data_dir` \
"wikitext2-delex-person" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-5 `#learning_rate` \
1 `#gradient_accumulation_steps` 

# sanitization, 3.3%, miss
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/wiki_3.3_miss `#output_dir` \
../../data `#data_dir` \
"wikitext2-delex-person" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-5 `#learning_rate` \
1 `#gradient_accumulation_steps` 


# sanitization, 11.3%, not miss
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/wiki_11.3 `#output_dir` \
../../data `#data_dir` \
"wikitext2-delex-medium" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
no `#miss_canary` \
10 `#canary_times` \
5e-5 `#learning_rate` \
1 `#gradient_accumulation_steps` 

# sanitization, 3.3%, miss
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki/wiki_11.3_miss `#output_dir` \
../../data `#data_dir` \
"wikitext2-delex-medium" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
yes `#non_private` \
no `#is_sdp_finetune` \
3 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-5 `#learning_rate` \
1 `#gradient_accumulation_steps` 


# finetune, low
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki_person_sdp_finetune \
../../data \
"wikitext2" \
"table2text/output/wiki_person/checkpoint-2298" \
3 \
yes \
no \
yes

# finetune, medium
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki_medium_sdp_finetune \
../../data \
"wikitext2" \
"table2text/output/wiki_medium/checkpoint-1127" \
3 \
yes \
no \
yes

# finetune, high
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki_high_sdp_finetune \
../../data \
"wikitext2" \
"table2text/output/wiki_high/checkpoint-1125" \
3 \
yes \
no \
yes


######################################################################## 
# DPSGD
######################################################################## 
# 1024 batch size, 200 epochs ==> 400 updates, larger lr
CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki_dpsgd_1024_200 `#output_dir` \
../../data `#data_dir` \
"wikitext2" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-4 `#learning_rate` \
512 `#gradient_accumulation_steps` 


# 512 batch size, 200 epochs ==> 800 updates, larger lr
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki_dpsgd_512_200 `#output_dir` \
../../data `#data_dir` \
"wikitext2" `#task_mode` \
"gpt2" `#model_name_or_path` \
3 `#target_epsilon` \
yes `#ghost_clipping` \
no `#non_private` \
no `#is_sdp_finetune` \
200 `#num_train_epochs` \
yes `#add_canary` \
yes `#miss_canary` \
10 `#canary_times` \
5e-4 `#learning_rate` \
256 `#gradient_accumulation_steps` 


######################################################################## 
# abcd
######################################################################## 
# public
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/abcd_public \
../../data \
"wikitext2-abcd" \
"gpt2" \
3 \
yes \
yes \
no \
10


# sanitization, low
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/abcd_delex \
../../data \
"wikitext2-abcd-delex" \
"gpt2" \
3 \
yes \
yes \
no \
10

# dpsgd
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/abcd_dpsgd \
../../data \
"wikitext2-abcd" \
"gpt2" \
3 \
yes \
no \
no \
30


# finetune, abcd
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/abcd_sdp_finetune \
../../data \
"wikitext2-abcd" \
"table2text/output/abcd_delex/checkpoint-1033" \
3 \
yes \
no \
yes \
10
