python /local-scratch1/data/wyshi/privacy/random_response/RRWithPrior.py -data /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/train_split_4/train_0.txt --pred_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage0 -m /local-scratch1/data/wyshi/privacy/pate/checkpoint/20220129/train10/clm_0 -d cuda:6 -b 1 -eps 3.3 -st 0


CUDA_VISIBLE_DEVICES=6 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/valid.txt --num_train_epochs 3 --train_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage0.txt --output_dir /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage0 --log_file /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage0.log


python /local-scratch1/data/wyshi/privacy/random_response/RRWithPrior.py -data /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/train_split_4/train_1.txt --pred_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage1 -m /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage0 -d cuda:6 -b 1 -eps 3.3 -st 1


CUDA_VISIBLE_DEVICES=2 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/valid.txt --num_train_epochs 3 --train_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage1.txt --output_dir /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage1 --log_file /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage1.log


python /local-scratch1/data/wyshi/privacy/random_response/RRWithPrior.py -data /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/train_split_4/train_2.txt --pred_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage2 -m /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage1 -d cuda:2 -b 1 -eps 3.3 -st 2


CUDA_VISIBLE_DEVICES=2 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/valid.txt --num_train_epochs 3 --train_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage2.txt --output_dir /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage2 --log_file /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage2.log

python /local-scratch1/data/wyshi/privacy/random_response/RRWithPrior.py -data /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/train_split_4/train_3.txt --pred_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage3 -m /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage2 -d cuda:2 -b 1 -eps 3.3 -st 3


# last stage
CUDA_VISIBLE_DEVICES=6 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/valid.txt --num_train_epochs 3 --train_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage3.txt --output_dir /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage3 --log_file /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_stage3.log





# random
CUDA_VISIBLE_DEVICES=2 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/valid.txt --num_train_epochs 3 --train_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/pred/pred_random_not_limit_to_digits.txt --output_dir /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_random --log_file /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_random.log



CUDA_VISIBLE_DEVICES=5 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/valid.txt --num_train_epochs 3 --train_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/train_split_4/train_0.txt --output_dir /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_0_no_noise --log_file /local-scratch1/data/wyshi/privacy/random_response/checkpoint/20220219/train_with_0_no_noise.log --block_size 512

