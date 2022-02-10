
# train student model
CUDA_VISIBLE_DEVICES=5 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /home/wyshi/privacy/data/wikitext-2/valid.txt --train_file /home/wyshi/privacy/data/wikitext-2/pred.txt --output_dir /home/wyshi/privacy/pate/checkpoint/20210117/train10/student1

CUDA_VISIBLE_DEVICES=5 python run_clm_no_trainer.py --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --validation_file /home/wyshi/privacy/data/wikitext-2/valid.txt --train_file /home/wyshi/privacy/data/wikitext-2/train.txt --output_dir /home/wyshi/privacy/pate/checkpoint/20210117/train10/nopate