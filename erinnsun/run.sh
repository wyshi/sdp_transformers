#!/bin/bash
#SBATCH --job-name=63ghost
#SBATCH --output=zlogs/slurm.log
#SBATCH --error=zlogs/slurm.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -p rtx8000,v100
#SBATCH --mem=200000M
#SBATCH --time=40:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liangshentiaodafen@gmail.com

source $HOME/.bashrc

#mark=air_glw_lr3en5_g49r35
mark=air_glw_eps0p2dt6_lr3en5_g63
accounting_mode=glw
#mark=e2e_d1
output_dir="output/"$mark
log=logs/$mark
err=errs/$mark
data_dir="data/prefix-tuning"
task_mode="air"
model_name_or_path="gpt2" # One of distilgpt2, gpt2, gpt2-medium, gpt2-large
target_epsilon=0.2
cache_dir="/scratch/zt2080/tianzhiliang/bigsave_dpnlg/cache/"
ghost_clipping="yes" # Fill 'no' to turn this off.
non_private="no"

if [[ ${task_mode} == "e2e" ]]; then
  data_dir="${data_dir}/data/e2e_data"
  target_delta=8e-6
  num_train_epochs=10
  learning_rate=2e-3
  max_seq_len=100
else
  if [[ ${task_mode} == "dart" ]]; then
    target_delta=1e-5
    data_dir="${data_dir}/data/dart"
    num_train_epochs=15 # Approximately same number of updates.
    learning_rate=5e-4  # Lower learning rate for stability in large models.
    max_seq_len=120
  else
  if [[ ${task_mode} == "air" ]]; then
    target_delta=1e-6
    data_dir="${data_dir}/data/airv031_99w"
    num_train_epochs=10 # Approximately same number of updates.
    learning_rate=3e-5  # Lower learning rate for stability in large models.
    max_seq_len=40
  else
  if [[ ${task_mode} == "euro" ]]; then
    target_delta=1e-6
    data_dir="${data_dir}/data/euro172w_for_dpsgd"
    num_train_epochs=10 # Approximately same number of updates.
    learning_rate=3e-5  # Lower learning rate for stability in large models.
    max_seq_len=40
  fi
  fi
  fi
fi

# Arguments in the last two lines are the most important.
python -m run_language_modeling \
  --output_dir ${output_dir} --overwrite_output_dir --cache_dir $cache_dir \
  --task_mode ${task_mode} \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --do_train --do_eval \
  --line_by_line \
  --save_steps 100 --save_total_limit 1 --save_at_last no \
  --logging_dir ${output_dir} --logging_steps 1 \
  --seed 0 --accounting_mode $accounting_mode \
  --eval_steps 10 --eval_epochs 1 --max_eval_batches 100 --evaluation_strategy epoch --evaluate_before_training "yes" --evaluate_during_training "yes" --per_device_eval_batch_size 10 \
  --max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
  --max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
  --data_folder ${data_dir} --max_seq_len ${max_seq_len} --format_mode cat \
  --per_example_max_grad_norm 0.1 --target_delta ${target_delta} --target_epsilon ${target_epsilon} \
  --learning_rate ${learning_rate} --lr_decay "no" --num_train_epochs ${num_train_epochs} --per_device_train_batch_size 8 --gradient_accumulation_steps 256 \
  --non_private ${non_private} \
  --ghost_clipping ${ghost_clipping} > $log 2> $err
