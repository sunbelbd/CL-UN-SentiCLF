#!/usr/bin/bash

tgt_range=(fr de jp)
product_range=(music book dvd)
output_path=./logs/

for tgt in ${tgt_range[*]}
do
  for pdt in ${product_range[*]}
  do
    log_file="${output_path}/paddle-mvec-en-${tgt}-${pdt}.log"
    sbatch --job-name="pd$tgt$pdt" --nodes=1 --gres=gpu:1 --partition=V100x8 --wrap "srun -e ${log_file} bash runPaddle_${tgt}.sh --clf_steps 50 --num_gpu 1 --exp_name unsupMTDiscCLF_en${tgt} --data_category $pdt --train_dis True --train_encdec True --train_bt True --clf_atten False --clf_mtv True --tokens_per_batch 600"
    sleep 1
  done
done

